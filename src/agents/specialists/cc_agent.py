from typing import Any

import httpx
from langchain.agents import create_agent
from langchain.agents.structured_output import ProviderStrategy
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from src.models.cc import CCAgentGraphState, CCBeginSpec, CCMCPOutput, CCRecommendParams
from src.models.operation import OperationInterruptPayload, OperationResumePayload
from src.utils.logging_config import logger
from src.utils.PROMPT import CC_AGENT_PROMPT
from src.utils.messages import MsgUtils
from src.utils.models import CC_MODEL
from src.utils.tools import coerce_operation_resume


def get_recommended_params(spec: CCBeginSpec) -> CCRecommendParams:
    """Get the recommended CC parameters based on the spec."""
    host = "52.83.119.132"
    url = f"http://{host}:8000/api/column-choice"

    payload: dict[str, Any] = {}
    if spec.sample_amount:
        payload["sample_amount"] = spec.sample_amount
    if spec.tlc_json_path:
        payload["tlc_json_path"] = spec.tlc_json_path
    if spec.tlc_data_json_path:
        payload["tlc_data_json_path"] = spec.tlc_data_json_path
    if spec.column_size:
        payload["column_size"] = spec.column_size
    try:
        with httpx.Client(timeout=10.0) as client:
            resp = client.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()["result"]
            parsed = CCRecommendParams.model_validate(data)
    except Exception:
        logger.exception("Failed to get recommended CC parameters from MCP server")
        raise
    else:
        logger.info(f"Recommended CC parameters: {parsed}")
    return parsed


class CCAgent:
    """CC agent implemented as a LangGraph subgraph."""

    def __init__(self, *, with_checkpointer: bool = False) -> None:
        """
        Build and compile the internal LangGraph subgraph.

        Args:
            with_checkpointer: If True, compiles with MemorySaver for standalone usage with HITL.
                               If False (default), parent graph's checkpointer will be used.

        """
        logger.info("CCAgent initialized with model={}", CC_MODEL)

        subgraph = StateGraph(CCAgentGraphState)

        subgraph.add_node("extract_spec", self._extract_spec)
        subgraph.add_node("interrupt_1", self._interrupt_1)
        subgraph.add_node("free_timepoint_1", self._free_timepoint_1)
        subgraph.add_node("fetch_params", self._fetch_params)
        subgraph.add_node("interrupt_2", self._interrupt_2)
        subgraph.add_node("free_timepoint_2", self._free_timepoint_2)

        subgraph.add_edge(START, "extract_spec")
        subgraph.add_edge("extract_spec", "free_timepoint_1")
        # subgraph.add_conditional_edges(
        #     "interrupt_1",
        #     self._route_interrupt_1,
        #     {
        #         "reject": "interrupt_1",
        #         "accept": "free_timepoint_1",
        #     },
        # )
        subgraph.add_conditional_edges( # TODO: 确认下这里逻辑怎么处理，能否根据前端点击confirm还是输入其他文本来判断
            "free_timepoint_1",
            self._route_free_timepoint_1,
            {
                "revise": "free_timepoint_1",
                "confirm": "fetch_params",
            }
        )
        subgraph.add_edge("fetch_params", "free_timepoint_2")
        # subgraph.add_conditional_edges(
        #     "interrupt_2",
        #     self._route_interrupt_2,
        #     {
        #         "reject": "interrupt_2",
        #         "accept": "free_timepoint_2",
        #     }
        # )
        subgraph.add_conditional_edges(
            "free_timepoint_2",
            self._route_free_timepoint_2,
            {
                "revise": "free_timepoint_2",
                "confirm": END,
            }
        )

        checkpointer = MemorySaver() if with_checkpointer else None
        
        self.compiled = subgraph.compile(checkpointer=checkpointer)
        self._agent = create_agent(
            model=CC_MODEL,
            system_prompt=CC_AGENT_PROMPT,
        )

    @staticmethod
    def _route_interrupt_1(state: CCAgentGraphState) -> str:
        return "accept" if state.user_approved_1 else "reject"

    @staticmethod
    def _route_free_timepoint_1(state: CCAgentGraphState) -> str:
        return "confirm" if state.user_confirmed_1 else "revise"
    
    @staticmethod
    def _route_interrupt_2(state: CCAgentGraphState) -> str:
        return "accept" if state.user_approved_2 else "reject"

    @staticmethod
    def _route_free_timepoint_2(state: CCAgentGraphState) -> str:
        return "confirm" if state.user_confirmed_2 else "revise"

    @staticmethod
    def _extract_spec(state: CCAgentGraphState) -> dict[str, Any]:
        spec = CCBeginSpec(
            sample_amount=0.1,
            tlc_json_path="data/tlc_result.json",
            tlc_data_json_path="data/tlc_data.json",
            column_size="12g",
        )
        # TODO: formulate the spec based on tlc result
        return {"payload": spec}

    @staticmethod
    def _interrupt_1(state: CCAgentGraphState) -> dict[str, Any]:
        """
        Approve/Reject 中断节点，支持用户编辑表单。
        """
        messages = state.messages

        
        # 构建中断 payload
        interrupt_payload = OperationInterruptPayload(
            message="请确认预填的 CC 表格内容，或修改后重新提交。",
            args={
                "cc": {"spec": state.payload.model_dump(mode="json") if state.payload else None},
                "mode": "approve_reject",
                "editable": True,  # 告诉前端表单可编辑
            },
        )

        # 触发中断
        raw = interrupt(interrupt_payload.model_dump(mode="json"))
        resume = coerce_operation_resume(raw)

        # 基础更新
        updates: dict[str, Any] = {
            "user_approved_1": bool(resume.approval),
        }

        if resume.approval:
            # 批准：检查是否有编辑后的数据
            if resume.data and isinstance(resume.data, dict):
                edited_spec = resume.data.get("spec")
                if edited_spec:
                    try:
                        new_payload = CCBeginSpec.model_validate(edited_spec)
                        updates["payload"] = new_payload
                        logger.info("Applied user-edited spec on approval")
                    except Exception:
                        logger.warning("Failed to parse edited spec, using original")
        else:
            # 拒绝：处理 comment 和可能的编辑数据
            comment_text = (resume.comment or "").strip()
            if comment_text:
                messages = MsgUtils.append_user_message(messages, comment_text)
                updates["messages"] = messages

            # 如果用户编辑了数据，可以选择应用或忽略
            if resume.data and isinstance(resume.data, dict):
                edited_spec = resume.data.get("spec")
                if edited_spec:
                    try:
                        new_payload = CCBeginSpec.model_validate(edited_spec)
                        updates["payload"] = new_payload
                        logger.info("Applied user-edited spec on rejection")
                    except Exception:
                        logger.warning("Failed to parse edited spec from resume.data")

        return updates

    def _free_timepoint_1(self, state: CCAgentGraphState) -> dict[str, Any]:
        """
        自由对话节点：
        - 用户点击 confirm → user_confirmed_1 = True，进入下一阶段
        - 用户继续对话 → user_confirmed_1 = False，循环回本节点
        """
        messages = state.messages
        payload = state.payload

        # 构建中断 payload
        interrupt_payload = OperationInterruptPayload(
            message="请确认当前配置，或继续对话进行调整。",
            args={
                "cc": {"spec": payload.model_dump(mode="json") if payload else None},
                "mode": "free_chat_with_confirm",
            },
        )

        # 中断并等待用户响应
        raw = interrupt(interrupt_payload.model_dump(mode="json"))
        resume: OperationResumePayload = coerce_operation_resume(raw)
            # -approval -comment -data

        # 用户点击了 confirm 按钮
        if resume.approval:
            return {"user_confirmed_1": True}
        
        # 用户继续对话
        updates: dict[str, Any] = {"user_confirmed_1": False}

        # 处理用户编辑的数据（如果有）
        if resume.data and isinstance(resume.data, dict):
            edited_spec = resume.data.get("spec")
            if edited_spec:
                try:
                    new_payload = CCBeginSpec.model_validate(edited_spec)
                    updates["payload"] = new_payload
                    payload = new_payload  # 用于后续 LLM 上下文
                except Exception:
                    logger.warning("Failed to parse edited spec from resume.data")

        # 处理用户消息
        user_text = (resume.comment or "").strip()
        if user_text:
            messages = MsgUtils.append_user_message(messages, user_text)

            # 调用 LLM 处理对话（可选）
            result = self._agent.invoke({"messages": [*messages]})
            
            # 如果 LLM 返回了更新的 spec，应用它
            if hasattr(result, "structured_response"):
                # TODO：根据你的 LLM 输出结构处理
                pass
            
            # 将 AI 回复加入消息历史
            # create_agent 返回 {"messages": [...]} 结构
            if "messages" in result and result["messages"]:
                last_msg = result["messages"][-1]
                ai_content = getattr(last_msg, "content", str(last_msg))
                messages = MsgUtils.append_response(messages, ai_content)

        updates["messages"] = messages
        return updates

    @staticmethod
    def _fetch_params(state: CCAgentGraphState) -> dict[str, Any]:
        """
        调用 MCP 获取推荐的 CC 参数。
        
        - 成功：payload 替换为 CCRecommendParams
        - 失败：抛出异常（或可改为返回错误状态）
        """
        if not isinstance(state.payload, CCBeginSpec):
            raise TypeError("Payload must be a CCBeginSpec for fetching params")

        messages = state.messages

        try:
            result = CCRecommendParams(
                silica_amount=10.5,
                column_size="12g",
                flow_rate=30.0,
                solvent_system="PE/EA",
                start_solvent_ratio="30:1",
                end_solvent_ratio="5:1",
                estimated_time=45.0,
                complex_tlc=False,
                column_volume=15.0,
                air_purge_time=2.0
            )
            # TODO: result = get_recommended_params(state.payload)
            logger.info(f"CC params fetched: {result}")

            # 添加系统消息记录
            messages = MsgUtils.append_thinking(
                messages,
                f"[CC] 根据 spec 获取到推荐参数:\n"
                f"  - 溶剂系统: {getattr(result, 'solvent_system', 'N/A')}\n"
                f"  - 比例: {getattr(result, 'ratio', 'N/A')}"
            )

            return {
                "payload": result,
                "messages": messages,
            }

        except Exception as e:
            logger.exception("Failed to fetch CC recommended params")
            raise

    @staticmethod
    def _interrupt_2(state: CCAgentGraphState) -> dict[str, Any]:
        """
        Approve/Reject 中断节点，支持用户编辑表单。
        """
        messages = state.messages

        interrupt_payload = OperationInterruptPayload(
            message="请确认 CC 推荐参数，或修改后重新提交。",
            args={
                "cc": {"params": state.payload.model_dump(mode="json") if state.payload else None},
                "mode": "approve_reject",
                "editable": True,  # 告诉前端表单可编辑
            },
        )

        # 触发中断
        raw = interrupt(interrupt_payload.model_dump(mode="json"))
        resume: OperationResumePayload = coerce_operation_resume(raw)

        # 基础更新
        updates: dict[str, Any] = {
            "user_approved_2": bool(resume.approval),
        }

        if resume.approval:
            # 批准：检查是否有编辑后的数据
            if resume.data and isinstance(resume.data, dict):
                edited_params = resume.data.get("params")
                if edited_params:
                    try:
                        new_payload = CCRecommendParams.model_validate(edited_params)
                        updates["payload"] = new_payload
                        logger.info("Applied user-edited params on approval")
                    except Exception:
                        logger.warning("Failed to parse edited params, using original")
        else:
            # 拒绝：处理 comment 和可能的编辑数据
            comment_text = (resume.comment or "").strip()
            if comment_text:
                messages = MsgUtils.append_user_message(messages, comment_text)
                updates["messages"] = messages

            # 如果用户编辑了数据，可以选择应用或忽略
            if resume.data and isinstance(resume.data, dict):
                edited_params = resume.data.get("params")
                if edited_params:
                    try:
                        new_payload = CCRecommendParams.model_validate(edited_params)
                        updates["payload"] = new_payload
                        logger.info("Applied user-edited params on rejection")
                    except Exception:
                        logger.warning("Failed to parse edited params from resume.data")

        return updates

    def _free_timepoint_2(self, state: CCAgentGraphState) -> dict[str, Any]:
        """
        自由对话节点：
        - 用户点击 confirm → user_confirmed_2 = True，进入下一阶段
        - 用户继续对话 → user_confirmed_2 = False，循环回本节点
        """
        messages = state.messages
        payload = state.payload

        # 构建中断 payload
        interrupt_payload = OperationInterruptPayload(
            message="请确认当前配置，或继续对话进行调整。",
            args={
                "cc": {"params": payload.model_dump(mode="json") if payload else None},
                "mode": "free_chat_with_confirm",
            },
        )

        # 中断并等待用户响应
        raw = interrupt(interrupt_payload.model_dump(mode="json"))
        resume: OperationResumePayload = coerce_operation_resume(raw)

        # 用户点击了 confirm 按钮
        if resume.approval:
            return {"user_confirmed_2": True}
        
        # 用户继续对话
        updates: dict[str, Any] = {"user_confirmed_2": False}

        # 处理用户编辑的数据（如果有）
        if resume.data and isinstance(resume.data, dict):
            edited_params = resume.data.get("params")
            if edited_params:
                try:
                    new_payload = CCRecommendParams.model_validate(edited_params)
                    updates["payload"] = new_payload
                    payload = new_payload  # 用于后续 LLM 上下文
                except Exception:
                    logger.warning("Failed to parse edited params from resume.data")

        # 处理用户消息
        user_text = (resume.comment or "").strip()
        if user_text:
            messages = MsgUtils.append_user_message(messages, user_text)

            # 调用 LLM 处理对话（可选）
            result = self._agent.invoke({"messages": [*messages]})
            
            # 如果 LLM 返回了更新的 spec，应用它
            if hasattr(result, "structured_response"):
                # TODO：根据你的 LLM 输出结构处理
                pass
            
            # 将 AI 回复加入消息历史
            # create_agent 返回 {"messages": [...]} 结构
            if "messages" in result and result["messages"]:
                last_msg = result["messages"][-1]
                ai_content = getattr(last_msg, "content", str(last_msg))
                messages = MsgUtils.append_response(messages, ai_content)

        updates["messages"] = messages
        return updates


if __name__ == "__main__":
    from pathlib import Path
    import uuid
    from langchain_core.messages import HumanMessage
    from langchain_core.runnables.config import RunnableConfig
    from src.utils.tools import terminal_approval_handler

    # Step 1: 创建 agent
    agent = CCAgent(with_checkpointer=True)

    # Step 2: 生成 workflow 图
    output_path = Path(__file__).resolve().parents[3] / "assets" / "cc_agent_workflow.png"
    agent.compiled.get_graph(xray=True).draw_mermaid_png(output_file_path=str(output_path))
    logger.success(f"Workflow exported to {output_path}")

    # Step 3: 创建会话配置
    thread_id = str(uuid.uuid4())
    config: RunnableConfig = RunnableConfig(configurable={"thread_id": thread_id})

    # Step 4: 初始化输入
    text = input("[user]: ").strip() or "帮我做柱层析分离"
    next_input = CCAgentGraphState(
        messages=[HumanMessage(content=text)],
    )

    # Step 5: 运行循环
    while True:
        interrupted = False
        
        for event in agent.compiled.stream(next_input, config=config, stream_mode="values"):
            if "__interrupt__" in event:
                interrupted = True
                print("\n" + "=" * 50)
                print("[INTERRUPT] 等待用户确认...")
                print("=" * 50)
                
                # 显示中断信息（Interrupt 对象用 .value 访问数据）
                interrupt_data = event["__interrupt__"]
                if interrupt_data:
                    for item in interrupt_data:
                        # item 是 Interrupt 对象，用 .value 获取实际数据
                        payload = item.value if hasattr(item, 'value') else item
                        if isinstance(payload, dict):
                            print(f"Message: {payload.get('message', 'N/A')}")
                            print(f"Args: {payload.get('args', {})}")
                        else:
                            print(f"Payload: {payload}")
                
                # 终端交互处理
                next_input = terminal_approval_handler(event)
                break
            
            # 打印当前状态
            print("\n" + "-" * 40)
            if "messages" in event:
                for msg in event["messages"][-3:]:  # 只显示最后3条消息
                    print(f"[{type(msg).__name__}]: {getattr(msg, 'content', str(msg))[:200]}")
            if "payload" in event and event["payload"]:
                print(f"[Payload]: {event['payload']}")
            print("-" * 40)

        if not interrupted:
            print("\n" + "=" * 50)
            print("Graph reached END")
            print("=" * 50)
            break