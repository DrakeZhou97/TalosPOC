from typing import Any

import httpx
from langchain.agents import create_agent
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from src.models.enums import REPhase
from src.models.re import REAgentGraphState, REBeginSpec, RERecommendParams, RESolvent
from src.models.operation import OperationInterruptPayload, OperationResumePayload
from src.utils.logging_config import logger
from src.utils.PROMPT import RE_AGENT_PROMPT
from src.utils.messages import MsgUtils
from src.utils.models import RE_MODEL
from src.utils.tools import coerce_operation_resume


def get_recommended_params(spec: REBeginSpec) -> RERecommendParams:
    """Get the recommended RE parameters based on the spec."""
    host = "52.83.119.132"
    url = f"http://{host}:8000/api/rotary-evaporator"

    payload: dict[str, Any] = {}
    if spec.solvent:
        payload["solvent"] = spec.solvent.model_dump(mode="json")
    if spec.volume:
        payload["volume"] = spec.volume
    try:
        with httpx.Client(timeout=10.0) as client:
            resp = client.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()["result"]
            parsed = RERecommendParams.model_validate(data)
    except Exception:
        logger.exception("Failed to get recommended RE parameters from MCP server")
        raise
    else:
        logger.info(f"Recommended RE parameters: {parsed}")
    return parsed


class REAgent:
    """RE agent implemented as a LangGraph subgraph."""

    def __init__(self, *, with_checkpointer: bool = False) -> None:
        """
        Build and compile the internal LangGraph subgraph.

        Args:
            with_checkpointer: If True, compiles with MemorySaver for standalone usage with HITL.
                               If False (default), parent graph's checkpointer will be used.

        """
        logger.info("REAgent initialized with model={}", RE_MODEL)

        subgraph = StateGraph(REAgentGraphState)

        subgraph.add_node("extract_spec", self._extract_spec)
        subgraph.add_node("free_timepoint_spec", self._free_timepoint_spec)
        subgraph.add_node("fetch_params", self._fetch_params)
        subgraph.add_node("free_timepoint_params", self._free_timepoint_params)

        subgraph.add_edge(START, "extract_spec")
        subgraph.add_edge("extract_spec", "free_timepoint_spec")
        subgraph.add_conditional_edges( # TODO: 确认下这里逻辑怎么处理，能否根据前端点击confirm还是输入其他文本来判断
            "free_timepoint_spec",
            self._route_free_timepoint_spec,
            {
                "revise": "free_timepoint_spec",
                "confirm": "fetch_params",
            }
        )
        subgraph.add_edge("fetch_params", "free_timepoint_params")
        subgraph.add_conditional_edges(
            "free_timepoint_params",
            self._route_free_timepoint_params,
            {
                "revise": "free_timepoint_params",
                "confirm": END,
            }
        )

        checkpointer = MemorySaver() if with_checkpointer else None
        
        self.compiled = subgraph.compile(checkpointer=checkpointer)
        self._agent = create_agent(
            model=RE_MODEL,
            system_prompt=RE_AGENT_PROMPT,
        )

    @staticmethod
    def _route_free_timepoint_spec(state: REAgentGraphState) -> str:
        return "confirm" if state.re.phase == REPhase.SPEC_CONFIRMED else "revise"

    @staticmethod
    def _route_free_timepoint_params(state: REAgentGraphState) -> str:
        return "confirm" if state.re.phase == REPhase.PARAMS_CONFIRMED else "revise"

    @staticmethod
    def _extract_spec(state: REAgentGraphState) -> dict[str, Any]:
        spec = REBeginSpec(
            solvent=RESolvent(
                iupac_name="ethyl acetate",
                smiles="CC(=O)OCC",
            ),
            volume=100.0,
        )
        return {"re": state.re.model_copy(update={"payload": spec})}

    def _free_timepoint_spec(self, state: REAgentGraphState) -> dict[str, Any]:
        """
        自由对话节点：
        - 用户点击 confirm → re.spec_confirmed = True，进入下一阶段
        - 用户继续对话 → re.spec_confirmed = False，循环回本节点
        """
        messages = state.messages
        payload = state.re.payload

        # 构建中断 payload
        interrupt_payload = OperationInterruptPayload(
            message="请确认当前配置，或继续对话进行调整。",
            args={
                "re": {"spec": payload.model_dump(mode="json") if payload else None},
                "mode": "free_chat_with_confirm",
            },
        )

        # 中断并等待用户响应
        raw = interrupt(interrupt_payload.model_dump(mode="json"))
        resume: OperationResumePayload = coerce_operation_resume(raw)
            # -approval -comment -data

        # 用户点击了 confirm 按钮
        if resume.approval:
            return {"re": state.re.model_copy(update={"phase": REPhase.SPEC_CONFIRMED})}
        
        # 用户继续对话
        updates: dict[str, Any] = {"re": state.re.model_copy(update={"phase": REPhase.COLLECTING})}

        # 处理用户编辑的数据（如果有）
        if resume.data and isinstance(resume.data, dict):
            edited_spec = resume.data.get("spec")
            if edited_spec:
                try:
                    new_payload = REBeginSpec.model_validate(edited_spec)
                    updates["re"] = state.re.model_copy(update={"payload": new_payload})
                    payload = new_payload  # 用于后续 LLM 上下文
                except Exception:
                    logger.warning("Failed to parse edited spec from resume.data")

        # 处理用户消息（仅在拒绝/继续对话时有效）
        user_text = (resume.comment or "").strip()
        if user_text:
            messages = MsgUtils.append_user_message(messages, user_text)

            # 调用 LLM 处理对话
            result = self._agent.invoke({"messages": [*messages]})
            
            # 如果 LLM 返回了更新的 spec，应用它
            if hasattr(result, "structured_response"):
                # TODO：根据你的 LLM 输出结构处理
                pass
            
            # 将 AI 回复加入消息历史
            if "messages" in result and result["messages"]:
                last_msg = result["messages"][-1]
                ai_content = getattr(last_msg, "content", str(last_msg))
                messages = MsgUtils.append_response(messages, ai_content)

        updates["messages"] = messages
        return updates

    @staticmethod
    def _fetch_params(state: REAgentGraphState) -> dict[str, Any]:
        """
        调用 MCP 获取推荐的 RE 参数。
        
        - 成功：re.payload 替换为 RERecommendParams
        - 失败：抛出异常（或可改为返回错误状态）
        """
        if not isinstance(state.re.payload, REBeginSpec):
            raise TypeError("Payload must be a REBeginSpec for fetching params")

        messages = state.messages

        try:
            # result = CCRecommendParams(
            #     silica_amount=10.5,
            #     column_size="12g",
            #     flow_rate=30.0,
            #     solvent_system="PE/EA",
            #     start_solvent_ratio="30:1",
            #     end_solvent_ratio="5:1",
            #     estimated_time=45.0,
            #     complex_tlc=False,
            #     column_volume=15.0,
            #     air_purge_time=2.0
            # )
            result = get_recommended_params(state.re.payload)
            logger.info(f"RE params fetched: {result}")

            # 添加系统消息记录
            messages = MsgUtils.append_thinking(
                messages,
                f"[RE] 根据 spec 获取到推荐参数:\n"
                f"  - 溶剂系统: {getattr(result, 'solvent_system', 'N/A')}\n"
                f"  - 体积: {getattr(result, 'volume', 'N/A')}"
            )

            return {
                "re": state.re.model_copy(update={"payload": result}),
                "messages": messages,
            }

        except Exception as e:
            logger.exception("Failed to fetch RE recommended params")
            raise

    def _free_timepoint_params(self, state: REAgentGraphState) -> dict[str, Any]:
        """
        自由对话节点：
        - 用户点击 confirm → re.params_confirmed = True，进入下一阶段
        - 用户继续对话 → re.params_confirmed = False，循环回本节点
        """
        messages = state.messages
        payload = state.re.payload

        # 构建中断 payload
        interrupt_payload = OperationInterruptPayload(
            message="请确认当前配置，或继续对话进行调整。",
            args={
                "re": {"params": payload.model_dump(mode="json") if payload else None},
                "mode": "free_chat_with_confirm",
            },
        )

        # 中断并等待用户响应
        raw = interrupt(interrupt_payload.model_dump(mode="json"))
        resume: OperationResumePayload = coerce_operation_resume(raw)

        # 用户点击了 confirm 按钮
        if resume.approval:
            return {"re": state.re.model_copy(update={"phase": REPhase.PARAMS_CONFIRMED})}
        
        # 用户继续对话
        updates: dict[str, Any] = {"re": state.re.model_copy(update={"phase": REPhase.SPEC_CONFIRMED})}

        # 处理用户编辑的数据（如果有）
        if resume.data and isinstance(resume.data, dict):
            edited_params = resume.data.get("params")
            if edited_params:
                try:
                    new_payload = RERecommendParams.model_validate(edited_params)
                    updates["re"] = state.re.model_copy(update={"payload": new_payload})
                    payload = new_payload  # 用于后续 LLM 上下文
                except Exception:
                    logger.warning("Failed to parse edited params from resume.data")

        # 处理用户消息（仅在拒绝/继续对话时有效）
        user_text = (resume.comment or "").strip()
        if user_text:
            messages = MsgUtils.append_user_message(messages, user_text)

            # 调用 LLM 处理对话
            result = self._agent.invoke({"messages": [*messages]})
            
            # 如果 LLM 返回了更新的 params，应用它
            if hasattr(result, "structured_response"):
                # TODO：根据你的 LLM 输出结构处理
                pass
            
            # 将 AI 回复加入消息历史
            if "messages" in result and result["messages"]:
                last_msg = result["messages"][-1]
                ai_content = getattr(last_msg, "content", str(last_msg))
                messages = MsgUtils.append_response(messages, ai_content)

        updates["messages"] = messages
        return updates


if __name__ == "__main__":
    import asyncio
    import uuid
    from pathlib import Path

    from langchain_core.messages import HumanMessage
    from langchain_core.runnables.config import RunnableConfig
    from langgraph.types import Command

    from src.utils.tools import terminal_approval_handler

    # Step 1: 创建 agent
    agent = REAgent(with_checkpointer=True)

    # Step 2: 生成 workflow 图
    output_path = Path(__file__).resolve().parents[3] / "assets" / "re_agent_workflow.png"
    agent.compiled.get_graph(xray=True).draw_mermaid_png(output_file_path=str(output_path))
    logger.success(f"Workflow exported to {output_path}")

    # Step 3: 创建会话配置
    thread_id = str(uuid.uuid4())
    config: RunnableConfig = RunnableConfig(configurable={"thread_id": thread_id})

    # Step 4: 初始化输入
    text = input("[user]: ").strip() or "帮我做旋转蒸发"
    next_input: REAgentGraphState | Command = REAgentGraphState(
        messages=[HumanMessage(content=text)],
    )

    async def _run(next_input: REAgentGraphState | Command) -> None:
        """异步运行 RE Agent 测试循环"""
        while True:
            interrupted = False

            async for state in agent.compiled.astream(next_input, config=config, stream_mode="values"):
                if "__interrupt__" in state:
                    interrupted = True
                    print("\n" + "=" * 50)
                    print("[INTERRUPT] 等待用户确认...")
                    print("=" * 50)

                    # 终端交互处理（terminal_approval_handler 会显示中断信息）
                    next_input = terminal_approval_handler(state)
                    break

                # 打印当前状态
                print("\n" + "-" * 40)
                if "messages" in state:
                    for msg in state["messages"][-3:]:  # 只显示最后3条消息
                        print(f"[{type(msg).__name__}]: {getattr(msg, 'content', str(msg))[:200]}")
                if "re" in state and state["re"]:
                    print(f"[RE]: {state['re']}")
                print("-" * 40)

            if not interrupted:
                print("\n" + "=" * 50)
                print("Graph reached END")
                print("=" * 50)
                break

    asyncio.run(_run(next_input))