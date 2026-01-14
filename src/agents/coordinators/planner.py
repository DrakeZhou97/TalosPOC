"""
Planner Agent: Generate a plan based on user input. The output should be a sequence of TODOs: a set of restricted commands.

1. We need to have a registry of executors, which is a dictionary of ExecutorKey and a callable function.
2. Planner only output executor that exist in the registry. we may need to verify that.

Guardrails:
1. Plan freeze, once approved, we will have a none changeable plan hash, if user want to change / edit it, we need to re-enter the audit workflow.
2. 幂等/续跑: 基于 step.id + status, 已完成的不重复跑; 失败标记 on_hold 并记录错误

TODO: Clean Code
"""

from __future__ import annotations

import uuid
from contextlib import suppress
from typing import Any

from copilotkit.langgraph import copilotkit_emit_state
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain_core.runnables.config import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from src.agents.specialists.presenter import present_final, present_review
from src.models.enums import ExecutionStatusEnum, ExecutorKey
from src.models.operation import OperationInterruptPayload
from src.models.planner import PlannerAgentGraphState, PlannerAgentOutput, PlanStep
from src.utils.logging_config import logger
from src.utils.messages import MsgUtils
from src.utils.models import PLANNER_MODEL
from src.utils.PROMPT import PLANNER_SYSTEM_PROMPT
from src.utils.tools import coerce_operation_resume

# Step messages for CopilotKit frontend progress display
PLANNER_STEP_MESSAGES = {
    "generate_plan": "Creating an execution plan for your request...",
    "plan_review": "Waiting for plan approval...",
}


class PlannerAgent:
    """Planning subgraph: generate_plan -> plan_review (approve/revise loop)."""

    def __init__(self, *, with_checkpointer: bool = False) -> None:
        """
        Build and compile the internal LangGraph subgraph.

        Args:
            with_checkpointer: If True, compiles with MemorySaver for standalone usage with HITL.
                               If False (default), parent graph's checkpointer will be used.

        """
        logger.info("[Agent][PlannerAgent] PlannerAgent initialized with model={}", PLANNER_MODEL)

        subgraph = StateGraph(PlannerAgentGraphState)
        subgraph.add_node("generate_plan", self._generate_plan)
        subgraph.add_node("interrupt_plan_review", self._interrupt_plan_review)

        subgraph.add_edge(START, "generate_plan")
        subgraph.add_edge("generate_plan", "interrupt_plan_review")
        subgraph.add_conditional_edges(
            "interrupt_plan_review",
            self._route_plan_review,
            {
                "revise": "generate_plan",
                "approved": END,
            },
        )

        checkpointer = MemorySaver() if with_checkpointer else None
        self.compiled = subgraph.compile(checkpointer=checkpointer)

        # Keep the LLM agent initialization here (after compile) to match the TLCAgent pattern and avoid unnecessary init cost when only graph
        # structure is inspected.
        self._agent = create_agent(
            model=PLANNER_MODEL,
            response_format=ToolStrategy[PlannerAgentOutput](PlannerAgentOutput),
            system_prompt=PLANNER_SYSTEM_PROMPT,
        )

    async def _generate_plan(self, state: PlannerAgentGraphState, config: RunnableConfig) -> dict[str, Any]:
        # Emit step progress to frontend (best-effort; ignore if no callback context)
        with suppress(RuntimeError):
            await copilotkit_emit_state(
                config,
                {
                    "current_step": "generate_plan",
                    "step_message": PLANNER_STEP_MESSAGES["generate_plan"],
                },
            )

        logger.info("[Agent][PlannerAgent] PlannerAgent.generate_plan with messages")

        # Step 1. Generate the plan steps call the model

        # TODO: Call Agent to generate the plan and compose to PlannerAgentOutput class

        # Step 2. Compose the plan steps to PlannerAgentOutput class

        plan_steps = [
            PlanStep(
                id=uuid.uuid4().hex,
                title="Rotary Evaporation",
                executor=ExecutorKey.RE_AGENT,
                args={},
                status=ExecutionStatusEnum.NOT_STARTED,
                output=None,
            ),
        ]

        # NOTE: After have the real steps,

        plan_out = PlannerAgentOutput(
            plan_steps=plan_steps,
            plan_hash=uuid.uuid4().hex,
            user_approval=False,
        )

        return {
            "plan": plan_out,
            "plan_cursor": 0,
            # "messages": MsgUtils.append_thinking(
            #     state.messages,
            #     f"[planner] plan_created plan_hash={plan_out.plan_hash} steps={len(plan_out.plan_steps)}",
            # ),
            "current_step": None,
            "step_message": None,
        }

    async def _interrupt_plan_review(self, state: PlannerAgentGraphState, config: RunnableConfig) -> dict[str, Any]:
        """
        Interrupt + apply resume for plan review.

        This mirrors the TLCAgent HITL pattern: build the review payload, interrupt, then either approve (END) or append a revision message and
        loop back to `generate_plan`.
        """
        # Emit step progress to frontend (best-effort; ignore if no callback context)
        with suppress(RuntimeError):
            await copilotkit_emit_state(
                config,
                {
                    "current_step": "plan_review",
                    "step_message": PLANNER_STEP_MESSAGES["plan_review"],
                },
            )

        plan = state.plan

        msgs = list(state.messages)

        base_result = {"current_step": None, "step_message": None}

        if plan is None:
            # Defensive: if we somehow reached review without a plan, loop back to regenerate.
            msgs = MsgUtils.append_thinking(list(msgs), "[planner] missing plan at review; regenerating")
            return {
                "messages": msgs,
                "plan": None,
                "plan_cursor": 0,
                **base_result,
            }

        # Generate review message and interrupt payload

        # review_msg = await present_review(msgs, kind="plan_review", args=plan.model_dump()) # NOTE: Not used for now

        review_msg = "[PRSNT] Please confirm following plan"

        payload = OperationInterruptPayload(message=review_msg, args=plan.model_dump())

        logger.critical(f"[Agent][PlannerAgent] PlannerAgent._interrupt_plan_review reached, payload={payload}")

        raw = interrupt(payload.model_dump(mode="json"))

        logger.critical(f"[Agent][PlannerAgent] PlannerAgent._interrupt_plan_review reached, raw={raw}")

        # Interrupting ... following are after resume logic.

        resume = coerce_operation_resume(raw)

        edited_text = (resume.comment or "").strip()
        if edited_text:
            msgs = MsgUtils.append_user_message(msgs, edited_text)

        # On approve: return the approved plan.

        # resp_msg = await present_final(msgs)

        resp_msg = "[PLANNER][PRSNT] Plan approved, will start executing"

        if resume.approval:
            return {
                "messages": MsgUtils.append_response(msgs, resp_msg),
                "plan": plan.model_copy(update={"user_approval": True}),
                **base_result,
            }

        # On reject: append the revision instruction (if provided) and clear old plan so parent graph won't treat it as executable.

        return {"messages": msgs, "plan": None, "plan_cursor": 0, **base_result}

    @staticmethod
    def _route_plan_review(state: PlannerAgentGraphState) -> str:
        return "approved" if state.plan and state.plan.user_approval else "revise"


if __name__ == "__main__":
    from pathlib import Path

    from langchain_core.messages import HumanMessage
    from langchain_core.runnables.config import RunnableConfig
    from langgraph.types import Command

    from src.models.operation import OperationResumePayload
    from src.utils.tools import _pretty

    agent = PlannerAgent(with_checkpointer=True)

    output_path = Path(__file__).resolve().parents[3] / "assets" / "planner_graph.png"
    agent.compiled.get_graph(xray=True).draw_mermaid_png(output_file_path=str(output_path))

    thread_id = str(uuid.uuid4())
    config: RunnableConfig = RunnableConfig(configurable={"thread_id": thread_id})

    next_input: PlannerAgentGraphState | Command = PlannerAgentGraphState(
        messages=[HumanMessage(content="我正在进行水杨酸的乙酰化反应制备乙酰水杨酸帮我进行中控监测IPC")],
    )

    res = agent.compiled.invoke(
        next_input,
        config=config,
        stream_mode="values",
    )

    print(_pretty(res))

    next_input = Command(resume=OperationResumePayload(approval=True, comment="", data=None))

    res = agent.compiled.invoke(
        next_input,
        config=config,
        stream_mode="values",
    )

    print(_pretty(res))
