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

from datetime import datetime
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain_core.messages import AIMessage, AnyMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt
from pydantic import BaseModel, ConfigDict, Field

from src.agents.coordinators.human_interaction import HumanInLoop
from src.models.core import PlanningAgentOutput, PlanStep
from src.models.enums import ExecutionStatusEnum, ExecutorKey
from src.models.operation import OperationInterruptPayload, OperationResponse
from src.presenter import present_review
from src.utils.logging_config import logger
from src.utils.messages import MessagesUtils
from src.utils.models import PLANNER_MODEL
from src.utils.PROMPT import PLANNER_SYSTEM_PROMPT
from src.utils.tools import coerce_operation_resume

if TYPE_CHECKING:
    from collections.abc import Callable

# Registry of executors
executor_registry: dict[ExecutorKey, Callable[..., Any]] = {}


class Planner:
    def __init__(self) -> None:
        """Initialize the Planner Agent."""
        self.planner = create_agent(
            model=PLANNER_MODEL,
            response_format=ToolStrategy[PlanningAgentOutput](PlanningAgentOutput),
            system_prompt=PLANNER_SYSTEM_PROMPT,
        )
        logger.info("Planner initialized")

    def run(self, user_input: list[AnyMessage]) -> OperationResponse[list[AnyMessage], PlanningAgentOutput]:
        """
        Generate a plan based on user input. The output should be a sequence of TODOs: a set of restricted commands.

        Args:
            user_input (list[AnyMessage]): The user input messages ordered chronologically.

        Returns:
            OperationResponse[list[AnyMessage], PlanningAgentOutput]: The operation response containing the plan.

        """
        start_time = datetime.now()
        logger.info("Planner.run triggered with {} messages", len(user_input))

        # NOTE: Placeholder planning logic.
        plan_steps = [
            PlanStep(
                id="1",
                title="TLC IPC monitoring (extract compounds + lookup Rf)",
                executor=ExecutorKey.TLC_AGENT,
                args={},
                # TLC execution includes its own form confirmation stage, so avoid double HITL.
                requires_human_approval=False,
                status=ExecutionStatusEnum.NOT_STARTED,
                output=None,
            ),
        ]

        plan = PlanningAgentOutput(plan_steps=plan_steps, plan_hash="MOCK HASH")

        end_time = datetime.now()

        return OperationResponse[list[AnyMessage], PlanningAgentOutput](
            operation_id=f"planner_{uuid4()}",
            input=user_input,
            output=plan,
            start_time=start_time.isoformat(),
            end_time=end_time.isoformat(),
        )


class PlannerGraphState(BaseModel):
    """LangGraph subgraph state schema for planning + HITL review."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="ignore")

    messages: list[AnyMessage] = Field(default_factory=list)
    user_input: list[AnyMessage] = Field(default_factory=list)
    thinking: list[AnyMessage] = Field(default_factory=list)

    plan: OperationResponse[list[AnyMessage], PlanningAgentOutput] | None = None
    plan_cursor: int = 0
    plan_approved: bool = False
    pending_interrupt: OperationInterruptPayload | None = None
    revision_text: str | None = None


class PlannerSubgraph:
    """Planning subgraph: generate_plan -> plan_review (approve/revise loop)."""

    def __init__(self, *, with_checkpointer: bool = False) -> None:
        """Build and compile the internal planning subgraph."""
        self._planner = Planner()
        self._human = HumanInLoop()

        subgraph = StateGraph(PlannerGraphState)
        subgraph.add_node("generate_plan", self._generate_plan)
        subgraph.add_node("plan_review_present", self._plan_review_present)
        subgraph.add_node("plan_review_interrupt", self._plan_review_interrupt)
        subgraph.add_node("plan_revision_present", self._plan_revision_present)

        subgraph.add_edge(START, "generate_plan")
        subgraph.add_edge("generate_plan", "plan_review_present")
        subgraph.add_edge("plan_review_present", "plan_review_interrupt")
        subgraph.add_conditional_edges(
            "plan_review_interrupt",
            self._route_plan_review,
            {
                "revise": "plan_revision_present",
                "approved": END,
            },
        )
        subgraph.add_edge("plan_revision_present", "generate_plan")

        checkpointer = MemorySaver() if with_checkpointer else None
        self.compiled = subgraph.compile(checkpointer=checkpointer)

    @staticmethod
    def _ensure_messages(state: PlannerGraphState) -> list[AnyMessage]:
        return MessagesUtils.ensure_messages(state)  # type: ignore[arg-type]

    @staticmethod
    def _ensure_work_messages(state: PlannerGraphState) -> list[AnyMessage]:
        # Use the latest human-only messages as execution context.
        # (For now, revisions are appended by presenter nodes into `messages`.)
        return MessagesUtils.only_human_messages(MessagesUtils.ensure_messages(state))  # type: ignore[arg-type]

    def _generate_plan(self, state: PlannerGraphState) -> dict[str, Any]:
        work_messages = self._ensure_work_messages(state)
        logger.info("PlannerSubgraph.generate_plan with {} messages", len(work_messages))

        res = self._planner.run(user_input=work_messages)
        plan_out = res.output

        return {
            "plan": res,
            "plan_cursor": 0,
            "plan_approved": False,
            "messages": MessagesUtils.append_thinking(
                MessagesUtils.ensure_messages(state),  # type: ignore[arg-type]
                f"[planner] plan_created plan_hash={plan_out.plan_hash} steps={len(plan_out.plan_steps)}",
            ),
        }

    def _plan_review_present(self, state: PlannerGraphState) -> dict[str, Any]:
        """Presenter step before interrupt: write user-facing review prompt into messages."""
        if state.plan is None:
            raise ValueError("Missing 'plan' before plan_review")
        plan_out = state.plan.output

        # 1) Build structured payload.
        base_payload = self._human.build_plan_review_payload(plan_out=plan_out)
        payload = OperationInterruptPayload(
            message=present_review(
                MessagesUtils.only_human_messages(MessagesUtils.ensure_messages(state)),  # type: ignore[arg-type]
                kind="plan_review",
                args=base_payload.args,
            ),
            args=base_payload.args,
        )

        # 2) Presenter message to UI before interrupt. (Keep history; final presenter will clean.)
        messages = [*MessagesUtils.ensure_messages(state), AIMessage(content=payload.message)]  # type: ignore[arg-type]

        return {"messages": messages, "pending_interrupt": payload, "plan_approved": False}

    def _plan_review_interrupt(self, state: PlannerGraphState) -> dict[str, Any]:
        """Interrupt + apply resume result (approval/edit) after UI response."""
        if state.pending_interrupt is None:
            raise ValueError("Missing 'pending_interrupt' before plan_review_interrupt")

        raw = interrupt(state.pending_interrupt.model_dump(mode="json"))
        resume = self._human.normalize_resume(coerce_operation_resume(raw))

        updates: dict[str, Any] = {"pending_interrupt": None, "plan_approved": bool(resume.approval)}
        if resume.approval:
            return updates

        edited_text = (resume.comment or "").strip()
        if edited_text:
            updates["revision_text"] = edited_text
        # Explicitly clear old plan on reject, so parent graph won't treat it as executable.
        updates["plan"] = None
        updates["plan_cursor"] = 0
        return updates

    def _plan_revision_present(self, state: PlannerGraphState) -> dict[str, Any]:
        """
        Presenter for human revision after plan rejection.

        This node is the only place that appends the revised HumanMessage into UI `messages`.
        Execution context is derived from human messages and may be overridden via `thinking` markers.
        """
        edited_text = (state.revision_text or "").strip()
        if not edited_text:
            return {"revision_text": None}

        messages = [*list(state.messages), HumanMessage(content=edited_text)]
        return {"messages": messages, "revision_text": None}

    @staticmethod
    def _route_plan_review(state: PlannerGraphState) -> str:
        return "approved" if state.plan_approved else "revise"


planner_subgraph = PlannerSubgraph()


if __name__ == "__main__":
    from langchain_core.messages import HumanMessage

    planner = Planner()
    result = planner.run(user_input=[HumanMessage(content="帮我查一下阿司匹林的属性,然后设计一个TLC条件")])
    print(result.model_dump_json(indent=2))
