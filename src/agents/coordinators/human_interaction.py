from __future__ import annotations

from typing import TYPE_CHECKING, Any

from langchain_core.messages import AnyMessage
from langgraph.types import interrupt
from pydantic import Field

from src.models.operation import OperationInterruptPayload, OperationResumePayload
from src.utils.messages import MsgUtils
from src.utils.tools import coerce_operation_resume

if TYPE_CHECKING:
    from src.models.core import PlanningAgentOutput, PlanStep

DecisionPayload = dict[str, Any]


class HumanInLoop:
    reviewer: str = Field(
        default="supervisor_bot",
        description="Default reviewer identifier when simulating approvals.",
    )

    def review_plan(self, *, plan_out: PlanningAgentOutput, messages: list[AnyMessage]) -> dict[str, Any]:
        """
        HITL plan review.

        - approve: returns {"plan_approved": True}
        - reject: returns {"plan_approved": False} and may revise `messages`/`user_input`
        """
        interrupt_payload = self.build_plan_review_payload(plan_out=plan_out)
        payload = interrupt(interrupt_payload.model_dump(mode="json"))
        resume = coerce_operation_resume(payload)
        resume = self.normalize_resume(resume)

        updates: dict[str, Any] = {"plan_approved": bool(resume.approval)}
        if resume.approval:
            return updates

        edited_text = (resume.comment or "").strip()
        if edited_text:
            revised_messages = MsgUtils.apply_human_revision(messages, edited_text)
            updates["messages"] = revised_messages
            updates["user_input"] = MsgUtils.only_human_messages(revised_messages)
        return updates

    def approve_step(self, *, step: PlanStep) -> OperationResumePayload:
        """
        HITL per-step approval.

        Returns the normalized OperationResumePayload (structure-only; caller decides how to apply to PlanStep).
        """
        interrupt_payload = self.build_step_approval_payload(step=step)
        payload = interrupt(interrupt_payload.model_dump(mode="json"))
        resume = coerce_operation_resume(payload)
        return self.normalize_resume(resume)

    # NOTE: These helpers keep "content / formatting / interaction contract" out of `agent_mapper.py`.
    # `agent_mapper.py` should only do `interrupt()` + state updates.

    @staticmethod
    def build_plan_review_payload(*, plan_out: PlanningAgentOutput) -> OperationInterruptPayload:
        """Build interrupt payload for plan review UI."""
        steps_preview = [
            {
                "id": s.id,
                "title": s.title,
                "executor": str(s.executor),
                "requires_human_approval": s.requires_human_approval,
                "status": s.status.value,
                "args": s.args,
            }
            for s in plan_out.plan_steps
        ]
        return OperationInterruptPayload(
            message="Please review the PLAN. Approve to execute; reject to revise. Optionally provide edits in comment.",
            args={"plan_hash": plan_out.plan_hash, "plan_steps": steps_preview},
        )

    @staticmethod
    def build_step_approval_payload(*, step: PlanStep) -> OperationInterruptPayload:
        """Build interrupt payload for per-step approval UI."""
        return OperationInterruptPayload(
            message="Approve executing this step? If not, reject; optionally edit input in 'comment'.",
            args={
                "step": {
                    "id": step.id,
                    "title": step.title,
                    "executor": str(step.executor),
                    "args": step.args,
                    "requires_human_approval": step.requires_human_approval,
                    "status": step.status.value,
                },
            },
        )

    @staticmethod
    def normalize_resume(value: OperationResumePayload) -> OperationResumePayload:
        """Place to normalize/validate resume payload before applying it to state."""
        return value
