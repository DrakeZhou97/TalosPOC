from __future__ import annotations

from typing import Any

from langchain_core.messages import AnyMessage
from langgraph.types import interrupt
from pydantic import Field

from src.classes.operation import OperationInterruptPayload, OperationResponse, OperationResume
from src.classes.system_state import HumanApproval, IntentionDetectionFin, PlanningAgentOutput, PlanStep
from src.utils.logging_config import logger
from src.utils.messages import apply_human_revision, dump_messages_for_human_review
from src.utils.tools import coerce_operation_resume

DecisionPayload = dict[str, Any]


class HumanInLoop:
    reviewer: str = Field(
        default="supervisor_bot",
        description="Default reviewer identifier when simulating approvals.",
    )

    def post_human_confirmation(
        self,
        comment: str | None,
        reviewed: IntentionDetectionFin,
        approval: bool = False,
    ) -> OperationResponse[str, HumanApproval]:
        """Verify / Normalize user input etc."""
        # TODO: Add actual logic here

        logger.info("Posting human confirmation. approval={} comment='{}' reviewed={}", approval, comment, reviewed)

        return OperationResponse[str, HumanApproval](
            operation_id="human_confirmation_1",
            input="See reviewd attr",
            output=HumanApproval(
                approval=approval,
                reviewed=reviewed,
                comment=comment,
            ),
            start_time=None,
            end_time=None,
        )

    def confirm_intention(
        self,
        *,
        reviewed: IntentionDetectionFin,
        messages: list[AnyMessage],
    ) -> dict[str, Any]:
        """
        HITL confirmation for intention detection.

        Returns a TLCState patch dict: always includes `human_confirmation`,
        and may include revised `messages`/`user_input` when rejected with edits.
        """
        interrupt_payload = self.build_intention_review_payload(
            reviewed=reviewed,
            messages_dump=dump_messages_for_human_review(messages),
        )
        payload = interrupt(interrupt_payload.model_dump(mode="json"))

        resume = coerce_operation_resume(payload)
        resume = self.normalize_resume(resume)

        confirmation = self.post_human_confirmation(
            reviewed=reviewed,
            approval=resume.approval,
            comment=resume.comment,
        )

        updates: dict[str, Any] = {"human_confirmation": confirmation}
        edited_text = (resume.comment or "").strip()
        if not resume.approval and edited_text:
            revised_messages = apply_human_revision(messages, edited_text)
            updates["messages"] = revised_messages
            updates["user_input"] = revised_messages
        return updates

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
            revised_messages = apply_human_revision(messages, edited_text)
            updates["messages"] = revised_messages
            updates["user_input"] = revised_messages
        return updates

    def approve_step(self, *, step: PlanStep) -> OperationResume:
        """
        HITL per-step approval.

        Returns the normalized OperationResume (structure-only; caller decides how to apply to PlanStep).
        """
        interrupt_payload = self.build_step_approval_payload(step=step)
        payload = interrupt(interrupt_payload.model_dump(mode="json"))
        resume = coerce_operation_resume(payload)
        return self.normalize_resume(resume)

    # region <LangGraph-facing payload builders>
    # NOTE: These helpers keep "content / formatting / interaction contract" out of `agent_mapper.py`.
    # `agent_mapper.py` should only do `interrupt()` + state updates.

    @staticmethod
    def build_intention_review_payload(
        *,
        reviewed: IntentionDetectionFin,
        messages_dump: list[dict[str, Any]],
    ) -> OperationInterruptPayload:
        """Build interrupt payload for intention review UI."""
        return OperationInterruptPayload(
            message=f"Your intention is to {reviewed.winner_id}",
            args={
                "intention": reviewed.model_dump(mode="json"),
                "current_user_input": messages_dump,
            },
        )

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
    def normalize_resume(value: OperationResume) -> OperationResume:
        """Place to normalize/validate resume payload before applying it to state."""
        return value

    # endregion
