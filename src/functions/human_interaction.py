from typing import Any

from pydantic import Field

from src.classes.operation import OperationResponse
from src.classes.system_state import HumanApproval, IntentionDetectionFin
from src.utils.logging_config import logger

DecisionPayload = dict[str, Any]


class HumanInLoop:
    reviewer: str = Field(
        default="supervisor_bot",
        description="Default reviewer identifier when simulating approvals.",
    )

    def request_human_input(self, user_input: str) -> str:
        """Simulate requesting human input outside of LangGraph (useful for tests)."""
        logger.info("Requesting human input. user_input='{}'", user_input)

        return user_input

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
