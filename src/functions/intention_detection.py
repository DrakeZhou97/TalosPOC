from datetime import datetime

from langchain.agents import create_agent
from langchain.agents.middleware.types import after_agent
from langchain.agents.structured_output import StructuredOutputValidationError
from langchain_core.messages import AnyMessage

from src.classes.operation import OperationResponse
from src.classes.PROMPT import INTENTION_DETECTION_SYSTEM_PROMPT
from src.classes.system_state import IDAIDetermine, IntentionDetectionFin
from src.utils.logging_config import logger
from src.utils.models import INTENTION_DETECTION_MODEL


class IntentionDetectionAgent:
    def __init__(self) -> None:
        """Initialize the IntentionDetectionAgent."""
        self.intention_detection_agent = create_agent(
            model=INTENTION_DETECTION_MODEL,
            response_format=IDAIDetermine,
            system_prompt=INTENTION_DETECTION_SYSTEM_PROMPT,
        )

    def run(self, user_input: list[AnyMessage]) -> OperationResponse[list[AnyMessage], IntentionDetectionFin]:
        """
        Detect user intention from input messages.

        Args:
            user_input (list[AnyMessage]): The user input messages ordered chronologically.

        Returns:
            OperationResponse[list[AnyMessage], IntentionDetectionFin]: The operation response containing the input messages and the detected intention.

        """
        start_time = datetime.now()

        logger.info("IntentionDetectionAgent.run triggered with {} messages", len(user_input))

        # Step 1. AI determine the intention of which goal
        ai_output = None
        try:
            ai_output = self.intention_detection_agent.invoke(input={"messages": user_input})["structured_response"]  # type: ignore

            if not isinstance(ai_output, IDAIDetermine):
                raise TypeError(
                    f"IntentionDetectionAgent output is not an IDAIDetermine: {ai_output}. It is {type(ai_output)}",
                )
        except StructuredOutputValidationError as se:
            # TODO: Add Structured Output Validation Error Handling
            logger.error(f"IntentionDetectionAgent error={se}")
            raise

        # Step 2. Construct the intention detection fin output
        output = IntentionDetectionFin(
            matched_goal_type=ai_output.matched_goal_type,
            reason=ai_output.reason,
            winner_id=ai_output.matched_goal_type.value,
            evidences=[ai_output],
        )

        end_time = datetime.now()

        logger.info(f"IntentionDetectionAgent output={output}")

        return OperationResponse[list[AnyMessage], IntentionDetectionFin](
            operation_id="intention_detection_001",
            input=user_input,
            output=output,
            start_time=start_time.isoformat(timespec="microseconds"),
            end_time=end_time.isoformat(timespec="microseconds"),
        )


if __name__ == "__main__":
    from langchain_core.messages import HumanMessage

    agent = IntentionDetectionAgent()
    agent.run(user_input=[HumanMessage(content="帮我做个 TLC 点板分析")])
