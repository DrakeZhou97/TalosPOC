from datetime import datetime

from langchain.agents import create_agent
from langchain.agents.structured_output import StructuredOutputValidationError, ToolStrategy
from langchain_core.messages import AnyMessage

from src.models.core import IDAIDetermine, IntentionDetectionFin
from src.models.operation import OperationResponse
from src.utils.logging_config import logger
from src.utils.models import INTENTION_DETECTION_MODEL
from src.utils.PROMPT import INTENTION_DETECTION_SYSTEM_PROMPT


class IntentionDetectionAgent:
    def __init__(self) -> None:
        self._agent = create_agent(
            model=INTENTION_DETECTION_MODEL,
            response_format=ToolStrategy[IDAIDetermine](IDAIDetermine),
            system_prompt=INTENTION_DETECTION_SYSTEM_PROMPT,
        )

        logger.info("IntentionDetectionAgent initialized with model={}", self._agent)

    def run(self, user_input: list[AnyMessage]) -> OperationResponse[list[AnyMessage], IntentionDetectionFin]:
        """Detect user intention from input messages."""
        start_time = datetime.now()

        logger.info("[Agent][IntentionDetectionAgent] IntentionDetectionAgent.run triggered with {} messages", len(user_input))

        ai_output = None
        try:
            result = self._agent.invoke(input={"messages": user_input})
            ai_output = result["structured_response"]

            if not isinstance(ai_output, IDAIDetermine):
                raise TypeError(
                    f"IntentionDetectionAgent output is not an IDAIDetermine: {ai_output}. It is {type(ai_output)}",
                )
        except StructuredOutputValidationError as ve:
            # TODO: Add Structured Output Validation Error Handling
            logger.error(f"[Agent][IntentionDetectionAgent] IntentionDetectionAgent error={ve}")
            raise

        output = IntentionDetectionFin(
            matched_goal_type=ai_output.matched_goal_type,
            reason=ai_output.reason,
            winner_id=ai_output.matched_goal_type.value,
            evidences=[ai_output],
        )

        end_time = datetime.now()

        logger.info(f"[Agent][IntentionDetectionAgent] IntentionDetectionAgent output={output}")

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
