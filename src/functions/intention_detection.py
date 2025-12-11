from datetime import datetime

from langchain.agents import create_agent
from langchain.agents.structured_output import StructuredOutputValidationError
from langchain_core.messages import HumanMessage

from classes.system_state import IntentionDetectionFin
from src.classes.operation import OperationResponse
from src.classes.PROMPT import INTENTION_DETECTION_SYSTEM_PROMPT
from src.classes.system_state import IDAIDetermine, IntentionDetectionFin
from src.utils.logging_config import logger
from src.utils.models import MAX_MODEL


class IntentionDetectionAgent:
    def __init__(self) -> None:
        """Initialize the IntentionDetectionAgent."""
        self.intention_detection_agent = create_agent(
            model=MAX_MODEL,
            response_format=IDAIDetermine,
            system_prompt=INTENTION_DETECTION_SYSTEM_PROMPT,
        )

    def run(self, user_input: str) -> OperationResponse[str, IntentionDetectionFin]:
        """
        Detect user intention from input text.

        Args:
            user_admittance (UserAdmittance): The admittance information of the user.

        Returns:
            OperationResponse[UserAdmittance, IntentionDetectionFin]: The operation response containing the input admittance and the detected intention.

        """
        # Step 1. AI determine the intention of which gola
        ai_output = None
        try:
            ai_output = self.intention_detection_agent.invoke({"messages": [HumanMessage(content=user_input)]})[
                "structured_response"
            ]

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
            goal_id=ai_output.goal_id,
            matching_score=ai_output.matching_score,
            reason=ai_output.reason,
            winner_id=ai_output.goal_id,
            evidences=[ai_output],
        )

        logger.info(f"IntentionDetectionAgent output={output}")

        return OperationResponse[str, IntentionDetectionFin](
            operation_id="intention_detection_001",
            input=user_input,
            output=output,
            start_time=datetime.now().isoformat(),
            end_time=datetime.now().isoformat(),
        )
