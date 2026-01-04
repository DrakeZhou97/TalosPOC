from datetime import datetime

from langchain.agents import create_agent
from langchain.agents.structured_output import StructuredOutputValidationError, ToolStrategy
from langchain_core.messages import AnyMessage

from src.models.core import UserAdmittance, WatchDogAIDetermined
from src.models.operation import OperationResponse
from src.utils.logging_config import logger
from src.utils.models import WATCHDOG_MODEL
from src.utils.PROMPT import WATCH_DOG_SYSTEM_PROMPT


class WatchDogAgent:
    def __init__(self) -> None:
        """Initialize the WatchDogAgent."""
        self.watch_dog = create_agent(
            model=WATCHDOG_MODEL,
            response_format=ToolStrategy[WatchDogAIDetermined](WatchDogAIDetermined),
            system_prompt=WATCH_DOG_SYSTEM_PROMPT,
        )

        logger.info("WatchDogAgent initialized with model={}", self.watch_dog)

    def run(self, user_input: list[AnyMessage]) -> OperationResponse[list[AnyMessage], UserAdmittance]:
        """
        Run the intention detection agent to analyze user input.

        Args:
            user_input (list[AnyMessage]): The user input messages ordered chronologically.

        Returns:
            OperationResponse[list[AnyMessage], UserAdmittance]: The operation response containing the raw input and the admittance decision.

        """
        start_time = datetime.now()

        logger.info("WatchDogAgent.run triggered with {} messages", len(user_input))

        ai_decision = self._watch_dog_release(input_msg=user_input)

        end_time = datetime.now()

        return OperationResponse[list[AnyMessage], UserAdmittance](
            operation_id="watchdog_admittance_001",
            input=user_input,
            output=UserAdmittance(
                id="user_admittance_001",
                user_input=user_input,
                within_domain=ai_decision.within_domain,
                within_capacity=ai_decision.within_capacity,
                feedback=ai_decision.feedback,
            ),
            start_time=start_time.isoformat(timespec="microseconds"),
            end_time=end_time.isoformat(timespec="microseconds"),
        )

    def _watch_dog_release(self, input_msg: list[AnyMessage]) -> WatchDogAIDetermined:
        """
        Understanding user intention from input text and detemining if it's within domain and capacity.

        Args:
            input_msg (list[BaseMessage]): The user input message.

        Returns:
            WatchDogAIDetermined: _description_

        """
        result = {}
        try:
            result = self.watch_dog.invoke(input={"messages": input_msg})  # type: ignore
            logger.info(
                "WatchDogAgent result: within_domain={}, within_capacity={}, feedback={}",
                result["structured_response"].within_domain,
                result["structured_response"].within_capacity,
                result["structured_response"].feedback,
            )
        except StructuredOutputValidationError as se:
            logger.error(f"WatchDog structured output validation error. error={se}")
            raise

        # TODO: Post validation and error handling here.

        return WatchDogAIDetermined(
            within_domain=result["structured_response"].within_domain,
            within_capacity=result["structured_response"].within_capacity,
            feedback=result["structured_response"].feedback,
        )


if __name__ == "__main__":
    from langchain_core.messages import HumanMessage

    watch_dog = WatchDogAgent()
    result = watch_dog.run(user_input=[HumanMessage(content="I want to know the current time.")])
    print(result)
