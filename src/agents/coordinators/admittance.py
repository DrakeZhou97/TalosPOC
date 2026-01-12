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
        self._agent = create_agent(
            model=WATCHDOG_MODEL,
            response_format=ToolStrategy[WatchDogAIDetermined](WatchDogAIDetermined),
            system_prompt=WATCH_DOG_SYSTEM_PROMPT,
        )

        logger.info("WatchDogAgent initialized with model={}", self._agent)

    def run(self, user_input: list[AnyMessage]) -> OperationResponse[list[AnyMessage], UserAdmittance]:
        """Run admittance check on user input."""
        start_time = datetime.now()

        logger.info("[Agent][WatchDogAgent] WatchDogAgent.run triggered with {} messages", len(user_input))

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
        """Determine if request is within domain and capacity."""
        try:
            result = self._agent.invoke(input={"messages": input_msg})
            ai_output = result["structured_response"]

            if not isinstance(ai_output, WatchDogAIDetermined):
                raise TypeError(
                    f"WatchDogAgent output is not a WatchDogAIDetermined: {ai_output}. It is {type(ai_output)}",
                )
        except StructuredOutputValidationError as ve:
            logger.error(f"[Agent][WatchDogAgent] WatchDog structured output validation error. error={ve}")
            raise

        # TODO: Post validation and error handling here.

        else:
            logger.info(
                "[Agent][WatchDogAgent] WatchDogAgent result: within_domain={}, within_capacity={}, feedback={}",
                ai_output.within_domain,
                ai_output.within_capacity,
                ai_output.feedback,
            )
            return ai_output


if __name__ == "__main__":
    from langchain_core.messages import HumanMessage

    watch_dog = WatchDogAgent()
    result = watch_dog.run(user_input=[HumanMessage(content="I want to know the current time.")])
    print(result)
