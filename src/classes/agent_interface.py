from abc import ABC, abstractmethod

from src.classes.operation import OperationResponse


class IAgentBase(ABC):
    @abstractmethod
    def run(self, user_input: list[AnyMessage]) -> OperationResponse[list[AnyMessage], Any]:
        """
        Run the agent.

        Args:
            user_input (list[AnyMessage]): The user input messages ordered chronologically.

        Returns:
            OperationResponse[list[AnyMessage], Any]: The operation response containing the raw input and the agent output.
        """
