"""
Planner Agent: Generate a plan based on user input. The output should be a sequence of TODOs: a set of restricted commands.

1. We need to have a registry of executors, which is a dictionary of ExecutorKey and a callable function.
2. Planner only output executor that exist in the registry. we may need to verify that.

Guardrails:
1. Plan freeze, once approved, we will have a none changeable plan hash, if user want to change / edit it, we need to re-enter the audit workflow.
2. 幂等/续跑: 基于 step.id + status, 已完成的不重复跑; 失败标记 on_hold 并记录错误
"""

from collections.abc import Callable
from datetime import datetime
from uuid import uuid4

from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain_core.messages import AnyMessage

from src.classes.operation import OperationResponse
from src.classes.PROMPT import PLANNER_SYSTEM_PROMPT
from src.classes.system_enum import ExecutionStatusEnum
from src.classes.system_state import ExecutorKey, PlanningAgentOutput, PlanStep
from src.utils.logging_config import logger
from src.utils.models import PLANNER_MODEL

# Registry of executors
executor_registry: dict[ExecutorKey, Callable] = {}


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
        # The workflow routes to `execute_tlc` if the todo name contains "tlc" (case-insensitive).
        # PLAN should only contains

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


if __name__ == "__main__":
    from langchain_core.messages import HumanMessage

    planner = Planner()
    result = planner.run(user_input=[HumanMessage(content="帮我查一下阿司匹林的属性,然后设计一个TLC条件")])
    print(result.model_dump_json(indent=2))
