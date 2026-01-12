from typing import Any

from langchain_core.messages import AnyMessage
from pydantic import BaseModel, ConfigDict, Field

from src.models.enums import ExecutionStatusEnum, ExecutorKey


class PlannerAIOutput(BaseModel):
    """
    PlanAgent AI Output, which is the sequence of executor running order.

    Will be parsed and assembled into more complete object later.
    """

    steps: list[ExecutorKey] = Field(..., description="Sequence of execution, must be in the executor registry")


class PlanStep(BaseModel):
    """
    PlanAIOutput step will be parsed and wrapped afterwards by this.
    """

    id: str
    title: str
    executor: ExecutorKey
    args: dict[str, Any] = Field(default_factory=dict)  # Different for each executor
    status: ExecutionStatusEnum = ExecutionStatusEnum.NOT_STARTED
    output: Any | None = None


class PlannerAgentOutput(BaseModel):
    plan_steps: list[PlanStep] = Field(..., description="List of steps to be executed")
    plan_hash: str = Field(..., description="Hash of the plan")
    user_approval: bool = False


class PlannerAgentGraphState(BaseModel):
    """LangGraph subgraph state schema for planning + HITL review."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="ignore")

    messages: list[AnyMessage] = Field(default_factory=list)
    user_input: list[AnyMessage] = Field(default_factory=list)
    thinking: list[AnyMessage] = Field(
        default_factory=list,
        description="Internal trace AI messages ordered chronologically.",
    )

    plan: PlannerAgentOutput | None = None
    plan_cursor: int = 0

    # Step-by-step progress tracking for CopilotKit frontend
    current_step: str | None = Field(
        default=None,
        description="Current node/step name for real-time progress display.",
    )
    step_message: str | None = Field(
        default=None,
        description="User-friendly message describing the current step.",
    )
