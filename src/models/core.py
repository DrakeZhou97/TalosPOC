from typing import Any
from uuid import uuid4

from langchain_core.messages import AIMessage, AnyMessage
from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.models.enums import AdmittanceState, ExecutionStatusEnum, ExecutorKey, GoalTypeEnum
from src.models.operation import OperationResponse
from src.models.tlc import TLCAgentOutput, TLCPhase


class IDAIDetermine(BaseModel):
    matched_goal_type: GoalTypeEnum
    reason: str

    def __str__(self) -> str:
        return f"Matched goal type: {self.matched_goal_type}, reason: {self.reason}"


class IntentionDetectionFin(IDAIDetermine):
    winner_id: str  # matched goal_id
    evidences: list[IDAIDetermine]


class WatchDogAIDetermined(BaseModel):
    within_domain: bool = Field(..., description="表示用户输入是否在系统领域范围内")
    within_capacity: bool = Field(..., description="表示用户输入是否在系统可执行范围内")
    feedback: str = Field(..., description="对用户输入的判别依据和进一步输入建议")

    @field_validator("within_capacity", "within_domain")
    @classmethod
    def check_boolean(cls, v: bool) -> bool:
        """Validate that the field is a boolean."""
        if not isinstance(v, bool):
            raise TypeError("within_prefixed attr must be a boolean")
        return v


class UserAdmittance(WatchDogAIDetermined):
    id: str
    user_input: list[AnyMessage]


class HumanApproval(BaseModel):
    """Represents the outcome of a human approval step."""

    id: str = Field(default_factory=lambda: str(uuid4()), description="Unique identifier for this approval decision.")
    approval: bool = Field(..., description="Whether the human reviewer approved proceeding.")
    reviewed: IntentionDetectionFin = Field(..., description="The operation response that was reviewed.")
    comment: str | None = Field(None, description="Revised user input provided when the decision is 'edit'.")


class PlanStep(BaseModel):
    id: str
    title: str
    executor: ExecutorKey
    args: dict[str, Any] = Field(default_factory=dict)  # Different for each executor
    requires_human_approval: bool = True
    status: ExecutionStatusEnum = ExecutionStatusEnum.NOT_STARTED
    output: Any | None = None


class PlanningAgentOutput(BaseModel):
    plan_steps: list[PlanStep] = Field(..., description="List of steps to be executed")
    plan_hash: str = Field(..., description="Hash of the plan")


class TLCExecutionState(BaseModel):
    """
    TLC subflow execution state.

    Keeps TLC-specific multi-turn fields in a dedicated namespace so the main
    workflow state doesn't grow unbounded as we add more executors.
    """

    phase: TLCPhase = TLCPhase.COLLECTING
    spec: TLCAgentOutput | None = None


class AgentState(BaseModel):
    """LangGraph state schema (main workflow)."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="ignore")

    # Received Input
    messages: list[AnyMessage] = Field(default_factory=list, description="The whole conversation messages ordered chronologically.")
    user_input: list[AnyMessage] = Field(default_factory=list, description="The latest user input message.")
    resp_msg: AIMessage = Field(..., description="The response message from the latest ran agent.")

    bottom_line_feedback: str | None = None

    # Agent Output
    admittance: OperationResponse[list[AnyMessage], UserAdmittance] | None = None
    admittance_state: AdmittanceState | None = None
    intention: OperationResponse[list[AnyMessage], IntentionDetectionFin] | None = None

    # Planner Output
    plan: OperationResponse[list[AnyMessage], PlanningAgentOutput] | None = None

    # Planner Execution
    plan_cursor: int = 0

    # Plan Approval
    plan_approved: bool = False

    # Executor namespaces
    tlc: TLCExecutionState = Field(default_factory=TLCExecutionState)

    # System Stage for Routing purpose
    mode: str | None = None
