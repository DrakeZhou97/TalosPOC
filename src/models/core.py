from langchain_core.messages import AnyMessage, HumanMessage
from pydantic import BaseModel, ConfigDict, Field, computed_field, field_validator

from src.models.enums import AdmittanceState, GoalTypeEnum
from src.models.operation import OperationResponse
from src.models.planner import PlannerAgentOutput
from src.models.tlc import TLCExecutionState
from src.models.cc import CCExecutionState
from src.models.re import REExecutionState

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


class AgentState(BaseModel):
    """LangGraph state schema (main workflow)."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="ignore")

    messages: list[AnyMessage] = Field(
        default_factory=list,
        description="The whole conversation messages ordered chronologically.",
    )
    thinking: list[AnyMessage] = Field(
        default_factory=list,
        description="Internal trace AI messages ordered chronologically.",
    )

    # Step-by-step progress tracking for CopilotKit frontend
    current_step: str | None = Field(
        default=None,
        description="Current node/step name for real-time progress display.",
    )
    step_message: str | None = Field(
        default=None,
        description="User-friendly message describing the current step.",
    )

    @computed_field(return_type=list[HumanMessage])
    @property
    def user_input(self) -> list[HumanMessage]:
        """
        Derived user-only messages from `messages`.

        This is a read-only view to avoid duplicating state and inconsistent writes.
        """
        return [m for m in self.messages if isinstance(m, HumanMessage)]

    bottom_line_feedback: str | None = None

    # Agent Output
    admittance: OperationResponse[list[AnyMessage], UserAdmittance] | None = None
    admittance_state: AdmittanceState | None = None
    intention: OperationResponse[list[AnyMessage], IntentionDetectionFin] | None = None

    # Planner
    plan: PlannerAgentOutput | None = None
    plan_cursor: int = 0

    # Executor namespaces
    tlc: TLCExecutionState = Field(default_factory=TLCExecutionState)
    cc: CCExecutionState = Field(default_factory=CCExecutionState)
    re: REExecutionState = Field(default_factory=REExecutionState)
    # System Stage for Routing purpose
    mode: str | None = None
