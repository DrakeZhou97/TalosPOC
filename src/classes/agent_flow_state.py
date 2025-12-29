from langchain_core.messages import AnyMessage
from pydantic import BaseModel, ConfigDict, Field

from src.classes.operation import OperationResponse
from src.classes.system_enum import AdmittanceState, TLCPhase
from src.classes.system_state import (
    HumanApproval,
    IntentionDetectionFin,
    PlanningAgentOutput,
    TLCAgentOutput,
    UserAdmittance,
)


class TLCState(BaseModel):
    """
    LangGraph state schema.

    Fields have defaults to behave like TypedDict(total=False).
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="ignore")

    # Received Input
    messages: list[AnyMessage] = Field(
        default_factory=list,
        description="The whole conversation messages ordered chronologically.",
    )
    user_input: list[AnyMessage] = Field(
        default_factory=list,
        description="The latest user input message.",
    )
    human_confirmation: OperationResponse[str, HumanApproval] | None = None
    plan_approved: bool = False
    bottom_line_feedback: str | None = None

    # Internal Usage
    id_res: OperationResponse[list[AnyMessage], UserAdmittance] | None = None

    # Agent Output
    admittance: OperationResponse[list[AnyMessage], UserAdmittance] | None = None
    admittance_state: AdmittanceState | None = None
    intention: OperationResponse[list[AnyMessage], IntentionDetectionFin] | None = None
    classify_res: OperationResponse[UserAdmittance, str] | None = None

    # Planner Output
    plan: OperationResponse[list[AnyMessage], PlanningAgentOutput] | None = None

    # Planner Execution
    plan_cursor: int = 0

    # TLC Execution (multi-turn)
    tlc_phase: TLCPhase = TLCPhase.COLLECTING
    tlc_spec: TLCAgentOutput | None = None

    mode: str | None = None
