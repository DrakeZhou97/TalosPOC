from langchain_core.messages import AnyMessage, HumanMessage
from pydantic import BaseModel, ConfigDict, Field

from src.models.enums import TLCPhase

# region <TLC MCP>


class Compound(BaseModel):
    compound_name: str = Field(..., description="Compound name, IUPAC standard name, English")
    smiles: str | None = Field(default=None, description="SMILES expression of the compound")


class TLCRatioResult(BaseModel):
    """MCP server output payload (inner object)."""

    solvent_system: str
    ratio: str
    rf_value: float | None = None
    description: str | None = None
    origin: str | None = None
    backend: str | None = None


class TLCRatioPayload(BaseModel):
    """MCP server output payload (outer envelope)."""

    request_id: str
    compound_name: str
    smiles: str | None = None
    tlc_parameters: TLCRatioResult
    timestamp: str


# endregion


class TLCAIOutput(BaseModel):
    """Extract compound information from user input text."""

    compounds: list[Compound | None] = Field(default_factory=list, description="List of compounds extracted from the text")


class TLCCompoundSpecItem(Compound, TLCRatioResult):
    """TLC compound spec used for form filling + MCP lookup."""


class TLCAgentOutput(TLCAIOutput):
    exp_params: list[TLCCompoundSpecItem] = Field(
        ...,
        description="List of compound specifications including MCP recommended ratios",
    )
    confirmed: bool = Field(default=False, description="Whether the compound form has been confirmed by user/human")


class TLCExecutionState(BaseModel):
    """
    TLC subflow execution state.

    Keeps TLC-specific multi-turn fields isolated so the main workflow state stays small.
    """

    phase: TLCPhase = TLCPhase.COLLECTING
    spec: TLCAgentOutput | None = None


class TLCAgentGraphState(BaseModel):
    """LangGraph subgraph state schema for TLCAgent (structure-only POC)."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="ignore")

    # Shared
    messages: list[AnyMessage] = Field(default_factory=list)
    user_input: list[HumanMessage] = Field(default_factory=list)
    tlc: TLCExecutionState = Field(default_factory=TLCExecutionState)

    # Private
    thinking: list[AnyMessage] = Field(default_factory=list)
    revision_text: str = ""
    user_approved: bool = False

    # Step-by-step progress tracking for CopilotKit frontend
    current_step: str | None = Field(
        default=None,
        description="Current node/step name for real-time progress display.",
    )
    step_message: str | None = Field(
        default=None,
        description="User-friendly message describing the current step.",
    )
