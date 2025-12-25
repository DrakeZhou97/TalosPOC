from enum import StrEnum
from typing import Any
from uuid import uuid4

from langchain_core.messages import AnyMessage
from pydantic import BaseModel, Field, field_validator

from src.classes.system_enum import ExecutionStatusEnum, GoalTypeEnum


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


# region <TLC-specific classes>


class Compound(BaseModel):
    compound_name: str = Field(..., description="Compound name, IUPAC standard name, English")
    smiles: str | None = Field(default=None, description="SMILES expression of the compound")

    # @field_validator("compound_name", "smiles", mode="before")
    # @classmethod
    # def _normalize_optional_str(cls, v: str | None) -> str | None:
    #     if v is None:
    #         return None
    #     if not isinstance(v, str):
    #         raise TypeError("compound_name/SMILES must be a string or None")
    #     stripped = v.strip()
    #     return stripped or None

    # @field_validator("smiles")
    # @classmethod
    # def _validate_smiles(cls, v: str | None) -> str | None:
    #     if v is None:
    #         return None

    #     mol = Chem.MolFromSmiles(v)
    #     if mol is None:
    #         raise ValueError("Invalid smiles: RDKit failed to parse")
    #     return v

    # @field_validator("compound_name")
    # @classmethod
    # def _validate_compound_name_as_chem(cls, v: str | None) -> str | None:
    #     """
    #     Validate `compound_name` as a chemical identifier when provided.

    #     Note: RDKit does not natively resolve common names (e.g. "acetone") to structures.
    #     This validator treats `compound_name` as a machine-readable chemical identifier
    #     (smiles or InChI).
    #     """
    #     if v is None:
    #         return None

    #     mol = Chem.MolFromInchi(v) if v.startswith("InChI=") else Chem.MolFromSmiles(v)
    #     if mol is None:
    #         raise ValueError("Invalid compound_name: expected a valid SMILES or InChI")
    #     return v

    # @model_validator(mode="after")
    # def _validate_presence(self) -> "Compound":
    #     if self.compound_name is None and self.smiles is None:
    #         raise ValueError("compound_name and SMILES cannot both be None")
    #     return self


class TLCRatioResult(BaseModel):
    """MCP server output payload (inner object)."""

    property1: str
    property2: str


class TLCRatioPayload(BaseModel):
    """MCP server output payload (outer envelope)."""

    result: TLCRatioResult


class TLCAIOutput(BaseModel):
    """Class for TLC Compound Extraction AI Output, which job is to extract compound information from user input text."""

    compounds: list[Compound] = Field(..., description="List of compounds extracted from the text")


class TLCCompoundSpecItem(Compound, TLCRatioResult):
    """
    TLC-specific compound spec used for form filling + MCP lookup.

    Contains two classes of fields:

    - From `Compound`: compound_name, smiles
    - From `TLCRatioResult`: property1, property2, which should be filled by MCP server
    """


class TLCAgentOutput(TLCAIOutput):
    spec: list[TLCCompoundSpecItem] = Field(
        ...,
        description="List of compound specifications including MCP recommended ratios",
    )
    confirmed: bool = Field(default=False, description="Whether the compound form has been confirmed by user/human")


# endregion


class ExecutorKey(StrEnum):
    TLC_AGENT = "tlc_agent.run"
    # COLUMN_AGENT = "column.recommend"
    # ROBOT_TLC = "robot.tlc_spot"
    # PROPERTY_LOOKUP = "property.lookup"


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
