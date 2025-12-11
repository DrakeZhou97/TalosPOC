from uuid import uuid4

from langchain_core.messages import AnyMessage
from pydantic import BaseModel, Field, field_validator


class IDAIDetermine(BaseModel):
    goal_id: str
    matching_score: int
    reason: str

    def __str__(self) -> str:
        return f"Goal id: {self.goal_id}'s matching score is {self.matching_score}, reason: {self.reason}"


class IntentionDetectionFin(IDAIDetermine):
    winner_id: str  # matched goal_id
    evidences: list[IDAIDetermine]


class WatchDogAIDetermined(BaseModel):
    within_domain: bool = Field(..., description="表示用户输入是否在系统领域范围内")
    within_capacity: bool = Field(..., description="表示用户输入是否在系统可执行范围内")
    feedback: str = Field(..., description="对用户输入的判别依据和进一步输入建议")

    @field_validator("within_capacity")
    @classmethod
    def check_boolean(cls, v: bool) -> bool:
        """Validate that the field is a boolean."""
        # TODO: Add actual validation logic here.
        return True


class UserAdmittance(WatchDogAIDetermined):
    id: str
    user_input: list[AnyMessage]


class HumanApproval(BaseModel):
    """Represents the outcome of a human approval step."""

    id: str = Field(default_factory=lambda: str(uuid4()), description="Unique identifier for this approval decision.")
    approval: bool = Field(..., description="Whether the human reviewer approved proceeding.")
    reviewed: IntentionDetectionFin = Field(..., description="The operation response that was reviewed.")
    comment: str | None = Field(None, description="Revised user input provided when the decision is 'edit'.")
