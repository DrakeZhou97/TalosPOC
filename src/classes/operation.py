from enum import Enum
from typing import Any, TypeVar

from pydantic import BaseModel, Field

T = TypeVar("T")

K = TypeVar("K")


class OperationRouting(str, Enum):
    PROCEED = "proceed"
    REVISE = "revise"


class OperationRunning(BaseModel):
    pass


class OperationResume(BaseModel):
    approval: bool = Field(..., description="Whether the operation is approved to resume.")
    comment: str | None = Field(default=None, description="Human reviewing feedback")
    data: Any | None = Field(
        default=None, description="Optional structured data payload from UI (e.g., edited form JSON)"
    )


class OperationResponse[T, K](BaseModel):
    operation_id: str = Field(..., description="Unique identifier for the operation.")
    input: T = Field(
        ...,
        description="The input parameters for the operation. Must be JSON serializable and stored in KV",
    )
    output: K = Field(
        ...,
        description="The output results of the operation. Must be JSON serializable and stored in KV",
    )
    start_time: str | None = Field(..., description="The start time of the operation.")
    end_time: str | None = Field(..., description="The end time of the operation.")

    model_config = {"arbitrary_types_allowed": True}


class OperationInterruptPayload(BaseModel):
    """HITL Operation Interrupt Payload Class."""

    message: str = Field(..., description="Message describe the current operation that need human approve")
    args: Any = Field(..., description="other meta data come alone with operation")
