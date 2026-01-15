from langchain_core.messages import AnyMessage
from pydantic import BaseModel, Field

from src.models.enums import CCPhase

# region <CC MCP>
# TODO: 一些参数可能其实是枚举类型，需要定义枚举类


class CCBeginSpec(BaseModel):
    sample_amount: float = Field(..., description="The amount of sample (g) to be used for the Column Chromatography")
    tlc_json_path: str = Field(..., description="The path to the TLC Result JSON file")
    tlc_data_json_path: str = Field(..., description="The path to the TLC Detailed Data JSON file")
    column_size: str | None = Field(default=None, description="The size of the column to be used for the Column Chromatography")

class CCRecommendParams(BaseModel):
    silica_amount: float
    column_size: str
    flow_rate: float
    solvent_system: str
    start_solvent_ratio: str
    end_solvent_ratio: str
    estimated_time: float
    complex_tlc: bool
    column_volume: float
    air_purge_time: float
    ai_note: str | None = None

class CCMCPOutput(BaseModel):
    success: bool
    result: CCRecommendParams
    message: str


# endregion


# region <CC Agent>


class CCExecutionState(BaseModel):
    payload: CCBeginSpec | CCRecommendParams | None = None
    phase: CCPhase = CCPhase.COLLECTING


class CCAgentGraphState(BaseModel):
    # Shared
    messages: list[AnyMessage] = Field(default_factory=list)
    cc: CCExecutionState = Field(default_factory=CCExecutionState)


# endregion