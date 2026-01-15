from langchain_core.messages import AnyMessage
from pydantic import BaseModel, Field

from src.models.enums import REPhase

# region <RE MCP>
# TODO: 一些参数可能其实是枚举类型，需要定义枚举类


class RESolvent(BaseModel):
    iupac_name: str = Field(..., description="The IUPAC name of the solvent")
    smiles: str = Field(..., description="The SMILES of the solvent")

class REBeginSpec(BaseModel):
    solvent: RESolvent = Field(..., description="The solvent to be used for the Rotary Evaporation")
    volume: float = Field(..., description="The volume (mL) of the solvent to be used for the Rotary Evaporation")

class RESoventInfo(BaseModel):
    name: str = Field(..., description="The name of the solvent")
    normal_boiling_point: float = Field(..., description="The normal boiling point (℃) of the solvent")
    boiling_point_used: float = Field(..., description="The boiling point (℃) of the solvent used")

class REPGElement(BaseModel):
    time: float
    pressure: float

class RERecommendParams(BaseModel):
    solvent_info: RESoventInfo = Field(..., description="The solvent info to be used for the Rotary Evaporation")
    bath_temperature: float = Field(..., description="The bath temperature (℃) of the Rotary Evaporation")
    coolant_temperature: float = Field(..., description="The coolant temperature (℃) of the Rotary Evaporation")
    rotation_speed: float = Field(..., description="The rotation speed (rpm) of the Rotary Evaporation")
    flask_size: int
    condenser_type: str
    pressure_gradient: list[REPGElement] = Field(..., description="The pressure gradient to be used for the Rotary Evaporation")
    solution_volume: float = Field(..., description="The volume (mL) of the solution to be used for the Rotary Evaporation")
    fill_percentage: float = Field(..., description="The fill percentage (%) of the solution to be used for the Rotary Evaporation")

class REMCPOutput(BaseModel):
    success: bool
    result: RERecommendParams
    message: str


# endregion


# region <RE Agent>


class REExecutionState(BaseModel):
    payload: REBeginSpec | RERecommendParams | None = None
    phase: REPhase = REPhase.COLLECTING


class REAgentGraphState(BaseModel):
    # Shared
    messages: list[AnyMessage] = Field(default_factory=list)
    re: REExecutionState = Field(default_factory=REExecutionState)


# endregion