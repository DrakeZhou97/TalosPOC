from typing import TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from src.classes.operation import OperationResponse
from src.classes.system_enum import AdmittanceState
from src.classes.system_state import HumanApproval, IntentionDetectionFin
from src.functions.admittance import UserAdmittance


class TLCState(TypedDict):
    # Received Input
    messages: list[HumanMessage | AIMessage | SystemMessage]
    user_input: str
    human_confirmation: OperationResponse[str, HumanApproval]

    # Routing
    survey: dict[str, bool]

    # Internal Usage
    route_survey: str
    id_res: OperationResponse[str, UserAdmittance]

    # Agent Output
    admittance: OperationResponse[str, UserAdmittance]
    admittance_state: AdmittanceState
    intention: OperationResponse[str, IntentionDetectionFin]
    classify_res: OperationResponse[UserAdmittance, str]
