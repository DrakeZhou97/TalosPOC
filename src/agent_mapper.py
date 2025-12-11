"""
# Wrap functions in Agent, turn into Langgraph nodes.

Also including other Langgraph specific function e.g., interrupt().
"""

from typing import Literal

from langgraph.types import interrupt

from src.classes.agent_flow_state import TLCState
from src.classes.operation import OperationResponse, OperationRouting
from src.classes.system_enum import AdmittanceState
from src.classes.system_state import HumanApproval, IntentionDetectionFin, UserAdmittance
from src.functions.admittance import WatchDogAgent
from src.functions.human_interaction import HumanInLoop
from src.functions.intention_detection import IntentionDetectionAgent
from src.functions.survey import Survey
from src.utils.logging_config import logger

watch_dog = WatchDogAgent()
human_interact_agent = HumanInLoop()
intention_detect_agent = IntentionDetectionAgent()
survey = Survey()


# region <dependent functions>


def request_user_confirm(state: TLCState) -> dict[str, OperationResponse[str, HumanApproval]]:
    """Request user confirmation on the admittance decision."""
    logger.info("Requesting user confirmation for intention detection result.")

    # reviewed = state["intention"].output  # Forward to client for human verification

    logger.info(f"Intention detection result: {state['intention']}")

    approval = interrupt("Do you Approve this intention?")  # Received approval from client

    if approval and approval["approval"]:
        logger.info("User approved via interrupt. payload={}", approval)
    else:
        logger.warning("User rejected via interrupt. payload={}", approval)

    confirmation = human_interact_agent.post_human_confirmation(
        reviewed="See reviewed attr",
        approval=approval["approval"],
        comment=approval["comment"],
    )

    return {"human_confirmation": confirmation}


def user_admittance_node(state: TLCState) -> dict[str, OperationResponse[str, UserAdmittance] | AdmittanceState]:
    """If user input is within domain and capacity of this Agent, return YES, otherwise NO."""
    if "user_input" not in state:
        raise ValueError("State must contain 'user_input' key.")

    logger.info("Running user_admittance_node with user_input='{}'", state["user_input"])
    res = watch_dog.run(user_input=state["user_input"])

    logger.debug(
        "Admittance decision: within_domain={}, within_capacity={}",
        res.output.within_domain,
        res.output.within_capacity,
    )

    return {
        "admittance": res,
        "admittance_state": AdmittanceState.YES
        if res.output.within_domain and res.output.within_capacity
        else AdmittanceState.NO,
    }


def intention_detection_node(state: TLCState) -> dict[str, OperationResponse[str, IntentionDetectionFin]]:
    """Run intention detection on user input."""
    if "user_input" not in state:
        raise ValueError("State must contain 'user_input' key.")

    logger.info("Running intention_detection_node with user_input='{}'", state["user_input"])
    res = intention_detect_agent.run(user_input=state["user_input"])
    logger.debug(f"Intention detection output: {res}")

    return {
        "intention": res,
    }


# endregion


def survey_inspect(state: TLCState) -> dict[str, dict]:
    """Inspect Survey."""
    c_survey = state.get("survey", {})

    if survey.is_finished:
        c_survey["FINISHED"] = True
    else:
        c_survey["FINISHED"] = False

    return {"survey": c_survey}


# region <router>


def route_admittance(state: TLCState) -> str:
    """Select the next node based on the admittance decision."""
    decision = state.get("admittance_state", AdmittanceState.NO)
    logger.info("Routing based on admittance_state={}", decision.value)
    return decision.value


def route_checkpoint(state: TLCState) -> Literal["PROCEED", "REVISE"]:
    """
    Checkpoint function determine if we clarify human intention and ready for execution.

    We suppose to have a state that store the survey filled status, after each round.
    This function is to check if the survey is finished or not, which means user's intention is clarified.
    """
    # TODO: Add actual logic here, simple example here
    survey = state.get("survey", {})
    if survey.get("FINISHED", False):
        logger.warning("Checkpoint: Human intention not clarified yet.")
        return "REVISE"

    return "PROCEED"


def route_human_confirm_intention(state: TLCState) -> str:
    """
    Route Human confirmation response after intention detection.

    Proceed to next node if confirmed, otherwise go back to intention detection (Revise).
    """
    human_confirmation: OperationResponse[str, HumanApproval] = state.get("human_confirmation")

    if human_confirmation and human_confirmation.output.approval:
        logger.info("Human confirmation approved, proceeding.")
        return OperationRouting.PROCEED

    logger.info("Human confirmation rejected or missing, revising.")
    return OperationRouting.REVISE


# endregion
