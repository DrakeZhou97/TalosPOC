"""
# Wrap functions in Agent, turn into Langgraph nodes.

Also including other Langgraph specific function e.g., interrupt().
"""

from typing import Any

from langchain_core.messages import AnyMessage, HumanMessage
from langgraph.types import interrupt

from src.agents.tlc_agent import TLCAgent
from src.classes.agent_flow_state import TLCState
from src.classes.operation import OperationInterruptPayload, OperationResponse, OperationResume, OperationRouting
from src.classes.system_enum import AdmittanceState, ExecutionStatusEnum, TLCPhase
from src.classes.system_state import (
    ExecutorKey,
    HumanApproval,
    IntentionDetectionFin,
    PlanningAgentOutput,
    PlanStep,
    UserAdmittance,
)
from src.functions.admittance import WatchDogAgent
from src.functions.human_interaction import HumanInLoop
from src.functions.intention_detection import IntentionDetectionAgent
from src.functions.planner import Planner
from src.utils.logging_config import logger

watch_dog = WatchDogAgent()
human_interact_agent = HumanInLoop()
intention_detect_agent = IntentionDetectionAgent()
planner_agent = Planner()
tlc_agent = TLCAgent()


# region <utils>


def _ensure_messages(state: TLCState) -> list[AnyMessage]:
    return list(state.messages or state.user_input or [])


def _dump_messages_for_human_review(messages: list[AnyMessage]) -> list[dict[str, Any]]:
    return [{"type": m.__class__.__name__, "content": getattr(m, "content", None)} for m in messages]


def _apply_human_revision(messages: list[AnyMessage], revised_text: str) -> list[AnyMessage]:
    revised_text = revised_text.strip()
    if not revised_text:
        return list(messages)

    updated = list(messages)
    for idx in range(len(updated) - 1, -1, -1):
        if isinstance(updated[idx], HumanMessage):
            updated[idx] = HumanMessage(content=revised_text)
            return updated

    updated.append(HumanMessage(content=revised_text))
    return updated


# endregion


# region <dependent functions>


def request_user_confirm(state: TLCState) -> dict[str, OperationResponse[str, HumanApproval]]:
    """Request user confirmation on the intention detection result."""
    logger.info("Requesting user confirmation for intention detection result.")

    intention = state.intention
    if intention is None:
        raise ValueError("Missing 'intention' in state")

    print("intention", intention)

    reviewed = intention.output

    messages = _ensure_messages(state)

    interrupt_payload = OperationInterruptPayload(
        message=f"You intention is to {reviewed.winner_id}",
        args={
            "intention": reviewed.model_dump(mode="json"),
            "current_user_input": _dump_messages_for_human_review(messages),
        },
    )
    payload = interrupt(interrupt_payload.model_dump(mode="json"))
    resume = OperationResume(**payload)

    if resume.approval:
        logger.info("User approved via interrupt. payload={}", payload)
    else:
        logger.warning("User rejected via interrupt. payload={}", payload)

    confirmation = human_interact_agent.post_human_confirmation(
        reviewed=reviewed,
        approval=resume.approval,
        comment=resume.comment,
    )

    updates: dict[str, Any] = {"human_confirmation": confirmation}

    edited_text = (resume.comment or "").strip()
    if not resume.approval and edited_text:
        revised_messages = _apply_human_revision(messages, edited_text)
        updates["messages"] = revised_messages
        updates["user_input"] = revised_messages

    return updates


def user_admittance_node(
    state: TLCState,
) -> dict[str, OperationResponse[list[AnyMessage], UserAdmittance] | AdmittanceState]:
    """If user input is within domain and capacity of this Agent, return YES, otherwise NO."""
    messages = _ensure_messages(state)
    res = watch_dog.run(user_input=messages)

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


def intention_detection_node(state: TLCState) -> dict[str, OperationResponse[list[AnyMessage], IntentionDetectionFin]]:
    """Run intention detection on user input."""
    messages = _ensure_messages(state)
    logger.info("Running intention_detection_node with {} messages", len(messages))
    res = intention_detect_agent.run(user_input=messages)
    logger.debug(f"Intention detection output: {res}")

    return {
        "intention": res,
    }


def planner_node(state: TLCState) -> dict[str, Any]:
    """Generate an executable plan (plan_steps) based on the latest user messages (post-confirmation)."""
    messages = _ensure_messages(state)
    logger.info("Running planner_node with {} messages", len(messages))
    res = planner_agent.run(user_input=messages)
    logger.debug("Planner output: {}", res)

    return {"plan": res, "plan_cursor": 0}


def _get_plan_output(state: TLCState) -> PlanningAgentOutput:
    plan = state.plan
    if plan is None:
        raise ValueError("Missing 'plan' in state")
    return plan.output


def _get_current_step(state: TLCState) -> tuple[int, PlanStep]:
    cursor = int(state.plan_cursor)
    return cursor, _get_plan_output(state).plan_steps[cursor]


def dispatch_todo_node(state: TLCState) -> dict[str, Any]:
    """Prepare next todo for execution (idempotency + optional human approval)."""
    plan_out = _get_plan_output(state)
    cursor = int(state.plan_cursor)

    while cursor < len(plan_out.plan_steps):
        step = plan_out.plan_steps[cursor]
        if step.status in {ExecutionStatusEnum.COMPLETED, ExecutionStatusEnum.CANCELLED, ExecutionStatusEnum.ON_HOLD}:
            cursor += 1
            continue
        break

    updates: dict[str, Any] = {}  # contains only what's changed
    if cursor != int(state.plan_cursor):
        updates["plan_cursor"] = cursor

    if cursor >= len(plan_out.plan_steps):
        return updates

    step = plan_out.plan_steps[cursor]
    if step.requires_human_approval:
        interrupt_payload = OperationInterruptPayload(
            message="Approve executing this step? If not, reject; optionally edit input in 'comment'.",
            args={
                "step": {
                    "id": step.id,
                    "title": step.title,
                    "executor": str(step.executor),
                    "args": step.args,
                    "requires_human_approval": step.requires_human_approval,
                    "status": step.status.value,
                },
            },
        )
        payload = interrupt(interrupt_payload.model_dump(mode="json"))
        resume = OperationResume(**payload)

        if resume.approval:
            step.requires_human_approval = False
            edited_text = (resume.comment or "").strip()
            if edited_text:
                step.args["input_text"] = edited_text
        else:
            step.status = ExecutionStatusEnum.CANCELLED
            step.output = {
                "agent": "human_review",
                "note": "Step execution rejected by human review.",
                "executor": str(step.executor),
                "args": step.args,
                "comment": resume.comment,
            }
            updates["plan"] = state.plan
            updates["plan_cursor"] = cursor + 1

    # copy exist
    if "plan" not in updates:
        updates["plan"] = state.plan
    return updates


def route_next_todo(state: TLCState) -> str:
    """Route to the next step executor based on step.executor; END when done."""
    plan_out = _get_plan_output(state)
    cursor = int(state.plan_cursor)
    if cursor >= len(plan_out.plan_steps):
        logger.info("All steps executed. cursor={} total={}", cursor, len(plan_out.plan_steps))
        return "done"

    step = plan_out.plan_steps[cursor]
    logger.info("Routing step cursor={} id={} executor={}", cursor, step.id, step.executor)

    if step.executor == ExecutorKey.TLC_AGENT:
        return "execute_tlc"

    return "execute_unsupported"


def execute_tlc_node(state: TLCState) -> dict[str, Any]:
    """Execute TLC-related step using TLCAgent (multi-turn form collection + confirmation + placeholder MCP lookup)."""
    cursor, step = _get_current_step(state)

    step.status = ExecutionStatusEnum.IN_PROGRESS
    messages = _ensure_messages(state)

    # Optional: allow plan args to override the latest human text input.
    # This is useful when planner produces a normalized instruction for the executor.
    input_text = str(step.args.get("input_text", "")).strip()
    if input_text:
        messages = _apply_human_revision(messages, input_text)

    spec_op = tlc_agent.run(user_input=messages, current_form=state.tlc_spec)
    approved_spec = spec_op.output

    step.output = {
        "agent": "tlc_agent",
        "executor": str(step.executor),
        "args": step.args,
        "spec": approved_spec.model_dump(mode="json"),
    }
    step.status = ExecutionStatusEnum.COMPLETED

    logger.info("Executed TLC step cursor={} id={} executor={}", cursor, step.id, step.executor)
    return {
        "plan": state.plan,
        "tlc_spec": approved_spec,
        "tlc_phase": TLCPhase.DONE,
        "messages": spec_op.input,
        "user_input": spec_op.input,
    }


def execute_unsupported_node(state: TLCState) -> dict[str, Any]:
    """Mark step as on-hold when no executor exists yet."""
    _cursor, step = _get_current_step(state)
    step.output = {
        "agent": "executor",
        "note": f"No executor implemented for executor '{step.executor}'.",
        "executor": str(step.executor),
        "args": step.args,
    }
    step.status = ExecutionStatusEnum.ON_HOLD
    return {"plan": state.plan}


def advance_todo_cursor_node(state: TLCState) -> dict[str, Any]:
    """Advance plan_cursor to the next todo."""
    cursor = int(state.plan_cursor)
    return {"plan_cursor": cursor + 1}


# endregion


# region <router>


def route_admittance(state: TLCState) -> str:
    """Select the next node based on the admittance decision."""
    decision = state.admittance_state or AdmittanceState.NO
    logger.info("Routing based on admittance_state={}", decision.value)
    return decision.value


def route_human_confirm_intention(state: TLCState) -> str:
    """
    Route Human confirmation response after intention detection.

    Proceed to next node if confirmed, otherwise go back to intention detection (Revise).
    """
    human_confirmation = state.human_confirmation

    if human_confirmation and human_confirmation.output.approval:
        logger.info("Human confirmation approved, proceeding.")
        return OperationRouting.PROCEED.value

    logger.info("Human confirmation rejected or missing, revising.")
    return OperationRouting.REVISE.value


# endregion
