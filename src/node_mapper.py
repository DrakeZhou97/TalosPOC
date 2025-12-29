"""Wrap functions in Agent, turn into LangGraph nodes."""

from typing import Any

from langchain_core.messages import AIMessage

from src.agents.tlc_agent import TLCAgent
from src.classes.agent_flow_state import TLCState
from src.classes.operation import OperationResponse, OperationRouting
from src.classes.system_enum import AdmittanceState, ExecutionStatusEnum, TLCPhase
from src.classes.system_state import (
    ExecutorKey,
    HumanApproval,
    PlanningAgentOutput,
    PlanStep,
)
from src.functions.admittance import WatchDogAgent
from src.functions.human_interaction import HumanInLoop
from src.functions.intention_detection import IntentionDetectionAgent
from src.functions.planner import Planner
from src.utils.logging_config import logger
from src.utils.messages import apply_human_revision, ensure_messages

watch_dog = WatchDogAgent()
human_interact_agent = HumanInLoop()
intention_detect_agent = IntentionDetectionAgent()
planner_agent = Planner()
tlc_agent = TLCAgent()


# region <utils>


_ensure_messages = ensure_messages
_apply_human_revision = apply_human_revision


def _get_plan_output(state: TLCState) -> PlanningAgentOutput:
    plan = state.plan
    if plan is None:
        raise ValueError("Missing 'plan' in state")
    return plan.output


def _get_current_step(state: TLCState) -> tuple[int, PlanStep]:
    cursor = int(state.plan_cursor)
    return cursor, _get_plan_output(state).plan_steps[cursor]


# endregion


# region <dependent functions>


def request_user_confirm(state: TLCState) -> dict[str, OperationResponse[str, HumanApproval]]:
    """Request user confirmation on the intention detection result."""
    logger.info("Requesting user confirmation for intention detection result.")

    intention = state.intention
    if intention is None:
        raise ValueError("Missing 'intention' in state")

    reviewed = intention.output
    messages = _ensure_messages(state)
    return human_interact_agent.confirm_intention(reviewed=reviewed, messages=messages)


def user_admittance_node(
    state: TLCState,
) -> dict[str, Any]:
    """If user input is within domain and capacity of this Agent, return YES, otherwise NO."""
    messages = _ensure_messages(state)
    res = watch_dog.run(user_input=messages)

    logger.debug(
        "Admittance decision: within_domain={}, within_capacity={}",
        res.output.within_domain,
        res.output.within_capacity,
    )

    updated_messages = list(messages)
    updated_messages.append(
        AIMessage(
            content="\n".join(
                [
                    "[watchdog] user_admittance_result",
                    f"within_domain={res.output.within_domain} within_capacity={res.output.within_capacity}",
                    f"feedback={res.output.feedback}",
                ],
            ),
        ),
    )

    return {
        "admittance": res,
        "admittance_state": AdmittanceState.YES
        if res.output.within_domain and res.output.within_capacity
        else AdmittanceState.NO,
        "messages": updated_messages,
        "user_input": updated_messages,
    }


def bottom_line_handler_node(state: TLCState) -> dict[str, Any]:
    """
    Bottom line handler (fallback) for out-of-domain / out-of-capacity requests.

    This node does not participate in the main business flow. It only provides user-facing feedback and exits.
    """
    messages = _ensure_messages(state)
    feedback = ""
    if state.admittance is not None:
        feedback = str(state.admittance.output.feedback or "").strip()

    if not feedback:
        feedback = "当前请求超出系统领域/能力范围, 无法执行。请提供与小分子合成或 DMPK 实验相关的需求。"

    updated_messages = list(messages)
    updated_messages.append(
        AIMessage(
            content=f"[bottom_line_handler] rejected\n{feedback}",
        ),
    )
    return {"messages": updated_messages, "user_input": updated_messages, "bottom_line_feedback": feedback}


def intention_detection_node(state: TLCState) -> dict[str, Any]:
    """Run intention detection on user input."""
    messages = _ensure_messages(state)
    logger.info("Running intention_detection_node with {} messages", len(messages))
    res = intention_detect_agent.run(user_input=messages)
    logger.debug(f"Intention detection output: {res}")

    out = res.output
    updated_messages = list(messages)
    updated_messages.append(
        AIMessage(
            content="\n".join(
                [
                    "[intention_detection] result",
                    f"winner_id={out.winner_id}",
                    f"matched_goal_type={out.matched_goal_type}",
                    f"reason={out.reason}",
                ],
            ),
        ),
    )

    return {
        "intention": res,
        "messages": updated_messages,
        "user_input": updated_messages,
    }


def planner_node(state: TLCState) -> dict[str, Any]:
    """Generate an executable plan (plan_steps) based on the latest user messages (post-confirmation)."""
    messages = _ensure_messages(state)
    logger.info("Running planner_node with {} messages", len(messages))
    res = planner_agent.run(user_input=messages)
    logger.debug("Planner output: {}", res)

    plan_out = res.output
    steps_preview = "\n".join(
        [
            f"- {idx + 1}. {s.title} (executor={s.executor}, requires_human_approval={s.requires_human_approval})"
            for idx, s in enumerate(plan_out.plan_steps)
        ],
    )
    updated_messages = list(messages)
    updated_messages.append(
        AIMessage(
            content="\n".join(
                [
                    "[planner] plan_created",
                    f"plan_hash={plan_out.plan_hash}",
                    "steps:",
                    steps_preview or "- (no steps)",
                ],
            ),
        ),
    )

    return {"plan": res, "plan_cursor": 0, "messages": updated_messages, "user_input": updated_messages}


def plan_review_node(state: TLCState) -> dict[str, Any]:
    """
    Human-in-the-loop plan review.

    - approve: proceed to executor
    - reject (optionally with comment): revise user input and re-enter planner
    """
    plan_out = _get_plan_output(state)
    messages = _ensure_messages(state)
    return human_interact_agent.review_plan(plan_out=plan_out, messages=messages)


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
        resume = human_interact_agent.approve_step(step=step)

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


def prepare_tlc_step_node(state: TLCState) -> dict[str, Any]:
    """Mark TLC step as in-progress and normalize messages before entering the TLC subgraph."""
    cursor, step = _get_current_step(state)
    step.status = ExecutionStatusEnum.IN_PROGRESS

    messages = _ensure_messages(state)

    # Optional: allow plan args to override the latest human text input.
    input_text = str(step.args.get("input_text", "")).strip()
    if input_text:
        messages = _apply_human_revision(messages, input_text)

    logger.info("Preparing TLC step cursor={} id={} executor={}", cursor, step.id, step.executor)
    return {"plan": state.plan, "messages": messages, "user_input": messages, "tlc_phase": TLCPhase.COLLECTING}


def finalize_tlc_step_node(state: TLCState) -> dict[str, Any]:
    """Finalize TLC step after the TLC subgraph has updated `tlc_spec`."""
    cursor, step = _get_current_step(state)
    approved_spec = state.tlc_spec
    if approved_spec is None:
        raise ValueError("Missing 'tlc_spec' after executing TLC subgraph")

    step.output = {
        "agent": "tlc_agent_subgraph",
        "executor": str(step.executor),
        "args": step.args,
        "spec": approved_spec.model_dump(mode="json"),
    }
    step.status = ExecutionStatusEnum.COMPLETED

    logger.info("Finalized TLC step cursor={} id={} executor={}", cursor, step.id, step.executor)
    return {"plan": state.plan, "tlc_phase": TLCPhase.DONE}


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


# endregion


# region <router>

# all function that behave as router function should have "route_" prefix and be placed here.


def route_admittance(state: TLCState) -> str:
    """Select the next node based on the admittance decision."""
    decision = state.admittance_state or AdmittanceState.NO
    logger.info("Routing based on admittance_state={}", decision.value)
    return decision.value


def route_bottom_line(state: TLCState) -> str:
    """Always terminate after bottom line handler."""
    _ = state
    return "done"


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
        return "prepare_tlc_step"

    return "execute_unsupported"


def route_advance_todo_cursor_node(state: TLCState) -> dict[str, Any]:
    """Advance plan_cursor to the next todo."""
    cursor = int(state.plan_cursor)
    return {"plan_cursor": cursor + 1}


def route_dispatcher(state: TLCState) -> str:
    """
    Pure function router.

    - Query: route to query handler
    - Consulting: route to consulting handler
    - Operational: if no PLAN -> planner, else -> executor
    """
    intention = state.intention
    if intention is None:
        return "done"

    goal_type = intention.output.matched_goal_type
    if goal_type.value == "management":
        return "query"
    if goal_type.value == "consulting":
        return "consulting"

    if state.plan is None:
        return "planner"
    return "executor"


def route_plan_review(state: TLCState) -> str:
    """Route after plan_review HITL."""
    if state.plan_approved:
        return "approved"
    return "revise"


def dispatcher_node(state: TLCState) -> dict[str, Any]:
    """No-op node. Dispatcher logic is in `route_dispatcher`."""
    _ = state
    return {}


def consulting_handler_node(state: TLCState) -> dict[str, Any]:
    """Placeholder consulting handler (no business logic yet)."""
    messages = _ensure_messages(state)
    updated_messages = list(messages)
    updated_messages.append(AIMessage(content="[consulting] TODO: implement consulting response"))
    return {"messages": updated_messages, "user_input": updated_messages}


def query_handler_node(state: TLCState) -> dict[str, Any]:
    """Placeholder query handler (no business logic yet)."""
    messages = _ensure_messages(state)
    updated_messages = list(messages)
    updated_messages.append(AIMessage(content="[query] TODO: implement query response"))
    return {"messages": updated_messages, "user_input": updated_messages}


# endregion
