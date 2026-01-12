"""
LangGraph node glue for the Talos workflow.

This module keeps LangGraph-specific wiring (routing + minimal state patching).
Business logic should live in `src/agents/coordinators/` and `src/agents/executors/`.
"""

import json
from typing import Any

from copilotkit.langgraph import copilotkit_emit_state
from langchain_core.messages import SystemMessage
from langchain_core.runnables.config import RunnableConfig

from src.agents.coordinators.admittance import WatchDogAgent
from src.agents.coordinators.intention_detection import IntentionDetectionAgent
from src.agents.coordinators.planner import PlannerAgent
from src.agents.specialists.presenter import present_final
from src.agents.specialists.tlc_agent import TLCAgent
from src.models.core import AgentState
from src.models.enums import AdmittanceState, ExecutionStatusEnum, ExecutorKey, TLCPhase
from src.models.planner import PlannerAgentOutput, PlanStep
from src.models.tlc import TLCExecutionState
from src.utils.logging_config import logger
from src.utils.messages import MsgUtils

# Step messages for CopilotKit frontend progress display
STEP_MESSAGES = {
    "user_admittance_node": "Checking if request is within system capabilities...",
    "intention_detection_node": "Analyzing your intent and matching to available operations...",
    "stage_dispatcher": "Determining the best workflow for your request...",
    "prepare_tlc_step_node": "Preparing TLC step for execution...",
    "finalize_tlc_step_node": "Finalizing TLC step results...",
}

watch_dog = WatchDogAgent()
intention_detect_agent = IntentionDetectionAgent()
tlc_agent = TLCAgent()
planner_agent = PlannerAgent()


# region <utils>
# NOTE: Small helpers only. Avoid business logic here.


def _get_plan_output(state: AgentState) -> PlannerAgentOutput:
    """Get `PlannerAgentOutput` from `state.plan` (raises if missing)."""
    plan = state.plan
    if plan is None:
        raise ValueError("Missing 'plan' in state")
    return plan


def _get_current_step(state: AgentState) -> tuple[int, PlanStep]:
    """Return `(cursor, current_step)` from `state.plan_cursor` and `state.plan.plan_steps`."""
    cursor = int(state.plan_cursor)
    return cursor, _get_plan_output(state).plan_steps[cursor]


def presenter_node(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
    """
    Final presenter node (single exit): compose and sanitize user-visible messages.

    This node is the only place that should emit the final assistant answer into
    `messages`. It rebuilds `messages` to contain only human inputs + the final
    assistant reply, preventing intermediate drafts/logs from leaking.
    """
    messages = MsgUtils.ensure_messages(state)

    context = {
        "mode": state.mode,
        "bottom_line_feedback": state.bottom_line_feedback,
        "admittance": state.admittance.output.model_dump(mode="json") if state.admittance else None,
        "intention": state.intention.output.model_dump(mode="json") if state.intention else None,
        "plan": state.plan.model_dump(mode="json") if state.plan else None,
        "plan_cursor": int(state.plan_cursor),
        "tlc": state.tlc.model_dump(mode="json"),
    }

    ctx_msg = SystemMessage(content=f"CONTEXT_JSON:\n{json.dumps(context, ensure_ascii=False)}")

    final_text = present_final([ctx_msg, *MsgUtils.only_human_messages(messages)], config=config)
    return {"messages": MsgUtils.append_response(messages, final_text)}


# endregion


# region <function node and agent wrapper>


async def user_admittance_node(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
    """
    Run the domain/capacity gate and write the result into state.

    This node calls `WatchDogAgent` to determine whether the request is in-domain and within capacity. It also appends a trace `AIMessage` for
    debugging/observability. The decision is written back to state as `admittance_state` and the updated message list is kept in sync via
    `messages` and `user_input`.

    Args:
        state: Current workflow state.
        config: LangGraph runnable config for CopilotKit state emission.

    Returns:
        A state patch with `admittance`, `admittance_state`, and updated `messages`/`user_input`.

    """
    # Emit step progress to frontend
    await copilotkit_emit_state(config, {
        "current_step": "user_admittance_node",
        "step_message": STEP_MESSAGES["user_admittance_node"],
    })

    messages = MsgUtils.ensure_messages(state)
    res = watch_dog.run(user_input=messages)

    logger.debug(
        "[LG][Admittance Node] Admittance decision: within_domain={}, within_capacity={}",
        res.output.within_domain,
        res.output.within_capacity,
    )

    # Note: Don't return current_step/step_message here - this node runs in parallel
    # with intention_detection_node, and LangGraph can't merge concurrent updates
    return {
        "admittance": res,
        "admittance_state": AdmittanceState.YES if res.output.within_domain and res.output.within_capacity else AdmittanceState.NO,
    }


async def intention_detection_node(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
    """
    Run intention detection and store the result in state.

    This node delegates to `IntentionDetectionAgent` and appends a trace message containing the classification result. It is used to route the
    workflow into query/consulting/planning/execution stages. The raw `OperationResponse` is stored on `state.intention`.

    Args:
        state: Current workflow state.
        config: LangGraph runnable config for CopilotKit state emission.

    Returns:
        A state patch with `intention` and updated `messages`/`user_input`.

    """
    # Emit step progress to frontend
    await copilotkit_emit_state(config, {
        "current_step": "intention_detection_node",
        "step_message": STEP_MESSAGES["intention_detection_node"],
    })

    messages = MsgUtils.ensure_messages(state)
    res = intention_detect_agent.run(user_input=messages)

    logger.debug(
        "[LG][Intention Detection Node] Intention detection decision: winner_id={}, matched_goal_type={}, reason={}",
        res.output.winner_id,
        res.output.matched_goal_type,
        res.output.reason,
    )

    # Note: Don't return current_step/step_message here - this node runs in parallel
    # with user_admittance_node, and LangGraph can't merge concurrent updates
    return {
        "intention": res,
    }


def bottom_line_handler_node(state: AgentState) -> dict[str, Any]:
    """
    Return a user-facing rejection message and exit the main flow.

    This node is the fallback when the request is rejected (out-of-domain/out-of-capacity). It formats a user-facing explanation (preferably from
    the admittance agent output) and appends a trace message. It does not participate in the execution flow beyond providing feedback.

    Args:
        state: Current workflow state.

    Returns:
        A state patch with updated `thinking` and `bottom_line_feedback`.

    """
    logger.debug("[LG][Bottom Line Handler Node] Bottom line handler node triggered")

    feedback = "当前请求超出系统领域/能力范围, 无法执行。请提供与小分子合成或 DMPK 实验相关的需求。"

    thinking = MsgUtils.append_thinking(MsgUtils.ensure_thinking(state), f"[bottom_line_handler] rejected\n{feedback}")
    return {"thinking": thinking, "bottom_line_feedback": feedback}


def consulting_handler_node(state: AgentState) -> dict[str, Any]:
    """
    Placeholder consulting handler.

    This node currently contains no consulting business logic and only appends a TODO trace message. It exists to keep the stage routing stable
    while consulting functionality is under development. It should not mutate any other workflow fields.

    Args:
        state: Current workflow state.

    Returns:
        A state patch with updated `thinking`.

    """
    logger.debug("[LG][Consulting Handler Node] Consulting handler node triggered")

    thinking = MsgUtils.append_thinking(MsgUtils.ensure_thinking(state), "[consulting] TODO: implement consulting response")
    return {"thinking": thinking}


def query_handler_node(state: AgentState) -> dict[str, Any]:
    """
    Placeholder query handler.

    This node currently contains no query/management business logic and only appends a TODO trace message. It exists to keep the stage routing
    stable while query functionality is under development. It should not mutate any other workflow fields.

    Args:
        state: Current workflow state.

    Returns:
        A state patch with updated `thinking`.

    """
    logger.debug("[LG][Query Handler Node] Query handler node triggered")

    thinking = MsgUtils.append_thinking(MsgUtils.ensure_thinking(state), "[query] TODO: implement query response")
    return {"thinking": thinking}


# endregion


# region <router>

# Router helpers:
# - Router functions should have a `route_` prefix.
# - Keep routing rules here; keep business logic in agents.


async def stage_dispatcher(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
    """
    Compute the next stage and write it into `state.mode`.

    This node is intended to run as a join/barrier after both `user_admittance_node` and `intention_detection_node` complete. It translates those
    results into a single `mode` string (`query`/`consulting`/`planner`/`execution`/`rejected`) and stores it on state for downstream routing. The
    actual edge selection is handled by `route_stage_handler`.

    Args:
        state: Current workflow state.
        config: LangGraph runnable config for CopilotKit state emission.

    Returns:
        A state patch containing `mode` plus updated `thinking`.

    """
    # Emit step progress to frontend
    await copilotkit_emit_state(config, {
        "current_step": "stage_dispatcher",
        "step_message": STEP_MESSAGES["stage_dispatcher"],
    })

    if state.admittance_state is None:
        raise ValueError("Missing 'admittance_state' before stage_dispatcher")
    if state.intention is None:
        raise ValueError("Missing 'intention' before stage_dispatcher")

    thinking = MsgUtils.ensure_thinking(state)
    adm = state.admittance.output if state.admittance is not None else None
    itn = state.intention.output

    resp_msg = "\n".join(
        [
            "[stage_dispatcher] join results",
            f"admittance_state={state.admittance_state}",
            f"within_domain={getattr(adm, 'within_domain', None)}, within_capacity={getattr(adm, 'within_capacity', None)}",
            f"winner_id={itn.winner_id}",
            f"matched_goal_type={itn.matched_goal_type}",
            f"reason={itn.reason}",
        ],
    )

    base_result = {"current_step": None, "step_message": None}
    thinking = MsgUtils.append_thinking(thinking, resp_msg)
    logger.debug("[LG][Stage Dispatcher Node] Stage dispatcher node triggered: {}", resp_msg)

    if state.admittance_state == AdmittanceState.NO:
        return {"mode": "rejected", "thinking": thinking, **base_result}

    goal = state.intention.output.matched_goal_type.value
    if goal == "management":
        return {"mode": "query", "thinking": thinking, **base_result}
    if goal == "consulting":
        return {"mode": "consulting", "thinking": thinking, **base_result}
    if goal == "execution":
        return {"mode": "execution", "thinking": thinking, **base_result}

    raise ValueError(f"Unknown matched_goal_type={goal!r}")


def route_stage_handler(state: AgentState) -> str:
    """
    Route to the stage handler based on `state.mode`.

    This router is expected to run immediately after `stage_dispatcher`, which writes `mode`. It keeps stage computation and stage routing
    separated so the graph wiring in `main.py` stays explicit and easy to change.

    Args:
        state: Current workflow state. (mode: str)

    Returns:
        The route key in `state.mode` (defaults to `"rejected"` if missing/empty).

    """
    return str(state.mode or "").strip() or "rejected"


def specialist_dispatcher(state: AgentState) -> dict[str, Any]:
    """
    Advance the plan cursor to the next step that is not finished and update system status.

    Args:
        state: Current workflow state.

    Returns:
        A state patch that may include updated `plan` and/or `plan_cursor`.

    """
    try:
        plan = _get_plan_output(state)
    except ValueError:
        return {}

    cursor = int(state.plan_cursor)

    while cursor < len(plan.plan_steps):
        step = plan.plan_steps[cursor]
        # Only skip steps that are truly finished. `ON_HOLD` should not be auto-skipped.
        if step.status in {ExecutionStatusEnum.COMPLETED, ExecutionStatusEnum.CANCELLED}:
            cursor += 1
            continue
        break

    # Return only what actually changed; do not write back `plan` unless mutated.
    if cursor != int(state.plan_cursor):
        return {"plan_cursor": cursor}
    return {}


def route_next_todo(state: AgentState) -> str:
    """Pure router. Route according to plan_cursor."""

    # Navigate to planner if plan is not generated yet.
    try:
        plan_out = _get_plan_output(state)
    except ValueError:
        return "planner"

    cursor = int(state.plan_cursor)

    # Cursor >= Steps means all done.
    if cursor >= len(plan_out.plan_steps):
        logger.info("All steps executed. cursor={} total={}", cursor, len(plan_out.plan_steps))
        return "done"

    # Subagents
    step = plan_out.plan_steps[cursor]

    logger.info("Routing step cursor={} id={} executor={}", cursor, step.id, step.executor)

    if step.executor == ExecutorKey.TLC_AGENT:
        return "tlc_router"

    return "done"


# endregion


# region <TLC>


def tlc_router(state: AgentState) -> dict[str, Any]:
    """Route to the TLC router."""
    return {}


def route_tlc_next_todo(state: AgentState) -> str:
    """Route to the TLC next todo."""

    if state.tlc.spec is None and state.tlc.phase != TLCPhase.CONFIRMED:
        return "prepare_tlc_step"
    if state.tlc.phase == TLCPhase.CONFIRMED:
        return "done"

    return "done"


async def prepare_tlc_step_node(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
    """Update state value and validate device status."""
    # Emit step progress to frontend
    await copilotkit_emit_state(config, {
        "current_step": "prepare_tlc_step_node",
        "step_message": STEP_MESSAGES["prepare_tlc_step_node"],
    })

    thinking = MsgUtils.append_thinking(MsgUtils.ensure_thinking(state), "[prepare_tlc_step] Device status validated")

    return {
        "tlc": TLCExecutionState(phase=TLCPhase.COLLECTING),
        "thinking": thinking,
        "current_step": None,
        "step_message": None,
    }


async def finalize_tlc_step_node(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
    """
    Finalize the current TLC plan step after the TLC subgraph completes.

    This node expects `state.tlc.spec` to have been produced/confirmed by the TLC subgraph. It writes the confirmed spec into the current
    `PlanStep.output`, marks the step `COMPLETED`, and advances `plan_cursor`. It also sets `tlc_phase=DONE` so upstream nodes can treat the TLC
    execution as finished for this step.

    Args:
        state: Current workflow state.
        config: LangGraph runnable config for CopilotKit state emission.

    Returns:
        A state patch with updated `plan`, `tlc.phase=DONE`, and `plan_cursor` advanced by 1.

    Raises:
        ValueError: If `state.tlc.spec` is missing after the TLC subgraph completes.

    """
    # Emit step progress to frontend
    await copilotkit_emit_state(config, {
        "current_step": "finalize_tlc_step_node",
        "step_message": STEP_MESSAGES["finalize_tlc_step_node"],
    })

    cursor, step = _get_current_step(state)
    approved_spec = state.tlc.spec
    if approved_spec is None:
        raise ValueError("Missing 'tlc.spec' after executing TLC subgraph")

    step.output = {
        "agent": "tlc_agent_subgraph",
        "executor": str(step.executor),
        "args": step.args,
        "spec": approved_spec.model_dump(mode="json"),
    }
    step.status = ExecutionStatusEnum.COMPLETED

    logger.info("Finalized TLC step cursor={} id={} executor={}", cursor, step.id, step.executor)
    return {
        "plan": state.plan,
        "tlc": state.tlc.model_copy(update={"phase": TLCPhase.DONE, "spec": approved_spec}),
        "plan_cursor": cursor + 1,
        "current_step": None,
        "step_message": None,
    }


# endregion
