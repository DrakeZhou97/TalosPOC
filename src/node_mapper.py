"""
LangGraph node glue for the Talos workflow.

This module keeps LangGraph-specific wiring (routing + minimal state patching).
Business logic should live in `src/agents/coordinators/` and `src/agents/executors/`.
"""

from typing import Any

from langchain_core.messages import AIMessage

from src.agents.coordinators.admittance import WatchDogAgent
from src.agents.coordinators.human_interaction import HumanInLoop
from src.agents.coordinators.intention_detection import IntentionDetectionAgent
from src.agents.specialists.tlc_agent import TLCAgent, TLCAgentGraphState
from src.models.core import AgentState, PlanningAgentOutput, PlanStep
from src.models.enums import AdmittanceState, ExecutionStatusEnum, ExecutorKey
from src.models.tlc import TLCAgentOutput, TLCPhase
from src.utils.logging_config import logger
from src.utils.messages import apply_human_revision, ensure_messages

watch_dog = WatchDogAgent()
human_interact_agent = HumanInLoop()
intention_detect_agent = IntentionDetectionAgent()
tlc_agent = TLCAgent()


# region <utils>
# NOTE: Small helpers only. Avoid business logic here.


_ensure_messages = ensure_messages
_apply_human_revision = apply_human_revision


def _get_plan_output(state: AgentState) -> PlanningAgentOutput:
    """Get `PlanningAgentOutput` from `state.plan` (raises if missing)."""
    plan = state.plan
    if plan is None:
        raise ValueError("Missing 'plan' in state")
    return plan.output


def _get_current_step(state: AgentState) -> tuple[int, PlanStep]:
    """Return `(cursor, current_step)` from `state.plan_cursor` and `state.plan.output.plan_steps`."""
    cursor = int(state.plan_cursor)
    return cursor, _get_plan_output(state).plan_steps[cursor]


# endregion


# region <function node and agent wrapper>


def user_admittance_node(state: AgentState) -> dict[str, Any]:
    """
    Run the domain/capacity gate and write the result into state.

    This node calls `WatchDogAgent` to determine whether the request is in-domain and within capacity. It also appends a trace `AIMessage` for
    debugging/observability. The decision is written back to state as `admittance_state` and the updated message list is kept in sync via
    `messages` and `user_input`.

    Args:
        state: Current workflow state.

    Returns:
        A state patch with `admittance`, `admittance_state`, and updated `messages`/`user_input`.

    """
    messages = _ensure_messages(state)
    res = watch_dog.run(user_input=messages)

    logger.debug(
        "Admittance decision: within_domain={}, within_capacity={}",
        res.output.within_domain,
        res.output.within_capacity,
    )

    return {
        "admittance": res,
        "admittance_state": AdmittanceState.YES if res.output.within_domain and res.output.within_capacity else AdmittanceState.NO,
    }


def bottom_line_handler_node(state: AgentState) -> dict[str, Any]:
    """
    Return a user-facing rejection message and exit the main flow.

    This node is the fallback when the request is rejected (out-of-domain/out-of-capacity). It formats a user-facing explanation (preferably from
    the admittance agent output) and appends a trace message. It does not participate in the execution flow beyond providing feedback.

    Args:
        state: Current workflow state.

    Returns:
        A state patch with updated `messages`/`user_input` and `bottom_line_feedback`.

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


def intention_detection_node(state: AgentState) -> dict[str, Any]:
    """
    Run intention detection and store the result in state.

    This node delegates to `IntentionDetectionAgent` and appends a trace message containing the classification result. It is used to route the
    workflow into query/consulting/planning/execution stages. The raw `OperationResponse` is stored on `state.intention`.

    Args:
        state: Current workflow state.

    Returns:
        A state patch with `intention` and updated `messages`/`user_input`.

    """
    messages = _ensure_messages(state)
    logger.info("Running intention_detection_node with {} messages", len(messages))
    res = intention_detect_agent.run(user_input=messages)
    logger.debug(f"Intention detection output: {res}")

    return {
        "intention": res,
    }


def consulting_handler_node(state: AgentState) -> dict[str, Any]:
    """
    Placeholder consulting handler.

    This node currently contains no consulting business logic and only appends a TODO trace message. It exists to keep the stage routing stable
    while consulting functionality is under development. It should not mutate any other workflow fields.

    Args:
        state: Current workflow state.

    Returns:
        A state patch with updated `messages`/`user_input`.

    """
    messages = _ensure_messages(state)
    updated_messages = list(messages)
    updated_messages.append(AIMessage(content="[consulting] TODO: implement consulting response"))
    return {"messages": updated_messages, "user_input": updated_messages}


def query_handler_node(state: AgentState) -> dict[str, Any]:
    """
    Placeholder query handler.

    This node currently contains no query/management business logic and only appends a TODO trace message. It exists to keep the stage routing
    stable while query functionality is under development. It should not mutate any other workflow fields.

    Args:
        state: Current workflow state.

    Returns:
        A state patch with updated `messages`/`user_input`.

    """
    messages = _ensure_messages(state)
    updated_messages = list(messages)
    updated_messages.append(AIMessage(content="[query] TODO: implement query response"))
    return {"messages": updated_messages, "user_input": updated_messages}


def prepare_tlc_step_node(state: AgentState) -> dict[str, Any]:
    """
    Prepare the current TLC plan step before entering the TLC subgraph.

    This node marks the current plan step as `IN_PROGRESS` and normalizes `messages`/`user_input` for the TLC subgraph. If a per-step human review
    provided an edited `input_text`, it is applied as the latest human revision so the subgraph sees the corrected prompt. It also sets
    `tlc.phase=COLLECTING` to indicate the TLC agent should start/continue its form-filling loop.

    If `step.args["input_text"]` is present (e.g., from per-step human approval), treat it as the latest human revision and apply it to messages.

    Args:
        state: Current workflow state.

    Returns:
        A state patch with updated `plan`, `messages`/`user_input`, and `tlc.phase=COLLECTING`.

    """
    cursor, step = _get_current_step(state)
    step.status = ExecutionStatusEnum.IN_PROGRESS

    messages = _ensure_messages(state)

    # Optional override from per-step approval: treat as "latest human input" for the TLC subgraph.
    input_text = str(step.args.get("input_text", "")).strip()
    if input_text:
        messages = _apply_human_revision(messages, input_text)

    logger.info("Preparing TLC step cursor={} id={} executor={}", cursor, step.id, step.executor)
    return {"plan": state.plan, "messages": messages, "user_input": messages, "tlc": state.tlc.model_copy(update={"phase": TLCPhase.COLLECTING})}


def finalize_tlc_step_node(state: AgentState) -> dict[str, Any]:
    """
    Finalize the current TLC plan step after the TLC subgraph completes.

    This node expects `state.tlc.spec` to have been produced/confirmed by the TLC subgraph. It writes the confirmed spec into the current
    `PlanStep.output`, marks the step `COMPLETED`, and advances `plan_cursor`. It also sets `tlc_phase=DONE` so upstream nodes can treat the TLC
    execution as finished for this step.

    Args:
        state: Current workflow state.

    Returns:
        A state patch with updated `plan`, `tlc.phase=DONE`, and `plan_cursor` advanced by 1.

    Raises:
        ValueError: If `state.tlc.spec` is missing after the TLC subgraph completes.

    """
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
    return {"plan": state.plan, "tlc": state.tlc.model_copy(update={"phase": TLCPhase.DONE}), "plan_cursor": cursor + 1}


# endregion


# region <router>

# Router helpers:
# - Router functions should have a `route_` prefix.
# - Keep routing rules here; keep business logic in agents.


def stage_dispatcher(state: AgentState) -> dict[str, Any]:
    """
    Compute the next stage and write it into `state.mode`.

    This node is intended to run as a join/barrier after both `user_admittance_node` and `intention_detection_node` complete. It translates those
    results into a single `mode` string (`query`/`consulting`/`planner`/`execution`/`rejected`) and stores it on state for downstream routing. The
    actual edge selection is handled by `route_stage_handler`.

    Args:
        state: Current workflow state.

    Returns:
        A state patch containing `mode` plus updated `messages`/`user_input`.

    """
    if state.admittance_state is None:
        raise ValueError("Missing 'admittance_state' before stage_dispatcher")
    if state.intention is None:
        raise ValueError("Missing 'intention' before stage_dispatcher")

    messages = _ensure_messages(state)
    updated_messages = list(messages)

    # Append admittance and intention detection result to messages
    if state.admittance is not None:
        updated_messages.append(
            AIMessage(
                content="\n".join(
                    [
                        "[watchdog] user_admittance_result",
                        f"within_domain={state.admittance.output.within_domain} within_capacity={state.admittance.output.within_capacity}",
                        f"feedback={state.admittance.output.feedback}",
                    ],
                ),
            ),
        )

    updated_messages.append(
        AIMessage(
            content="\n".join(
                [
                    "[intention_detection] result",
                    f"winner_id={state.intention.output.winner_id}",
                    f"matched_goal_type={state.intention.output.matched_goal_type}",
                    f"reason={state.intention.output.reason}",
                ],
            ),
        ),
    )

    if state.admittance_state == AdmittanceState.NO:
        return {"mode": "rejected", "messages": updated_messages, "user_input": updated_messages}

    goal = state.intention.output.matched_goal_type.value
    if goal == "management":
        return {"mode": "query", "messages": updated_messages, "user_input": updated_messages}
    if goal == "consulting":
        return {"mode": "consulting", "messages": updated_messages, "user_input": updated_messages}
    if goal == "execution" and (state.plan is None or not state.plan_approved):
        return {"mode": "planner", "messages": updated_messages, "user_input": updated_messages}
    if goal == "execution":
        return {"mode": "execution", "messages": updated_messages, "user_input": updated_messages}

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
    Normalize the plan cursor and optionally run per-step human approval.

    This node advances the plan cursor past already-finished steps and optionally calls `HumanInLoop.approve_step` when the current step requires
    approval. It may mutate the current `PlanStep` in-place (status/args/output/flags) and update `plan_cursor` based on the approval outcome. It
    returns a minimal state patch containing only the fields that changed.

    This node may mutate the current `PlanStep` in-place (approval flags, status, args/output) and advance `plan_cursor` accordingly.

    Args:
        state: Current workflow state.

    Returns:
        A state patch that may include updated `plan` and/or `plan_cursor`.

    """
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


def route_next_todo(state: AgentState) -> str:
    """
    Route to the next executor node for the current plan step.

    This router reads `state.plan.output.plan_steps` with `state.plan_cursor` and decides which executor-specific node to enter next. When all
    steps are done (cursor out of range), it returns `"done"`. Currently only `ExecutorKey.TLC_AGENT` is supported and maps to
    `"prepare_tlc_step"`.

    Args:
        state: Current workflow state.

    Returns:
        `"done"` if cursor is out of range; `"prepare_tlc_step"` for `ExecutorKey.TLC_AGENT`; otherwise `"done"`.

    """
    # Navigate to planner if plan is not generated yet.
    try:
        plan_out = _get_plan_output(state)
    except ValueError:
        return "planner"

    cursor = int(state.plan_cursor)
    if cursor >= len(plan_out.plan_steps):
        logger.info("All steps executed. cursor={} total={}", cursor, len(plan_out.plan_steps))
        return "done"

    step = plan_out.plan_steps[cursor]
    logger.info("Routing step cursor={} id={} executor={}", cursor, step.id, step.executor)

    if step.executor == ExecutorKey.TLC_AGENT:
        return "prepare_tlc_step"

    return "done"


# endregion


# region <specialists wrapper>
def tlc_agent_node(state: AgentState) -> dict[str, Any]:
    """Run TLC subgraph and map its output into `state.tlc` namespace."""
    out = tlc_agent.compiled.invoke(TLCAgentGraphState(messages=list(state.messages), tlc_spec=state.tlc.spec))

    tlc_spec = out.get("tlc_spec") if isinstance(out, dict) else None
    if tlc_spec is not None and not isinstance(tlc_spec, TLCAgentOutput):
        raise TypeError(f"Expected TLCAgentOutput from tlc subgraph, got {type(tlc_spec)}")

    messages = out.get("messages") if isinstance(out, dict) else None
    updates: dict[str, Any] = {"tlc": state.tlc.model_copy(update={"spec": tlc_spec})}
    if isinstance(messages, list) and messages:
        updates["messages"] = messages
        updates["user_input"] = messages
    return updates


# endregion
