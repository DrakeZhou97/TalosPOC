from __future__ import annotations

from typing import Any

import pytest
from langchain_core.messages import AnyMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, Interrupt

from src import node_mapper
from src.classes.agent_flow_state import TLCState
from src.classes.operation import OperationResponse, OperationResume
from src.classes.system_enum import ExecutionStatusEnum, GoalTypeEnum, TLCPhase
from src.classes.system_state import (
    Compound,
    HumanApproval,
    IntentionDetectionFin,
    PlanningAgentOutput,
    PlanStep,
    TLCAgentOutput,
    TLCCompoundSpecItem,
    UserAdmittance,
)
from src.main import create_talos_agent


def _op_response(*, operation_id: str, input_value: Any, output_value: Any) -> OperationResponse[Any, Any]:
    return OperationResponse(
        operation_id=operation_id,
        input=input_value,
        output=output_value,
        start_time="0",
        end_time="0",
    )


def _stub_tlc_subgraph(*, tlc_spec: TLCAgentOutput) -> Any:
    """Build a tiny subgraph that just writes `tlc_spec` (LangGraph v1 subgraph-as-node pattern)."""

    def _finish(state: TLCState) -> dict[str, Any]:
        return {"tlc_spec": tlc_spec, "messages": list(state.messages), "tlc_phase": TLCPhase.DONE}

    g = StateGraph(TLCState)
    g.add_node("finish", _finish)
    g.add_edge(START, "finish")
    g.add_edge("finish", END)
    return g.compile()


def _auto_confirm_intention(state: TLCState) -> dict[str, Any]:
    """Test stub: skip interrupt() and auto-approve intention confirmation."""
    if state.intention is None:
        raise ValueError("Missing 'intention' in state")
    reviewed = state.intention.output
    confirmation = _op_response(
        operation_id="human_confirmation_test",
        input_value="auto",
        output_value=HumanApproval(approval=True, reviewed=reviewed, comment=None),
    )
    return {"human_confirmation": confirmation}


def _auto_approve_plan(_state: TLCState) -> dict[str, Any]:
    """Test stub: skip interrupt() and auto-approve the plan."""
    return {"plan_approved": True}


def _build_agent() -> Any:
    # Build a fresh graph so monkeypatched nodes/subgraphs are picked up.
    return create_talos_agent(checkpointer=MemorySaver())


def test_bottom_line_handler_flow(monkeypatch: pytest.MonkeyPatch) -> None:
    """Out-of-domain/out-of-capacity should route to BottomLine handler and end."""

    def _watchdog_run(*, user_input: list[Any]) -> OperationResponse[list[Any], UserAdmittance]:
        out = UserAdmittance(
            id="adm_1",
            user_input=user_input,
            within_domain=False,
            within_capacity=False,
            feedback="out of domain",
        )
        return _op_response(operation_id="watchdog_test", input_value=user_input, output_value=out)

    monkeypatch.setattr(node_mapper.watch_dog, "run", _watchdog_run)

    agent = _build_agent()
    msgs: list[AnyMessage] = [HumanMessage(content="what's the weather?")]
    init = TLCState(messages=msgs, user_input=msgs)
    result = agent.invoke(init, config={"configurable": {"thread_id": "t-bottom-line"}})

    assert result["bottom_line_feedback"] == "out of domain"
    assert any("[bottom_line_handler] rejected" in m.content for m in result["messages"])


def test_consulting_route(monkeypatch: pytest.MonkeyPatch) -> None:
    """Consulting intent should route to consulting handler and end (no planner/executor)."""

    def _watchdog_run(*, user_input: list[Any]) -> OperationResponse[list[Any], UserAdmittance]:
        out = UserAdmittance(
            id="adm_1",
            user_input=user_input,
            within_domain=True,
            within_capacity=True,
            feedback="ok",
        )
        return _op_response(operation_id="watchdog_test", input_value=user_input, output_value=out)

    def _intention_run(*, user_input: list[Any]) -> OperationResponse[list[Any], IntentionDetectionFin]:
        out = IntentionDetectionFin(
            matched_goal_type=GoalTypeEnum.CONSULTING,
            reason="consult",
            winner_id=GoalTypeEnum.CONSULTING.value,
            evidences=[],
        )
        return _op_response(operation_id="intention_test", input_value=user_input, output_value=out)

    monkeypatch.setattr(node_mapper.watch_dog, "run", _watchdog_run)
    monkeypatch.setattr(node_mapper.intention_detect_agent, "run", _intention_run)
    monkeypatch.setattr(node_mapper, "request_user_confirm", _auto_confirm_intention)

    agent = _build_agent()
    msgs: list[AnyMessage] = [HumanMessage(content="帮我推荐一个 TLC 条件")]
    init = TLCState(messages=msgs, user_input=msgs)
    result = agent.invoke(init, config={"configurable": {"thread_id": "t-consult"}})

    assert any("[consulting]" in m.content for m in result["messages"])


def test_execution_tlc_subgraph_flow(monkeypatch: pytest.MonkeyPatch) -> None:
    """Execution intent + plan should route into executor and run the TLC subgraph."""

    def _watchdog_run(*, user_input: list[Any]) -> OperationResponse[list[Any], UserAdmittance]:
        out = UserAdmittance(
            id="adm_1",
            user_input=user_input,
            within_domain=True,
            within_capacity=True,
            feedback="ok",
        )
        return _op_response(operation_id="watchdog_test", input_value=user_input, output_value=out)

    def _intention_run(*, user_input: list[Any]) -> OperationResponse[list[Any], IntentionDetectionFin]:
        out = IntentionDetectionFin(
            matched_goal_type=GoalTypeEnum.EXECUTION,
            reason="execute",
            winner_id=GoalTypeEnum.EXECUTION.value,
            evidences=[],
        )
        return _op_response(operation_id="intention_test", input_value=user_input, output_value=out)

    def _planner_run(*, user_input: list[Any]) -> OperationResponse[list[Any], PlanningAgentOutput]:
        step = PlanStep(
            id="s1",
            title="TLC step",
            executor=node_mapper.ExecutorKey.TLC_AGENT,
            args={},
            requires_human_approval=False,
            status=ExecutionStatusEnum.NOT_STARTED,
            output=None,
        )
        out = PlanningAgentOutput(plan_steps=[step], plan_hash="hash_test")
        return _op_response(operation_id="planner_test", input_value=user_input, output_value=out)

    tlc_spec = TLCAgentOutput(
        compounds=[Compound(compound_name="Aspirin", smiles=None)],
        spec=[TLCCompoundSpecItem(compound_name="Aspirin", smiles=None, property1="p1", property2="p2")],
        confirmed=True,
    )

    monkeypatch.setattr(node_mapper.watch_dog, "run", _watchdog_run)
    monkeypatch.setattr(node_mapper.intention_detect_agent, "run", _intention_run)
    monkeypatch.setattr(node_mapper.planner_agent, "run", _planner_run)
    monkeypatch.setattr(node_mapper, "request_user_confirm", _auto_confirm_intention)
    monkeypatch.setattr(node_mapper, "plan_review_node", _auto_approve_plan)
    monkeypatch.setattr(node_mapper.tlc_agent, "subgraph", _stub_tlc_subgraph(tlc_spec=tlc_spec))

    agent = _build_agent()
    msgs: list[AnyMessage] = [HumanMessage(content="我要做 TLC 点板并生成条件")]
    init = TLCState(messages=msgs, user_input=msgs)
    result = agent.invoke(init, config={"configurable": {"thread_id": "t-exec-tlc"}})

    plan: OperationResponse[Any, PlanningAgentOutput] = result["plan"]
    assert plan.output.plan_steps[0].status == ExecutionStatusEnum.COMPLETED
    assert result["tlc_phase"] == TLCPhase.DONE
    assert result["tlc_spec"].confirmed is True


def test_individual_node_execution_user_admittance(monkeypatch: pytest.MonkeyPatch) -> None:
    """LangGraph recommended: test nodes via `compiled_graph.nodes[...]` without running the full graph."""

    def _watchdog_run(*, user_input: list[Any]) -> OperationResponse[list[Any], UserAdmittance]:
        out = UserAdmittance(
            id="adm_1",
            user_input=user_input,
            within_domain=True,
            within_capacity=True,
            feedback="ok",
        )
        return _op_response(operation_id="watchdog_test", input_value=user_input, output_value=out)

    monkeypatch.setattr(node_mapper.watch_dog, "run", _watchdog_run)
    agent = _build_agent()

    msgs: list[AnyMessage] = [HumanMessage(content="帮我做个 TLC 点板分析")]
    updates = agent.nodes["user_admittance"].invoke({"messages": msgs, "user_input": msgs})

    assert updates["admittance_state"].value == "yes"


def test_streaming_hitl_resume_two_interrupts(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    LangGraph recommended: test streaming + HITL by catching `__interrupt__` and resuming with Command(resume=...).

    This test covers two interrupts in a single run:
    1) intention confirmation
    2) plan review
    """

    def _watchdog_run(*, user_input: list[Any]) -> OperationResponse[list[Any], UserAdmittance]:
        out = UserAdmittance(
            id="adm_1",
            user_input=user_input,
            within_domain=True,
            within_capacity=True,
            feedback="ok",
        )
        return _op_response(operation_id="watchdog_test", input_value=user_input, output_value=out)

    def _intention_run(*, user_input: list[Any]) -> OperationResponse[list[Any], IntentionDetectionFin]:
        out = IntentionDetectionFin(
            matched_goal_type=GoalTypeEnum.EXECUTION,
            reason="execute",
            winner_id=GoalTypeEnum.EXECUTION.value,
            evidences=[],
        )
        return _op_response(operation_id="intention_test", input_value=user_input, output_value=out)

    def _planner_run(*, user_input: list[Any]) -> OperationResponse[list[Any], PlanningAgentOutput]:
        step = PlanStep(
            id="s1",
            title="TLC step",
            executor=node_mapper.ExecutorKey.TLC_AGENT,
            args={},
            requires_human_approval=False,  # avoid extra HITL in dispatch_todo
            status=ExecutionStatusEnum.NOT_STARTED,
            output=None,
        )
        out = PlanningAgentOutput(plan_steps=[step], plan_hash="hash_test")
        return _op_response(operation_id="planner_test", input_value=user_input, output_value=out)

    tlc_spec = TLCAgentOutput(
        compounds=[Compound(compound_name="Aspirin", smiles=None)],
        spec=[TLCCompoundSpecItem(compound_name="Aspirin", smiles=None, property1="p1", property2="p2")],
        confirmed=True,
    )

    # Keep real interrupting nodes: request_user_confirm + plan_review_node.
    # Stub all LLM calls + TLC subgraph to avoid network/tooling.
    monkeypatch.setattr(node_mapper.watch_dog, "run", _watchdog_run)
    monkeypatch.setattr(node_mapper.intention_detect_agent, "run", _intention_run)
    monkeypatch.setattr(node_mapper.planner_agent, "run", _planner_run)
    monkeypatch.setattr(node_mapper.tlc_agent, "subgraph", _stub_tlc_subgraph(tlc_spec=tlc_spec))

    agent = _build_agent()
    msgs: list[AnyMessage] = [HumanMessage(content="我要做 TLC 点板并生成条件")]
    config: dict[str, Any] = {"configurable": {"thread_id": "t-stream-hitl"}}

    next_input: TLCState | Command = TLCState(messages=msgs, user_input=msgs)
    last_state: dict[str, Any] | None = None
    interrupts_seen = 0

    while True:
        resumed = False
        for state in agent.stream(next_input, config=config, stream_mode="values"):
            assert isinstance(state, dict)
            last_state = state

            if "__interrupt__" in state:
                interrupts_seen += 1
                itp: Interrupt = state["__interrupt__"][0]
                assert "message" in itp.value

                resume = OperationResume(approval=True, comment=None, data=None)
                next_input = Command(resume=resume.model_dump())
                resumed = True
                break

        if not resumed:
            break

    assert interrupts_seen == 2
    assert last_state is not None
    assert last_state["tlc_spec"].confirmed is True
    plan: OperationResponse[Any, PlanningAgentOutput] = last_state["plan"]
    assert plan.output.plan_steps[0].status == ExecutionStatusEnum.COMPLETED
