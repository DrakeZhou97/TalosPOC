from __future__ import annotations

from typing import Any

import pytest
from langchain_core.messages import AnyMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, Interrupt
from langgraph.types import interrupt as lg_interrupt

from src import node_mapper
from src.agents.coordinators.planner import Planner
from src.main import create_talos_agent
from src.models.core import AgentState, IntentionDetectionFin, PlanningAgentOutput, PlanStep, UserAdmittance
from src.models.enums import ExecutionStatusEnum, ExecutorKey, GoalTypeEnum, TLCPhase
from src.models.operation import OperationInterruptPayload, OperationResponse, OperationResumePayload
from src.models.tlc import Compound, TLCAgentGraphState, TLCAgentOutput, TLCCompoundSpecItem
from src.utils.tools import coerce_operation_resume


def _op_response(*, operation_id: str, input_value: Any, output_value: Any) -> OperationResponse[Any, Any]:
    return OperationResponse(
        operation_id=operation_id,
        input=input_value,
        output=output_value,
        start_time="0",
        end_time="0",
    )


def _stub_planner_run(_self: Planner, *, user_input: list[AnyMessage]) -> OperationResponse[list[AnyMessage], PlanningAgentOutput]:
    step = PlanStep(
        id="s1",
        title="TLC step",
        executor=ExecutorKey.TLC_AGENT,
        args={},
        requires_human_approval=False,  # avoid extra HITL in specialist_dispatcher
        status=ExecutionStatusEnum.NOT_STARTED,
        output=None,
    )
    out = PlanningAgentOutput(plan_steps=[step], plan_hash="hash_test")
    return _op_response(operation_id="planner_test", input_value=user_input, output_value=out)


def _stub_planner_compiled_no_hitl(*, plan_out: PlanningAgentOutput) -> Any:
    """A tiny planner subgraph-as-node runnable that writes `plan` + `plan_approved=True` (no interrupts)."""

    def _finish(state: AgentState) -> dict[str, Any]:
        plan = _op_response(operation_id="planner_test", input_value=list(state.messages), output_value=plan_out)
        return {"plan": plan, "plan_cursor": 0, "plan_approved": True, "messages": list(state.messages)}

    g = StateGraph(AgentState)
    g.add_node("finish", _finish)
    g.add_edge(START, "finish")
    g.add_edge("finish", END)
    return g.compile()


def _stub_tlc_compiled_no_hitl(*, tlc_spec: TLCAgentOutput) -> Any:
    """A tiny TLC subgraph-as-node runnable that just writes `tlc_spec` (no interrupts)."""

    def _finish(state: TLCAgentGraphState) -> dict[str, Any]:
        return {"tlc": state.tlc.model_copy(update={"spec": tlc_spec}), "messages": list(state.messages)}

    g = StateGraph(TLCAgentGraphState)
    g.add_node("finish", _finish)
    g.add_edge(START, "finish")
    g.add_edge("finish", END)
    return g.compile()


def _stub_tlc_compiled_with_hitl(*, tlc_spec: TLCAgentOutput) -> Any:
    """A tiny TLC subgraph runnable with 1 interrupt, then deterministic ratio fill."""

    def _write_spec(state: TLCAgentGraphState) -> dict[str, Any]:
        _ = state
        return {"tlc": state.tlc.model_copy(update={"spec": tlc_spec.model_copy(update={"confirmed": False})})}

    def _user_confirm(state: TLCAgentGraphState) -> dict[str, Any]:
        assert state.tlc.spec is not None
        payload = OperationInterruptPayload(
            message="Please confirm the TLC spec.",
            args={"tlc": {"spec": state.tlc.spec.model_dump(mode="json")}},
        )
        resume_raw = lg_interrupt(payload.model_dump(mode="json"))
        resume = coerce_operation_resume(resume_raw)
        return {"user_approved": bool(resume.approval)}

    def _route_confirm(state: TLCAgentGraphState) -> str:
        return "confirm" if state.user_approved else "revise"

    def _fill_ratio(state: TLCAgentGraphState) -> dict[str, Any]:
        _ = state
        # Use the deterministic stubbed spec and mark confirmed; mimic TLCAgent's final output shape.
        return {"tlc": state.tlc.model_copy(update={"spec": tlc_spec.model_copy(update={"confirmed": True})})}

    g = StateGraph(TLCAgentGraphState)
    g.add_node("write_spec", _write_spec)
    g.add_node("user_confirm", _user_confirm)
    g.add_node("fill_ratio", _fill_ratio)
    g.add_edge(START, "write_spec")
    g.add_edge("write_spec", "user_confirm")
    g.add_conditional_edges("user_confirm", _route_confirm, {"revise": "user_confirm", "confirm": "fill_ratio"})
    g.add_edge("fill_ratio", END)
    return g.compile()


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

    def _intention_run(*, user_input: list[Any]) -> OperationResponse[list[Any], IntentionDetectionFin]:
        # Parallel start runs intention detection even if admittance rejects; keep this test offline by stubbing it.
        out = IntentionDetectionFin(
            matched_goal_type=GoalTypeEnum.CONSULTING,
            reason="stub",
            winner_id=GoalTypeEnum.CONSULTING.value,
            evidences=[],
        )
        return _op_response(operation_id="intention_test", input_value=user_input, output_value=out)

    monkeypatch.setattr(node_mapper.watch_dog, "run", _watchdog_run)
    monkeypatch.setattr(node_mapper.intention_detect_agent, "run", _intention_run)

    agent = _build_agent()
    msgs: list[AnyMessage] = [HumanMessage(content="what's the weather?")]
    init = AgentState(messages=msgs)
    result = agent.invoke(init, config={"configurable": {"thread_id": "t-bottom-line"}})

    assert result["bottom_line_feedback"] == "当前请求超出系统领域/能力范围, 无法执行。请提供与小分子合成或 DMPK 实验相关的需求。"
    assert len(result["messages"]) == 2


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

    agent = _build_agent()
    msgs: list[AnyMessage] = [HumanMessage(content="帮我推荐一个 TLC 条件")]
    init = AgentState(messages=msgs)
    result = agent.invoke(init, config={"configurable": {"thread_id": "t-consult"}})

    assert result["mode"] == "consulting"


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

    monkeypatch.setattr(node_mapper.watch_dog, "run", _watchdog_run)
    monkeypatch.setattr(node_mapper.intention_detect_agent, "run", _intention_run)

    step = PlanStep(
        id="s1",
        title="TLC step",
        executor=ExecutorKey.TLC_AGENT,
        args={},
        requires_human_approval=False,  # avoid extra HITL in specialist_dispatcher
        status=ExecutionStatusEnum.NOT_STARTED,
        output=None,
    )
    plan_out = PlanningAgentOutput(plan_steps=[step], plan_hash="hash_test")
    stub_planner = type("StubPlanner", (), {"compiled": _stub_planner_compiled_no_hitl(plan_out=plan_out)})()
    monkeypatch.setattr(node_mapper.planner_agent, "compiled", stub_planner.compiled)

    tlc_spec = TLCAgentOutput(
        compounds=[Compound(compound_name="Aspirin", smiles=None)],
        resp_msg="ok",
        spec=[
            TLCCompoundSpecItem(
                compound_name="Aspirin",
                smiles=None,
                solvent_system="DCM/MeOH",
                ratio="19:1",
                rf_value=None,
                description="test",
                origin="test",
                backend="test",
            ),
        ],
        confirmed=True,
    )
    stub_tlc = type("StubTLC", (), {"compiled": _stub_tlc_compiled_no_hitl(tlc_spec=tlc_spec)})()
    monkeypatch.setattr(node_mapper.tlc_agent, "compiled", stub_tlc.compiled)

    agent = _build_agent()
    msgs: list[AnyMessage] = [HumanMessage(content="我要做 TLC 点板并生成条件")]
    init = AgentState(messages=msgs)
    result = agent.invoke(init, config={"configurable": {"thread_id": "t-exec-tlc"}})

    plan: OperationResponse[Any, PlanningAgentOutput] = result["plan"]
    assert plan.output.plan_steps[0].status == ExecutionStatusEnum.COMPLETED
    tlc = result["tlc"]
    phase = tlc["phase"] if isinstance(tlc, dict) else tlc.phase
    spec = tlc["spec"] if isinstance(tlc, dict) else tlc.spec
    assert phase == TLCPhase.DONE
    assert spec is not None
    assert spec.confirmed is True


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
    1) plan review (planner subgraph)
    2) TLC spec confirmation (TLC subgraph)
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

    monkeypatch.setattr(node_mapper.watch_dog, "run", _watchdog_run)
    monkeypatch.setattr(node_mapper.intention_detect_agent, "run", _intention_run)
    monkeypatch.setattr(Planner, "run", _stub_planner_run)
    tlc_spec = TLCAgentOutput(
        compounds=[Compound(compound_name="Aspirin", smiles=None)],
        resp_msg="ok",
        spec=[
            TLCCompoundSpecItem(
                compound_name="Aspirin",
                smiles=None,
                solvent_system="DCM/MeOH",
                ratio="19:1",
                rf_value=None,
                description="test",
                origin="test",
                backend="test",
            ),
        ],
        confirmed=False,
    )
    stub_tlc = type("StubTLC", (), {"compiled": _stub_tlc_compiled_with_hitl(tlc_spec=tlc_spec)})()
    monkeypatch.setattr(node_mapper.tlc_agent, "compiled", stub_tlc.compiled)

    agent = _build_agent()
    msgs: list[AnyMessage] = [HumanMessage(content="我要做 TLC 点板并生成条件")]
    config: dict[str, Any] = {"configurable": {"thread_id": "t-stream-hitl"}}

    next_input: AgentState | Command = AgentState(messages=msgs)
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

                resume = OperationResumePayload(approval=True, comment=None, data=None)
                next_input = Command(resume=resume.model_dump())
                resumed = True
                break

        if not resumed:
            break

    assert interrupts_seen == 2
    assert last_state is not None
    tlc = last_state["tlc"]
    spec = tlc["spec"] if isinstance(tlc, dict) else tlc.spec
    assert spec is not None
    assert spec.confirmed is True
    plan: OperationResponse[Any, PlanningAgentOutput] = last_state["plan"]
    assert plan.output.plan_steps[0].status == ExecutionStatusEnum.COMPLETED
