from pathlib import Path
from typing import Any

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from src import node_mapper
from src.agents.coordinators.planner import planner_subgraph
from src.models.core import AgentState


def create_talos_workflow() -> StateGraph:
    """Create the Talos workflow definition (uncompiled)."""
    workflow = StateGraph(AgentState)

    # region <router function placeholder>
    workflow.add_node("user_admittance", node_mapper.user_admittance_node)
    workflow.add_node("intention_detection", node_mapper.intention_detection_node)
    workflow.add_node("bottom_line_handler", node_mapper.bottom_line_handler_node)
    workflow.add_node("stage_dispatcher", node_mapper.stage_dispatcher)

    workflow.add_node("consulting_handler", node_mapper.consulting_handler_node)
    workflow.add_node("query_handler", node_mapper.query_handler_node)

    workflow.add_node("planner", planner_subgraph.compiled)
    workflow.add_node("specialist_dispatcher", node_mapper.specialist_dispatcher)
    workflow.add_node("prepare_tlc_step", node_mapper.prepare_tlc_step_node)
    workflow.add_node("tlc_agent", node_mapper.tlc_agent_node)
    workflow.add_node("finalize_tlc_step", node_mapper.finalize_tlc_step_node)
    # workflow.add_node("checkpoint", node_mapper.survey_inspect)

    # region <draw>

    # Run admittance + intention detection in parallel, then join.
    workflow.add_edge(START, "user_admittance")
    workflow.add_edge(START, "intention_detection")
    workflow.add_edge(["user_admittance", "intention_detection"], "stage_dispatcher")

    # Go to bottom line handler if not admitted and END flow
    workflow.add_edge("bottom_line_handler", END)

    # Route to different handlers based on stage_dispatcher.mode.
    workflow.add_conditional_edges(
        "stage_dispatcher",
        node_mapper.route_stage_handler,
        {
            "execution": "specialist_dispatcher",  # Unified entry point for all execution ops
            "query": "query_handler",
            "consulting": "consulting_handler",
            "rejected": "bottom_line_handler",
        },
    )
    workflow.add_edge("consulting_handler", END)
    workflow.add_edge("query_handler", END)

    workflow.add_conditional_edges(
        "specialist_dispatcher",
        node_mapper.route_next_todo,
        {
            "planner": "planner",
            "prepare_tlc_step": "prepare_tlc_step",
            "done": END,
        },
    )

    workflow.add_edge("planner", "specialist_dispatcher")
    workflow.add_edge("prepare_tlc_step", "tlc_agent")
    workflow.add_edge("tlc_agent", "finalize_tlc_step")
    workflow.add_edge("finalize_tlc_step", "specialist_dispatcher")

    # endregion
    return workflow


def create_talos_agent(*, checkpointer: MemorySaver | None = None) -> Any:
    """Compile the Talos workflow into a runnable graph."""
    workflow = create_talos_workflow()
    if checkpointer is None:
        checkpointer = MemorySaver()
    return workflow.compile(checkpointer=checkpointer)


talos_agent = create_talos_agent()


def _export_workflow_png() -> None:
    output_path = Path(__file__).resolve().parents[1] / "assets" / "workflow.png"
    talos_agent.get_graph(xray=True).draw_mermaid_png(output_file_path=str(output_path))


def _test_workflow() -> None:
    from langchain_core.messages import AIMessage, HumanMessage  # noqa: PLC0415

    config = {"configurable": {"thread_id": "test"}}

    next_input = AgentState(
        messages=[],
        user_input=[],
        resp_msg=AIMessage(content=""),
    )

    for state in talos_agent.stream(next_input, config=config, stream_mode="values"):
        # print(str(state))
        print("--------------------------------")


if __name__ == "__main__":
    import sys

    if sys.argv[0] == "test":
        _test_workflow()
    else:
        _export_workflow_png()
