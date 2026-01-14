from pathlib import Path
from typing import Any

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from src import node_mapper
from src.models.core import AgentState


def create_talos_workflow() -> StateGraph:
    """Create the Talos workflow definition (uncompiled)."""
    workflow = StateGraph(AgentState)

    workflow.add_node("user_admittance", node_mapper.user_admittance_node)
    workflow.add_node("intention_detection", node_mapper.intention_detection_node)
    workflow.add_node("bottom_line_handler", node_mapper.bottom_line_handler_node)
    workflow.add_node("stage_dispatcher", node_mapper.stage_dispatcher)

    workflow.add_node("consulting_handler", node_mapper.consulting_handler_node)
    workflow.add_node("query_handler", node_mapper.query_handler_node)

    # PLAN
    workflow.add_node("planner", node_mapper.planner_agent.compiled)

    workflow.add_node("specialist_dispatcher", node_mapper.specialist_dispatcher)

    # TLC
    workflow.add_node("tlc_router", node_mapper.tlc_router)
    workflow.add_node("prepare_tlc_step", node_mapper.prepare_tlc_step_node)
    workflow.add_node("tlc_agent", node_mapper.tlc_agent.compiled)
    workflow.add_node("finalize_tlc_step", node_mapper.finalize_tlc_step_node)

    # CC
    workflow.add_node("cc_router", node_mapper.cc_router)
    workflow.add_node("prepare_cc_step", node_mapper.prepare_cc_step_node)
    workflow.add_node("cc_agent", node_mapper.cc_agent.compiled)
    workflow.add_node("finalize_cc_step", node_mapper.finalize_cc_step_node)

    workflow.add_node("presenter", node_mapper.presenter_node)
    # workflow.add_node("checkpoint", node_mapper.survey_inspect)

    # Run admittance + intention detection in parallel, then join.
    workflow.add_edge(START, "user_admittance")
    workflow.add_edge(START, "intention_detection")
    workflow.add_edge(["user_admittance", "intention_detection"], "stage_dispatcher")

    # Go to bottom line handler if not admitted and END flow
    workflow.add_edge("bottom_line_handler", "presenter")

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

    # when consult
    workflow.add_edge("consulting_handler", "presenter")

    # when query
    workflow.add_edge("query_handler", "presenter")

    # when execution
    workflow.add_conditional_edges(
        "specialist_dispatcher",
        node_mapper.route_next_todo,
        {
            "planner": "planner",
            "tlc_router": "tlc_router",
            "cc_router": "cc_router",
            # "re_router": "re_router",
            # "lcms_router": "lcms_router", # e.g.
            "done": "presenter",
        },
    )

    workflow.add_edge("planner", "specialist_dispatcher")
    workflow.add_edge("presenter", END)

    # TLC Agent
    workflow.add_edge("prepare_tlc_step", "tlc_agent")
    workflow.add_edge("tlc_agent", "finalize_tlc_step")
    workflow.add_edge("finalize_tlc_step", "tlc_router")
    workflow.add_conditional_edges(
        "tlc_router",
        node_mapper.route_tlc_next_todo,
        {
            "prepare_tlc_step": "prepare_tlc_step",
            "done": "specialist_dispatcher",
        },
    )

    # CC Agent
    workflow.add_edge("prepare_cc_step", "cc_agent")
    workflow.add_edge("cc_agent", "finalize_cc_step")
    workflow.add_edge("finalize_cc_step", "cc_router")
    workflow.add_conditional_edges(
        "cc_router",
        node_mapper.route_cc_next_todo,
        {
            "prepare_cc_step": "prepare_cc_step",
            "finalize_cc_step": "finalize_cc_step",
            "done": "specialist_dispatcher",
        },
    )

    # # RE Agent
    # workflow.add_edge("prepare_re_step", "re_agent")
    # workflow.add_edge("re_agent", "finalize_re_step")
    # workflow.add_edge("finalize_re_step", "re_router")
    # workflow.add_conditional_edges(
    #     "re_router",
    #     node_mapper.route_re_next_todo,
    #     {
    #         "prepare_re_step": "prepare_re_step",
    #         "done": "specialist_dispatcher",
    #     },
    # )

    return workflow


def create_talos_agent(*, checkpointer: MemorySaver | None = None) -> Any:
    """Compile the Talos workflow into a runnable graph."""
    workflow = create_talos_workflow()
    if checkpointer is None:
        # `langgraph dev` (LangGraph API runtime) manages persistence/checkpointing.
        # Passing a custom checkpointer causes the graph loader to fail.
        return workflow.compile()
    return workflow.compile(checkpointer=MemorySaver())


talos_agent = create_talos_agent()


if __name__ == "__main__":
    import argparse
    import asyncio
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="Test the workflow")
    parser.add_argument("--export", action="store_true", help="Export the workflow to a PNG file")
    parser.add_argument(
        "--auto-approve",
        action="store_true",
        help="Auto-approve HITL interrupts during --test (non-interactive). You can also set TALOS_AUTO_APPROVE=1.",
    )
    args = parser.parse_args()

    if args.test:
        from langchain_core.messages import HumanMessage
        from langgraph.types import Command

        from src.models.operation import OperationResumePayload
        from src.utils.tools import _pretty, terminal_approval_handler

        config = {"configurable": {"thread_id": "test"}}
        agent = create_talos_agent(checkpointer=MemorySaver())

        auto_approve = bool(args.auto_approve) or os.getenv("TALOS_AUTO_APPROVE", "").strip() in {"1", "true", "yes", "y"}

        async def _run_test() -> None:
            next_input: AgentState | Command = AgentState(
                messages=[HumanMessage(content="我想执行下旋蒸分离制备乙酰水杨酸")],
            )

            while True:
                interrupted = False
                async for state in agent.astream(next_input, config=config, stream_mode="values"):
                    if "__interrupt__" in state:
                        interrupted = True
                        for itp in state["__interrupt__"]:
                            print("------------HITL----------------")
                            print(_pretty(itp.value))
                            print("--------------------------------")

                        next_input = (
                            Command(resume=OperationResumePayload(approval=True, comment=None, data={}).model_dump())
                            if auto_approve
                            else terminal_approval_handler(state)
                        )
                        break

                    for msg in state["messages"]:
                        print(msg + "\n")
                    print("--------------------------------")

                if not interrupted:
                    break

        asyncio.run(_run_test())

    if args.export:
        from src.utils.logging_config import logger

        output_path = Path(__file__).resolve().parents[1] / "assets" / "main_workflow.png"
        talos_agent.get_graph(xray=True).draw_mermaid_png(output_file_path=str(output_path))
        logger.success(f"Workflow exported to {output_path}")
