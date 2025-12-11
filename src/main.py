from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from src import agent_mapper
from src.classes.agent_flow_state import TLCState
from src.classes.operation import OperationRouting
from src.classes.system_enum import AdmittanceState
from src.utils.logging_config import logger

checkpointer = MemorySaver()
talos_workflow = StateGraph(TLCState)

# region <router function placeholder>


talos_workflow.add_node("user_admittance", agent_mapper.user_admittance_node)
talos_workflow.add_node("intention_detection", agent_mapper.intention_detection_node)
talos_workflow.add_node("request_user_confirm", agent_mapper.request_user_confirm)
# talos_workflow.add_node("checkpoint", agent_mapper.survey_inspect)

talos_workflow.add_edge(START, "user_admittance")
talos_workflow.add_conditional_edges(
    "user_admittance",
    agent_mapper.route_admittance,
    {
        AdmittanceState.YES.value: "intention_detection",
        AdmittanceState.NO.value: END,
    },
)
talos_workflow.add_edge("intention_detection", "request_user_confirm")
talos_workflow.add_conditional_edges(
    "request_user_confirm",
    agent_mapper.route_human_confirm_intention,
    {
        OperationRouting.PROCEED.value: END,
        OperationRouting.REVISE.value: "intention_detection",
    },
)


# endregion

logger.info("Compiling Talos workflow graph.")
talos_agent = talos_workflow.compile(checkpointer=checkpointer)
# talos_agent = talos_workflow.compile()

talos_agent.get_graph().draw_png(output_file_path="/Users/drakezhou/Development/big-poc/src/static/workflow.png")

logger.info("Workflow graph exported to {}", "/Users/drakezhou/Development/big-poc/src/static/workflow.png")
