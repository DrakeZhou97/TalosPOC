from __future__ import annotations

import uuid
from typing import Any

from langchain_core.messages import AnyMessage, HumanMessage
from langgraph.types import Command, Interrupt

from src.classes.agent_flow_state import TLCState
from src.classes.operation import OperationResume
from src.main import talos_agent
from src.utils.tools import _pretty


def _parse_human_approval(inp: str) -> OperationResume:
    normalized = (inp or "").strip().lower()
    approval = normalized in {"y", "yes", "approve", "approved"}
    return OperationResume(approval=approval, comment=None)


def streaming() -> None:
    """Run the Talos LangGraph in a streaming mode for testing purposes."""
    thread_id = f"streaming-{uuid.uuid4()}"
    config: Any = {"configurable": {"thread_id": thread_id}}
    conversation: list[AnyMessage] = [
        HumanMessage(
            content="我正在进行水杨酸 (Salicylic Acid) 的 乙酰化反应 制备乙酰水杨酸 (Aspirin)。需要进行中控监测 (IPC)",
        ),
    ]

    next_input: TLCState | Command = TLCState(user_input=conversation, messages=conversation)

    while True:
        for state in talos_agent.stream(next_input, config=config, stream_mode="values"):
            print(f"Updated States: {_pretty(state)}")

            if "__interrupt__" in state:
                itp: Interrupt = state["__interrupt__"][0]

                user_inp = input(f"\nDo you approve? (y/n): {itp.value['message']} \n")
                resume = _parse_human_approval(user_inp)
                comment = input("Optional comment (enter to skip): ").strip()
                resume.comment = comment or None
                next_input = Command(resume=resume.model_dump())
                break


if __name__ == "__main__":
    streaming()
