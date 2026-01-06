"""Run as a package, using `python -m src.tests.main`."""

from __future__ import annotations

import json
import uuid
from typing import Any

from langchain_core.messages import AnyMessage, HumanMessage
from langgraph.types import Command, Interrupt

from src.main import create_talos_agent
from src.models.core import AgentState
from src.models.operation import OperationResumePayload
from src.utils.tools import _pretty


def _prompt(prompt: str) -> str:
    return input(prompt).strip()


def _parse_resume() -> OperationResumePayload:
    """
    Collect an OperationResumePayload from terminal input.

    - approve: y/yes/approve/approved
    - comment: optional free text
    - data: optional JSON (useful for UI-like structured edits); press enter to skip
    """
    approve_raw = _prompt("Approve? (y/n): ").lower()
    approval = approve_raw in {"y", "yes", "approve", "approved"}

    comment = _prompt("Optional comment (enter to skip): ") or None

    data_raw = _prompt("Optional JSON data (enter to skip): ")
    data: Any | None = None
    if data_raw:
        try:
            data = json.loads(data_raw)
        except json.JSONDecodeError as exc:
            print(f"[warn] Invalid JSON, ignoring. error={exc}")

    return OperationResumePayload(approval=approval, comment=comment, data=data)


def _render_last_assistant(messages: list[AnyMessage]) -> None:
    for msg in reversed(messages):
        if getattr(msg, "type", None) == "ai":
            msg.pretty_print()
            return


def main() -> None:
    """Interactive terminal runner that simulates a simple user portal for Talos agent."""
    thread_id = f"portal-{uuid.uuid4()}"
    config: dict[str, Any] = {"configurable": {"thread_id": thread_id}}
    agent = create_talos_agent()

    conversation: list[AnyMessage] = []

    while True:
        user_text = _prompt("[user] > ")
        if not user_text:
            continue

        # Explicit Exit condition
        if user_text.strip().lower() in {"/exit", "exit", "quit"}:
            break

        conversation.append(HumanMessage(content=user_text))
        next_input: AgentState | Command = AgentState(messages=list(conversation))
        last_state: dict[str, Any] | None = None

        while True:
            resumed = False
            for state in agent.stream(next_input, config=config, stream_mode="values"):
                assert isinstance(state, dict)
                last_state = state

                if "__interrupt__" in state:
                    itp: Interrupt = state["__interrupt__"][0]
                    print("\n[interrupt]")
                    print(_pretty(itp.value))
                    resume = _parse_resume()
                    next_input = Command(resume=resume.model_dump())
                    resumed = True
                    break

            if not resumed:
                break

        if last_state is not None and "messages" in last_state:
            conversation = list(last_state["messages"])
            _render_last_assistant(conversation)
        else:
            print("[warn] No final state/messages returned from graph.")


if __name__ == "__main__":
    main()
