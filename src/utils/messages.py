from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from langchain_core.messages import AnyMessage, HumanMessage

if TYPE_CHECKING:
    from src.classes.agent_flow_state import TLCState


def ensure_messages(state: TLCState | Mapping[str, Any]) -> list[AnyMessage]:
    """
    Return a copy of the current conversation messages.

    Prefer `state.messages`; fall back to `state.user_input` for compatibility.
    """
    # NOTE: When unit-testing individual nodes via `compiled_graph.nodes[...]`,
    # LangGraph may pass the raw state as a dict rather than a Pydantic model.
    if isinstance(state, Mapping):
        messages = state.get("messages") or []
        if messages:
            return list(messages)
        user_input = state.get("user_input") or []
        if user_input:
            return list(user_input)
        return []

    if state.messages:
        return list(state.messages)
    if state.user_input:
        return list(state.user_input)
    return []


def dump_messages_for_human_review(messages: list[AnyMessage]) -> list[dict[str, Any]]:
    """Serialize messages for UI review (LangGraph interrupt payload)."""
    return [{"type": m.__class__.__name__, "content": getattr(m, "content", None)} for m in messages]


def apply_human_revision(messages: list[AnyMessage], revised_text: str) -> list[AnyMessage]:
    """Apply a human revision to the latest HumanMessage (or append a new one)."""
    revised_text = revised_text.strip()
    if not revised_text:
        return list(messages)

    updated = list(messages)
    for idx in range(len(updated) - 1, -1, -1):
        if isinstance(updated[idx], HumanMessage):
            updated[idx] = HumanMessage(content=revised_text)
            return updated

    updated.append(HumanMessage(content=revised_text))
    return updated
