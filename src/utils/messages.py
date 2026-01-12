from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage

if TYPE_CHECKING:
    from src.models.core import AgentState


class MsgUtils:
    @staticmethod
    def append_thinking(thinking: list[AnyMessage], text: str) -> list[AnyMessage]:
        """Append an internal thinking AIMessage (use with state.thinking)."""
        text = str(text or "").strip()
        if not text:
            return list(thinking)
        return [*list(thinking), AIMessage(content=text, additional_kwargs={"display_label": "thinking"})]

    @staticmethod
    def ensure_thinking(state: "AgentState | Mapping[str, Any]") -> list[AnyMessage]:
        """Return a copy of thinking messages from state."""
        if isinstance(state, Mapping):
            return list(state.get("thinking") or [])
        return list(getattr(state, "thinking", None) or [])

    @staticmethod
    def append_user_message(message: list[AnyMessage], text: str) -> list[AnyMessage]:
        """Append a new HumanMessage to the conversation."""
        text = str(text or "").strip()
        if not text:
            return list(message)
        return [*list(message), HumanMessage(content=text)]

    @staticmethod
    def append_response(messages: list[AnyMessage], text: str) -> list[AnyMessage]:
        """Append a new AIMessage to the conversation."""
        text = str(text or "").strip()
        if not text:
            return list(messages)
        return [*list(messages), AIMessage(content=text)]

    @staticmethod
    def strip_thinking(messages: list[AnyMessage]) -> list[AnyMessage]:
        """Remove messages marked as internal thinking."""
        out: list[AnyMessage] = []
        for m in messages:
            kw = getattr(m, "additional_kwargs", None)
            if isinstance(kw, dict) and kw.get("display_label") == "thinking":
                continue
            out.append(m)
        return out

    @staticmethod
    def only_human_messages(messages: list[AnyMessage]) -> list[AnyMessage]:
        """Return only user-authored messages (HumanMessage) in chronological order."""
        return [m for m in messages if isinstance(m, HumanMessage)]

    @staticmethod
    def ensure_messages(state: AgentState | Mapping[str, Any]) -> list[AnyMessage]:
        """Return a copy of conversation messages (prefers `messages`, falls back to `user_input`)."""
        # LangGraph may pass raw dict state when unit-testing nodes.
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

    @staticmethod
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
