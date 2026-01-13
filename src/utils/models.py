"""
Factories and singletons for chat models used by agents.

Use settings from settings.py.

The model is a singleton, so it will be cached and reused. Written in Strategy pattern.
"""

from __future__ import annotations

from typing import Protocol

from langchain_openai import ChatOpenAI

from src.utils.settings import ChatModelConfig, settings

# class PromptLogger(BaseCallbackHandler):
#     """Simple callback handler that prints prompts for debugging."""

#     def on_chat_model_start(self, _serialized: Any, messages: list[list[Any]], **_kwargs: Any) -> None:  # type: ignore[override]
#         """Log outgoing chat messages for quick inspection."""
#         for i, batch in enumerate(messages):
#             print(f"[batch {i}] === outgoing messages ===")
#             for m in batch:
#                 print(f"{m.type}: {getattr(m, 'content', m)}")
#         print("END OF PROMPTS")


class ModelStrategy(Protocol):
    """
    Strategy interface for constructing chat models.

    Provides an extension point + static type contract for the registry:
    - Callers rely on build() existing; Static type checker can verify strategy shapes.
    - Different behaviors (singleton cache, hot-reload, per-tenant choice)
      can be plugged in without touching callers.
    - Tests can inject fakes to bypass real model init.
    """

    def build(self) -> ChatOpenAI: ...


class StaticConfigStrategy(ModelStrategy):
    """Build-and-cache strategy using a static model config."""

    def __init__(self, config: ChatModelConfig) -> None:
        self._config = config
        self._model: ChatOpenAI | None = None

    def build(self) -> ChatOpenAI:
        if self._model is None:
            self._model = ChatOpenAI(**self._config.model_dump(exclude_none=True))
        return self._model


# This registry maps agent identifiers to their respective model-building strategies.
#
# Purpose:
# - Centralizes model configuration, letting different agents (e.g., watchdog, intention_detection)
#   use potentially different model configs or instantiation logic.
# - Enables easy swapping of strategies (singleton, dynamic reloading, per-tenant, etc.) without
#   requiring change in code consuming models.
# - Facilitates extension and testing—simply register a new strategy for a new agent, or swap in a mock.
#
# Benefit:
# - Enforces a single source of truth for agent model selection and construction.
# - Encourages adherence to the Strategy pattern, promoting loose coupling and maintainability.
# - Supports type-safe retrieval of models, as each strategy implements ModelStrategy.

AGENT_MODEL_STRATEGIES: dict[str, ModelStrategy] = {
    "watchdog": StaticConfigStrategy(settings.agents.watchdog),
    "intention_detection": StaticConfigStrategy(settings.agents.intention_detection),
    "tlc_agent": StaticConfigStrategy(settings.agents.tlc_agent),
    "cc_agent": StaticConfigStrategy(settings.agents.cc_agent),
    "re_agent": StaticConfigStrategy(settings.agents.re_agent),
    "planner": StaticConfigStrategy(settings.agents.planner),
}


def get_agent_model(agent_key: str) -> ChatOpenAI:
    """Retrieve a chat model by agent key."""
    try:
        return AGENT_MODEL_STRATEGIES[agent_key].build()
    except KeyError as exc:  # pragma: no cover - defensive programming
        raise KeyError(f"Unknown agent model '{agent_key}'.") from exc


WATCHDOG_MODEL = get_agent_model("watchdog")
INTENTION_DETECTION_MODEL = get_agent_model("intention_detection")
TLC_MODEL = get_agent_model("tlc_agent")
CC_MODEL = get_agent_model("cc_agent")
RE_MODEL = get_agent_model("re_agent")
PLANNER_MODEL = get_agent_model("planner")

__all__ = [
    "INTENTION_DETECTION_MODEL",
    "PLANNER_MODEL",
    "TLC_MODEL",
    "CC_MODEL",
    "RE_MODEL",
    "WATCHDOG_MODEL",
    # "PromptLogger",
    "get_agent_model",
]


if __name__ == "__main__":
    TLC_MODEL.invoke(input="What is the capital of France?").pretty_print()
