from __future__ import annotations

from datetime import datetime
from typing import Any

from langchain_core.messages import AnyMessage, HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt
from pydantic import BaseModel, ConfigDict, Field

from src.classes.operation import OperationInterruptPayload, OperationResponse
from src.classes.system_state import TLCAgentOutput
from src.utils.logging_config import logger
from src.utils.tools import coerce_operation_resume


class TLCAgentGraphState(BaseModel):
    """LangGraph subgraph state schema for TLCAgent (structure-only POC)."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="ignore")

    # Parent graph will pass `messages` and may pass existing `tlc_spec`.
    messages: list[AnyMessage] = Field(default_factory=list)
    tlc_spec: TLCAgentOutput | None = None

    # Internal-only routing flag.
    user_approved: bool | None = None


class TLCAgent:
    """
    TLC agent implemented as a LangGraph subgraph, with a single public entrypoint: `run()`.

    Flow:
    - extract_compound_and_fill_spec
    - user_confirm
      - revise  -> back to extract_compound_and_fill_spec
      - confirm -> done (return `tlc_spec` to parent graph)
    """

    def __init__(self) -> None:
        """Build and compile the internal LangGraph subgraph."""
        subgraph = StateGraph(TLCAgentGraphState)
        subgraph.add_node("extract_compound_and_fill_spec", self._extract_compound_and_fill_spec)
        subgraph.add_node("user_confirm", self._user_confirm)

        subgraph.add_edge(START, "extract_compound_and_fill_spec")
        subgraph.add_edge("extract_compound_and_fill_spec", "user_confirm")
        subgraph.add_conditional_edges(
            "user_confirm",
            self._route_user_confirm,
            {
                "revise": "extract_compound_and_fill_spec",
                "confirm": END,
            },
        )
        # TODO: Missing get recommended ratio

        self._compiled_subgraph = subgraph.compile()
        # Public handle for embedding into the parent graph (main flow should enter subgraph and only return on confirm).
        self.subgraph = self._compiled_subgraph

    def run(
        self,
        *,
        user_input: list[AnyMessage],
        current_form: TLCAgentOutput | None = None,
    ) -> OperationResponse[list[AnyMessage], TLCAgentOutput]:
        """
        Unified entrypoint for TLC flow.

        NOTE: Structure-only POC. `extract_compound_and_fill_spec` is intentionally a mock.
        """
        start_time = datetime.now()

        final_state = self._compiled_subgraph.invoke(
            TLCAgentGraphState(messages=list(user_input), tlc_spec=current_form),
        )
        if isinstance(final_state, dict):
            output_form = final_state.get("tlc_spec")
            messages = final_state.get("messages", list(user_input))
        else:
            output_form = getattr(final_state, "tlc_spec", None)
            messages = getattr(final_state, "messages", list(user_input))

        if not isinstance(output_form, TLCAgentOutput):
            raise TypeError(f"TLCAgent did not produce TLCAgentOutput. Got: {type(output_form)}")

        end_time = datetime.now()
        return OperationResponse[list[AnyMessage], TLCAgentOutput](
            operation_id="tlc_agent.run",
            input=list(messages),
            output=output_form,
            start_time=start_time.isoformat(timespec="microseconds"),
            end_time=end_time.isoformat(timespec="microseconds"),
        )

    @staticmethod
    def _extract_compound_and_fill_spec(state: TLCAgentGraphState) -> dict[str, Any]:
        """
        Mock extractor for POC.

        - If there is an existing `tlc_spec`, keep it as-is.
        - Otherwise create an empty skeleton `TLCAgentOutput`.
        """
        current = state.tlc_spec
        if isinstance(current, TLCAgentOutput):
            logger.debug("TLCAgent: keep existing tlc_spec (structure-only).")
            return {"tlc_spec": current}

        logger.debug("TLCAgent: create empty tlc_spec skeleton (structure-only).")
        return {"tlc_spec": TLCAgentOutput(compounds=[], spec=[], confirmed=False)}

    @staticmethod
    def _user_confirm(state: TLCAgentGraphState) -> dict[str, Any]:
        """Human-in-the-loop confirm/revise for `tlc_spec`."""
        if not isinstance(state.tlc_spec, TLCAgentOutput):
            raise TypeError("Missing `tlc_spec` before user_confirm")

        interrupt_payload = OperationInterruptPayload(
            message="Please confirm the TLC spec. Approve to confirm; reject to revise. (Structure-only POC)",
            args={"tlc_spec": state.tlc_spec.model_dump(mode="json")},
        )
        payload = interrupt(interrupt_payload.model_dump(mode="json"))
        resume = coerce_operation_resume(payload)

        updates: dict[str, Any] = {"user_approved": bool(resume.approval)}

        edited_text = (resume.comment or "").strip()
        if edited_text and not resume.approval:
            updates["messages"] = [*list(state.messages), HumanMessage(content=edited_text)]

        if resume.approval:
            updates["tlc_spec"] = state.tlc_spec.model_copy(update={"confirmed": True})

        # Optional: allow UI to send edited form JSON in `resume.data`.
        updates_from_data = TLCAgent._coerce_spec(resume.data)
        if isinstance(updates_from_data, TLCAgentOutput):
            updates["tlc_spec"] = updates_from_data

        return updates

    @staticmethod
    def _route_user_confirm(state: TLCAgentGraphState) -> str:
        return "confirm" if state.user_approved else "revise"

    @staticmethod
    def _coerce_spec(value: Any) -> TLCAgentOutput | None:
        if not isinstance(value, dict):
            return None
        spec_dict = value.get("tlc_spec") if isinstance(value.get("tlc_spec"), dict) else value
        try:
            return TLCAgentOutput.model_validate(spec_dict)
        except Exception:
            return None
