"""
TLCAgent is a specialist agent that is responsible for filling the TLC spec with user and then recommend develop solvent and ratio.

TODO: Code needs to be cleaned up.
"""

from __future__ import annotations

import time
import uuid
from datetime import datetime
from typing import Any

import httpx
from langchain.agents import create_agent
from langchain.agents.structured_output import ProviderStrategy
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage
from langchain_core.runnables.config import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, Interrupt, interrupt

from src.models.enums import TLCPhase
from src.models.operation import OperationInterruptPayload, OperationResponse
from src.models.tlc import (
    Compound,
    TLCAgentGraphState,
    TLCAgentOutput,
    TLCAIOutput,
    TLCCompoundSpecItem,
    TLCExecutionState,
    TLCRatioPayload,
    TLCRatioResult,
)
from src.presenter import present_review
from src.utils.logging_config import logger
from src.utils.messages import MessagesUtils
from src.utils.models import TLC_MODEL
from src.utils.PROMPT import TLC_AGENT_PROMPT
from src.utils.tools import coerce_operation_resume


def get_recommended_ratio(compounds: list[Compound] | None = None) -> list[TLCRatioResult]:
    """Get the recommended ratio of the TLC experiment for each compound."""
    host = "52.83.119.132"
    url = f"http://{host}:8000/api/tlc-request"

    if not compounds:
        raise ValueError("Missing compounds")

    results: list[TLCRatioResult] = []
    for idx, compound in enumerate(compounds):
        compound_name = (compound.compound_name or "").strip()
        smiles = (compound.smiles or "").strip() if compound.smiles else ""
        if not compound_name and not smiles:
            raise ValueError(f"Missing `compound_name` and `smiles` in `compounds[{idx}]`")

        payload: dict[str, str] = {"model_backend": "GAT"}
        if compound_name:
            payload["compound_name"] = "aspirin"  # TODO: substitute with compound_name
        if smiles:
            payload["smiles"] = ""  # TODO: substitute with smiles

        try:
            t0 = time.perf_counter()
            with httpx.Client(timeout=20.0) as client:
                resp = client.post(url, json=payload)
                resp.raise_for_status()
                data = resp.json()["result"]
                parsed = TLCRatioPayload.model_validate(data)
        except Exception:
            logger.exception("Failed to get TLC recommended ratio from MCP server")
            raise
        else:
            results.append(parsed.tlc_parameters)
            logger.info(
                "TLC ratio lookup ok. idx={} backend={} elapsed_ms={}",
                idx,
                parsed.tlc_parameters.backend,
                int((time.perf_counter() - t0) * 1000),
            )
            logger.info(f"Recommended ratio: {parsed.tlc_parameters}")

    return results


class TLCAgent:
    """TLC agent implemented as a LangGraph subgraph."""

    def __init__(self, *, with_checkpointer: bool = False) -> None:
        """
        Build and compile the internal LangGraph subgraph.

        Args:
            with_checkpointer: If True, compiles with MemorySaver for standalone usage with HITL.
                               If False (default), parent graph's checkpointer will be used.

        """
        logger.info("TLCAgent initialized with model={}", TLC_MODEL)

        subgraph = StateGraph(TLCAgentGraphState)

        subgraph.add_node("extract_compound_and_fill_spec", self._extract_compound_and_fill_spec)
        subgraph.add_node("present_user_confirm", self._present_user_confirm)
        subgraph.add_node("interrupt_user_confirm", self._interrupt_user_confirm)
        subgraph.add_node("present_user_revision", self._present_user_revision)
        subgraph.add_node("fill_recommended_ratio", self._fill_recommended_ratio)

        subgraph.add_edge(START, "extract_compound_and_fill_spec")
        subgraph.add_edge("extract_compound_and_fill_spec", "present_user_confirm")
        subgraph.add_edge("present_user_confirm", "interrupt_user_confirm")
        subgraph.add_conditional_edges(
            "interrupt_user_confirm",
            self._route_user_confirm,
            {
                "revise": "present_user_revision",
                "confirm": "fill_recommended_ratio",
            },
        )
        subgraph.add_edge("present_user_revision", "extract_compound_and_fill_spec")
        subgraph.add_edge("fill_recommended_ratio", END)

        checkpointer = MemorySaver() if with_checkpointer else None

        self.compiled = subgraph.compile(checkpointer=checkpointer)
        self._agent = create_agent(
            model=TLC_MODEL,
            system_prompt=TLC_AGENT_PROMPT,
            response_format=ProviderStrategy(TLCAIOutput),
        )

    # NOTE: Run() is not necessary actually

    def run(
        self,
        *,
        tlc_state: TLCAgentGraphState | Command,
        thread_id: str = str(uuid.uuid4()),
    ) -> dict[str, Any]:
        """
        Unified entrypoint for the TLC subgraph in the main graph.

        This method intentionally does NOT simulate UI or loop. If the subgraph interrupts (HITL),
        it should bubble to the outer runtime (server/UI) to resume later.

        Args:
            tlc_state: The current TLC state.
            thread_id: The thread ID for the conversation.

        Returns:
            The complete state of the TLC subgraph execution.

        """
        return self.compiled.invoke(tlc_state, config=RunnableConfig(configurable={"thread_id": thread_id}))

    @staticmethod
    def _build_response(
        final_state: dict | TLCAgentGraphState,
        original_input: list[AnyMessage],
        start_time: datetime,
    ) -> OperationResponse[list[AnyMessage], TLCAgentOutput]:
        """Extract data from final state and build OperationResponse."""
        if isinstance(final_state, dict):
            tlc = final_state.get("tlc")
            output_form = tlc.get("spec") if isinstance(tlc, dict) else getattr(tlc, "spec", None)
            messages = final_state.get("messages", original_input)
        else:
            tlc = getattr(final_state, "tlc", None)
            output_form = getattr(tlc, "spec", None)
            messages = getattr(final_state, "messages", original_input)

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

    def _extract_compound_and_fill_spec(self, state: TLCAgentGraphState) -> dict[str, Any]:
        """Extract compound info from messages and build tlc.spec draft."""
        result = self._agent.invoke({"messages": state.user_input})  # pyright: ignore[reportArgumentType]
        model_resp: TLCAIOutput = result["structured_response"]

        updated_spec: TLCAgentOutput = TLCAgentOutput(
            compounds=model_resp.compounds,
            resp_msg=model_resp.resp_msg,
            exp_params=[],
            confirmed=False,
        )

        messages = MessagesUtils.append_thinking(state.messages, updated_spec.resp_msg)

        return {"tlc": state.tlc.model_copy(update={"spec": updated_spec}), "messages": messages}

    @staticmethod
    def _fill_recommended_ratio(state: TLCAgentGraphState) -> dict[str, Any]:
        if not isinstance(state.tlc.spec, TLCAgentOutput) or not state.tlc.spec.compounds:
            raise TypeError("Missing `tlc.spec`/`compounds` before fill_recommended_ratio")

        compounds = [c for c in state.tlc.spec.compounds if c is not None]
        if any((not (c.compound_name or "").strip()) and (not (c.smiles or "").strip()) for c in compounds):
            raise ValueError("Each compound must include at least one of `compound_name` or `smiles` before requesting ratio")
        ratios = get_recommended_ratio(compounds=compounds)
        spec = [
            TLCCompoundSpecItem(
                compound_name=c.compound_name,
                smiles=c.smiles,
                solvent_system=r.solvent_system,
                ratio=r.ratio,
                rf_value=r.rf_value,
                description=r.description,
                origin=r.origin,
                backend=r.backend,
            )
            for c, r in zip(compounds, ratios, strict=True)
        ]

        messages = MessagesUtils.append_thinking(
            MessagesUtils.ensure_messages({"messages": state.messages}),
            f"[tlc] fill_ratio done. compounds={len(compounds)}",
        )
        updated = state.tlc.spec.model_copy(update={"spec": spec, "confirmed": True})
        return {"tlc": state.tlc.model_copy(update={"spec": updated, "phase": TLCPhase.CONFIRMED}), "messages": messages}

    @staticmethod
    def _present_user_confirm(state: TLCAgentGraphState) -> dict[str, Any]:
        """Presenter step before interrupt: write a clean user-visible prompt into messages."""
        if not isinstance(state.tlc.spec, TLCAgentOutput):
            raise TypeError("Missing `tlc.spec` before user_confirm")

        payload = OperationInterruptPayload(
            message="",
            args={"tlc": {"spec": state.tlc.spec.model_dump(mode="json")}},
        )

        payload.message = present_review(
            MessagesUtils.only_human_messages(MessagesUtils.ensure_messages({"messages": state.messages})),
            kind="tlc_confirm",
            args=payload.args,
        )

        messages = MessagesUtils.append_thinking(state.messages, state.tlc.spec.resp_msg)

        return {"messages": messages, "pending_interrupt": payload, "user_approved": False}

    @staticmethod
    def _interrupt_user_confirm(state: TLCAgentGraphState) -> dict[str, Any]:
        """Interrupt + apply resume for `tlc.spec` confirm/revise."""
        if state.pending_interrupt is None:
            raise TypeError("Missing `pending_interrupt` before interrupt_user_confirm")

        raw = interrupt(state.pending_interrupt.model_dump(mode="json"))
        resume = coerce_operation_resume(raw)

        updates: dict[str, Any] = {
            "pending_interrupt": None,
            "user_approved": bool(resume.approval),
            "messages": MessagesUtils.append_thinking(
                MessagesUtils.ensure_messages({"messages": state.messages}),
                f"[tlc] user_confirm resume approval={bool(resume.approval)} has_comment={bool((resume.comment or '').strip())} has_data={resume.data is not None}",
            ),
        }
        edited_text = (resume.comment or "").strip()
        if edited_text and not resume.approval:
            updates["revision_text"] = edited_text

        if resume.approval:
            if not isinstance(state.tlc.spec, TLCAgentOutput):
                raise TypeError("Missing `tlc.spec` before applying approval")
            updates["tlc"] = state.tlc.model_copy(update={"spec": state.tlc.spec.model_copy(update={"confirmed": True})})

        updates_from_data = TLCAgent._coerce_spec(resume.data)
        if isinstance(updates_from_data, TLCAgentOutput):
            updates["tlc"] = state.tlc.model_copy(update={"spec": updates_from_data})
            logger.info("TLC user_confirm applied spec from resume.data")

        return updates

    @staticmethod
    def _present_user_revision(state: TLCAgentGraphState) -> dict[str, Any]:
        """
        Presenter for human revision after TLC confirm rejection.

        This node appends the revised HumanMessage into UI `messages`. Execution context
        is stored in `thinking` markers (replace latest HumanMessage).
        """
        edited_text = (state.revision_text or "").strip()
        if not edited_text:
            return {"revision_text": None}

        messages = [*MessagesUtils.ensure_messages({"messages": state.messages}), HumanMessage(content=edited_text)]
        return {"messages": messages, "revision_text": None}

    @staticmethod
    def _route_user_confirm(state: TLCAgentGraphState) -> str:
        return "confirm" if state.user_approved else "revise"

    @staticmethod
    def _coerce_spec(value: Any) -> TLCAgentOutput | None:
        if not isinstance(value, dict):
            return None
        if isinstance(value.get("tlc"), dict) and isinstance(value["tlc"].get("spec"), dict):
            spec_dict = value["tlc"]["spec"]
        elif isinstance(value.get("spec"), dict):
            spec_dict = value["spec"]
        elif isinstance(value.get("tlc_spec"), dict):  # backward-compat
            spec_dict = value["tlc_spec"]
        else:
            spec_dict = value
        try:
            return TLCAgentOutput.model_validate(spec_dict)
        except Exception:
            return None


# tlc_agent_subgraph = TLCAgent()
# Avoid import side effects and duplicate initialization, instantiate it in node_mapper.py

if __name__ == "__main__":
    from src.utils.tools import _pretty, terminal_approval_handler

    agent = TLCAgent(with_checkpointer=True)
    thread_id = str(uuid.uuid4())
    config: RunnableConfig = RunnableConfig(configurable={"thread_id": thread_id})

    text = input("[user]: ").strip() or "我正在进行水杨酸的乙酰化反应制备乙酰水杨酸帮我进行中控监测IPC"
    print(f"user input: {text}")
    next_input: TLCAgentGraphState | Command = TLCAgentGraphState(messages=[HumanMessage(content=text)], tlc=TLCExecutionState(spec=None))
    res: dict[str, Any] = {}

    while res.get("user_approved") is None or not res.get("user_approved"):
        res = agent.run(tlc_state=next_input, thread_id=thread_id)

        print(_pretty(res))  # which is the complete subgraph state

        if "__interrupt__" in res:  # HITL interrupt
            interrupts: list[Interrupt] = res["__interrupt__"]
            print(_pretty(interrupts[0].value))  # payload for UI, not resume

            next_input = terminal_approval_handler(res)
