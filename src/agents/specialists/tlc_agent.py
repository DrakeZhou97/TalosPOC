"""
TLCAgent is a specialist agent that is responsible for filling the TLC spec with user and then recommend develop solvent and ratio.

TODO: Code needs to be cleaned up.
"""

from __future__ import annotations

import time
import uuid
from typing import Any

import httpx
from langchain.agents import create_agent
from langchain.agents.structured_output import ProviderStrategy
from langchain_core.messages import HumanMessage
from langchain_core.runnables.config import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, Interrupt, interrupt

from src.models.enums import TLCPhase
from src.models.operation import OperationInterruptPayload, OperationResumePayload
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
from src.utils.messages import MsgUtils
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
        subgraph.add_node("interrupt_user_confirm", self._interrupt_user_confirm)
        subgraph.add_node("fill_recommended_ratio", self._fill_recommended_ratio)

        subgraph.add_edge(START, "extract_compound_and_fill_spec")
        subgraph.add_edge("extract_compound_and_fill_spec", "interrupt_user_confirm")
        subgraph.add_conditional_edges(
            "interrupt_user_confirm",
            self._route_user_confirm,
            {
                "revise": "extract_compound_and_fill_spec",
                "confirm": "fill_recommended_ratio",
            },
        )
        subgraph.add_edge("fill_recommended_ratio", END)

        checkpointer = MemorySaver() if with_checkpointer else None

        self.compiled = subgraph.compile(checkpointer=checkpointer)
        self._agent = create_agent(
            model=TLC_MODEL,
            system_prompt=TLC_AGENT_PROMPT,
            response_format=ProviderStrategy(TLCAIOutput),
        )

    def _extract_compound_and_fill_spec(self, state: TLCAgentGraphState) -> dict[str, Any]:
        """
        Extract compound info from messages and build tlc.spec draft.

        """
        # Step 1. Invoke
        result = self._agent.invoke({"messages": [*state.messages]})
        model_resp: TLCAIOutput = result["structured_response"]

        # Step 2. Build TLCAgentOutput
        updated_agent_output: TLCAgentOutput = TLCAgentOutput(
            compounds=model_resp.compounds,
            exp_params=[],
            confirmed=False,
        )

        messages = MsgUtils.append_thinking(state.messages, str(updated_agent_output))

        return {
            "tlc": state.tlc.model_copy(update={"spec": updated_agent_output, "phase": TLCPhase.COLLECTING}),
            "messages": messages,
        }

    @staticmethod
    def _interrupt_user_confirm(state: TLCAgentGraphState) -> dict[str, Any]:
        """Interrupt + apply resume for `tlc.spec` confirm/revise."""
        if not state.tlc.spec:
            raise ValueError("No SPEC yet")

        messages = state.messages

        # Step 0. Summarize and build interrupt payload
        review_msg = present_review(messages, kind="tlc_confirm", args=state.tlc.spec.model_dump())

        interrupt_payload: OperationInterruptPayload = OperationInterruptPayload(
            message=review_msg,
            args={"tlc": {"spec": state.tlc.spec.model_dump(mode="json")}},
        )

        # Step 1. Throw Interrupt
        raw = interrupt(interrupt_payload.model_dump(mode="json"))

        # Step 2. Coerce operation resume
        resume: OperationResumePayload = coerce_operation_resume(raw)

        # Append user message
        edited_text = (resume.comment or "").strip()
        if edited_text and not resume.approval:
            messages = MsgUtils.append_user_message(messages, edited_text)

        # Step 3. Build updates
        updates: dict[str, Any] = {
            "user_approved": bool(resume.approval),
            "messages": messages,
        }

        # Spec Update
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

        messages = MsgUtils.append_thinking(
            MsgUtils.ensure_messages({"messages": state.messages}),
            f"[tlc] fill_ratio done. compounds={len(compounds)}",
        )
        updated = state.tlc.spec.model_copy(update={"spec": spec, "confirmed": True})
        return {"tlc": state.tlc.model_copy(update={"spec": updated, "phase": TLCPhase.CONFIRMED}), "messages": messages}

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
    from pathlib import Path

    from src.utils.logging_config import logger
    from src.utils.tools import _pretty, terminal_approval_handler

    agent = TLCAgent(with_checkpointer=True)

    output_path = Path(__file__).resolve().parents[3] / "assets" / "tlc_agent_workflow.png"
    agent.compiled.get_graph(xray=True).draw_mermaid_png(output_file_path=str(output_path))
    logger.success(f"Workflow exported to {output_path}")

    thread_id = str(uuid.uuid4())
    config: RunnableConfig = RunnableConfig(configurable={"thread_id": thread_id})

    text = input("[user]: ").strip() or "我正在进行水杨酸的乙酰化反应制备乙酰水杨酸帮我进行中控监测IPC"
    next_input: TLCAgentGraphState | Command = TLCAgentGraphState(messages=[HumanMessage(content=text)], tlc=TLCExecutionState(spec=None))
    res: dict[str, Any] = {}

    while True:
        interrupted = False
        for state in agent.compiled.stream(next_input, config=config, stream_mode="values"):
            if "__interrupt__" in state:
                interrupted = True
                next_input = terminal_approval_handler(state)
                break

            # 正常输出
            for msg in state["messages"]:
                print(msg + "\n")

            print("--------------------------------")

        if not interrupted:
            print("graph reached END")
            break
