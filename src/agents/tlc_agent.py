from datetime import datetime
from typing import Any

from langchain.agents import create_agent
from langchain.agents.structured_output import StructuredOutputValidationError, ToolStrategy
from langchain.tools import tool
from langchain_core.messages import AnyMessage, HumanMessage
from langgraph.types import interrupt

from src.classes.operation import OperationInterruptPayload, OperationResponse, OperationResume
from src.classes.PROMPT import TLC_AGENT_PROMPT
from src.classes.system_state import (
    Compound,
    TLCAgentOutput,
    TLCAIOutput,
    TLCCompoundSpecItem,
    TLCRatioPayload,
    TLCRatioResult,
)
from src.utils.logging_config import logger
from src.utils.models import TLC_MODEL
from src.utils.settings import ChatModelConfig, settings


@tool
def get_tlc_ratio_from_mcp(
    compound_name: str | None = None,
    smiles: str | None = None,
) -> dict:
    """
    Get TLC ratio from MCP server for a given compound.

    Args:
        compound_name: Compound name (preferred when no structure is available).
        smiles: SMILES expression (preferred when available).

    Returns:
        MCP payload: {"result": {"property1": "...", "property2": "..."}}.

    """
    # TODO: Implement actual MCP server call when resources are available
    # This is a placeholder that can be extended with actual MCP integration
    logger.debug(
        "Getting TLC ratio from MCP. compound_name={} formula={} smiles={}",
        compound_name,
        None,
        smiles,
    )
    key = smiles or compound_name or "unknown"
    payload = TLCRatioPayload(result=TLCRatioResult(property1=f"mock_property1_for_{key}", property2="mock_property2"))
    return payload.model_dump(mode="json")


class TLCAgent:
    def __init__(self) -> None:
        """Initialize the TLCAgent."""
        self.config: ChatModelConfig = settings.agents.tlc_agent
        self.tlc_agent = create_agent(
            model=TLC_MODEL,
            response_format=ToolStrategy[TLCAIOutput](TLCAIOutput),
            system_prompt=TLC_AGENT_PROMPT,
            tools=[get_tlc_ratio_from_mcp],
        )

    def run(
        self,
        user_input: list[AnyMessage],
        current_form: TLCAgentOutput | None = None,
    ) -> OperationResponse[list[AnyMessage], TLCAgentOutput]:
        """
        Run the TLC flow end-to-end (multi-turn).

        - Extract/merge compound form from conversation
        - Ask for missing fields (HITL via interrupt)
        - Ask for final confirmation (HITL via interrupt)
        - Fill MCP ratios once confirmed
        """
        start_time = datetime.now()

        messages = list(user_input)
        current_spec = current_form

        while True:
            spec_op = self.update_form(user_input=messages, current_form=current_spec)
            spec = spec_op.output
            missing = self._missing_fields(spec)

            if missing:
                interrupt_payload = OperationInterruptPayload(
                    message="Need more information to proceed. Please provide missing fields. You can reply in plain text.",
                    args={"missing_fields": missing, "current_form": spec.model_dump(mode="json")},
                )
                payload = interrupt(interrupt_payload.model_dump(mode="json"))
                resume = OperationResume(**payload)

                edited_spec = self._coerce_spec(resume.data)
                current_spec = edited_spec if edited_spec is not None else spec
                messages = self._append_user_text(messages, resume.comment)
                continue

            interrupt_payload = OperationInterruptPayload(
                message="Please confirm the TLC compound form. If you want to edit, reject and provide edits in comment, or provide edited form JSON in data.",
                args={"current_form": spec.model_dump(mode="json")},
            )
            payload = interrupt(interrupt_payload.model_dump(mode="json"))
            resume = OperationResume(**payload)

            if resume.approval:
                approved_spec = self._coerce_spec(resume.data) or spec
                approved_spec = approved_spec.model_copy(update={"confirmed": True})
                approved_spec = self.fill_ratios(approved_spec)

                end_time = datetime.now()
                return OperationResponse[list[AnyMessage], TLCAgentOutput](
                    operation_id="tlc_agent_run_001",
                    input=messages,
                    output=approved_spec,
                    start_time=start_time.isoformat(timespec="microseconds"),
                    end_time=end_time.isoformat(timespec="microseconds"),
                )

            edited_spec = self._coerce_spec(resume.data)
            current_spec = edited_spec if edited_spec is not None else spec
            messages = self._append_user_text(messages, resume.comment)

    def update_form(
        self,
        user_input: list[AnyMessage],
        current_form: TLCAgentOutput | None = None,
    ) -> OperationResponse[list[AnyMessage], TLCAgentOutput]:
        """
        Update (merge) a TLC compound form from conversation messages.

        Args:
            user_input (list[AnyMessage]): The user input messages ordered chronologically.
            current_form (TLCAgentOutput | None): Previously collected form.

        Returns:
            OperationResponse[list[AnyMessage], TLCAgentOutput]: Updated form (not confirmed).

        """
        start_time = datetime.now()

        logger.info("TLCAgent.update_form triggered with {} messages", len(user_input))

        ai_output = self._extract_compounds(input_msg=user_input, current_form=current_form)

        merged = self._merge_form(current_form=current_form, extracted=ai_output)
        end_time = datetime.now()

        return OperationResponse[list[AnyMessage], TLCAgentOutput](
            operation_id="tlc_agent_001",
            input=user_input,
            output=merged,
            start_time=start_time.isoformat(timespec="microseconds"),
            end_time=end_time.isoformat(timespec="microseconds"),
        )

    def fill_ratios(self, form: TLCAgentOutput) -> TLCAgentOutput:
        """
        Fill `spec` for each compound using MCP tool (placeholder).

        Note: should be called only after the form is confirmed.
        """
        spec_items: list[TLCCompoundSpecItem] = []

        for compound in form.compounds:
            try:
                payload = get_tlc_ratio_from_mcp.invoke(
                    {
                        "compound_name": compound.compound_name,
                        "smiles": compound.smiles,
                    },
                )
            except Exception as exc:
                logger.warning("Failed to get TLC ratio for compound {}: {}", compound.compound_name, exc)
                payload = None

            mcp_result: TLCRatioResult | None = None
            if isinstance(payload, dict):
                try:
                    mcp_result = TLCRatioPayload.model_validate(payload).result
                except Exception as exc:
                    logger.warning("Failed to parse MCP payload for compound {}: {}", compound.compound_name, exc)

            property1 = mcp_result.property1 if mcp_result is not None else "unavailable"
            property2 = mcp_result.property2 if mcp_result is not None else "unavailable"
            spec_items.append(
                TLCCompoundSpecItem(
                    compound_name=compound.compound_name,
                    smiles=compound.smiles,
                    property1=property1,
                    property2=property2,
                )
            )

        return form.model_copy(update={"spec": spec_items})

    @staticmethod
    def _coerce_spec(value: Any) -> TLCAgentOutput | None:
        if value is None:
            return None

        def _normalize_form_dict(form_dict: dict[str, Any]) -> dict[str, Any]:
            # `spec` is required by TLCAgentOutput; user/HITL payloads may omit it.
            if "spec" not in form_dict:
                form_dict = {**form_dict, "spec": []}
            return form_dict

        # Accept either {"form": {...}} or direct form JSON
        if isinstance(value, dict) and "form" in value and isinstance(value["form"], dict):
            return TLCAgentOutput.model_validate(_normalize_form_dict(value["form"]))
        if isinstance(value, dict):
            return TLCAgentOutput.model_validate(_normalize_form_dict(value))
        return None

    @staticmethod
    def _missing_fields(spec: TLCAgentOutput) -> list[str]:
        if not spec.compounds:
            return ["compounds"]
        missing: list[str] = []
        for i, c in enumerate(spec.compounds):
            if not (c.compound_name or "").strip():
                missing.append(f"compounds[{i}].compound_name")
        return missing

    @staticmethod
    def _append_user_text(messages: list[AnyMessage], text: str | None) -> list[AnyMessage]:
        edited_text = (text or "").strip()
        if not edited_text:
            return list(messages)
        return [*messages, HumanMessage(content=edited_text)]

    def _extract_compounds(
        self,
        input_msg: list[AnyMessage],
        current_form: TLCAgentOutput | None,
    ) -> TLCAIOutput:
        """
        Extract compounds from user input text.

        Args:
            input_msg (list[AnyMessage]): The user input messages.
            current_form (TLCAgentOutput | None): Previously collected form, used for merge guidance.

        Returns:
            TLCAIOutput: The extracted compounds.

        Raises:
            StructuredOutputValidationError: If the agent output validation fails.

        """
        try:
            current_form_json = (current_form or TLCAgentOutput(compounds=[], spec=[], confirmed=False)).model_dump(
                mode="json"
            )
            prompt_msg = HumanMessage(
                content=(
                    "Current compound form (JSON). Update / fill it based on the conversation. "
                    "Do NOT invent SMILES; leave it null if unknown.\n\n"
                    f"{current_form_json}"
                ),
            )
            result = self.tlc_agent.invoke(input={"messages": [*input_msg, prompt_msg]})  # type: ignore
            ai_output = result["structured_response"]

            if not isinstance(ai_output, TLCAIOutput):
                raise TypeError(
                    f"TLCAgent output is not a TLCAIOutput: {ai_output}. It is {type(ai_output)}",
                )

            logger.info(
                "TLCAgent extracted {} compounds: {}",
                len(ai_output.compounds),
                [c.compound_name for c in ai_output.compounds],
            )
        except StructuredOutputValidationError as se:
            logger.error("TLCAgent structured output validation error. error={}", se)
            raise
        else:
            return ai_output

    @staticmethod
    def _merge_form(current_form: TLCAgentOutput | None, extracted: TLCAIOutput) -> TLCAgentOutput:
        existing_form = current_form or TLCAgentOutput(compounds=[], spec=[], confirmed=False)
        existing = list(existing_form.compounds)
        by_name: dict[str, Compound] = {c.compound_name.strip().lower(): c for c in existing if c.compound_name}
        merged: list[Compound] = []

        preserved_spec = list(existing_form.spec)

        def _merge_one(old: Compound | None, new: Compound) -> Compound:
            if old is None:
                return new
            return old.model_copy(
                update={
                    "smiles": old.smiles or new.smiles,
                },
            )

        for new_c in extracted.compounds:
            key = (new_c.compound_name or "").strip().lower()
            merged.append(_merge_one(by_name.get(key), new_c))

        # Keep any existing compounds not present in extracted output
        extracted_keys = {(c.compound_name or "").strip().lower() for c in extracted.compounds}
        for old_c in existing:
            old_key = (old_c.compound_name or "").strip().lower()
            if old_key and old_key not in extracted_keys:
                merged.append(old_c)

        return TLCAgentOutput(compounds=merged, spec=preserved_spec, confirmed=False)
