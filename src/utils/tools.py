import json
from collections.abc import Mapping, Sequence
from typing import Any

from langgraph.types import Command, Interrupt
from pydantic import BaseModel

from src.models.operation import OperationInterruptPayload, OperationResumePayload
from src.utils.logging_config import logger


def _jsonable(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump()
    if isinstance(value, Mapping):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (set, Sequence)) and not isinstance(value, (str, bytes, bytearray)):
        return [_jsonable(item) for item in value]
    return value


def _pretty(value: Any) -> str:
    """Pretty-print arbitrary objects as JSON (best-effort), for demos/debugging."""
    return json.dumps(_jsonable(value), indent=2, ensure_ascii=False, default=str)


def coerce_operation_resume(payload: Any) -> OperationResumePayload:
    """Coerce interrupt() resume payload (JSON string/dict/Pydantic model) into OperationResumePayload."""
    if payload is None:
        raise ValueError("interrupt() returned None; missing resume payload")

    if isinstance(payload, (bytes, bytearray)):
        payload = payload.decode("utf-8", errors="replace")

    if isinstance(payload, Mapping) and "resume" in payload:
        payload = payload["resume"]

    if isinstance(payload, OperationResumePayload):
        return payload

    if isinstance(payload, BaseModel):
        return OperationResumePayload.model_validate(payload.model_dump())

    if isinstance(payload, str):
        text = payload.strip()
        if not text:
            raise ValueError("interrupt() returned an empty resume string")
        return OperationResumePayload.model_validate_json(text)

    if isinstance(payload, Mapping):
        return OperationResumePayload.model_validate(dict(payload))

    return OperationResumePayload.model_validate(payload)


def terminal_approval_handler(state: dict[str, Any]) -> Command:
    """Terminal approval handler for HITL."""
    itp: Interrupt = state["__interrupt__"][0]

    interrupt_payload = OperationInterruptPayload.model_validate(itp.value)
    user_inp = input(f"\nDo you approve? (y/n): {interrupt_payload.message} \n")

    approval = user_inp.strip().lower() in {"y", "yes", "approve", "approved"}
    comment = input("Optional comment (enter to skip): ").strip()

    logger.info(f"Terminal approval handler received: approval={approval}, comment={comment}")

    return Command(resume=OperationResumePayload(approval=approval, comment=comment, data={}).model_dump())
