import json
from collections.abc import Mapping, Sequence
from typing import Any

from pydantic import BaseModel

from src.classes.operation import OperationResume


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


def coerce_operation_resume(payload: Any) -> OperationResume:
    """Coerce interrupt() resume payload (JSON string/dict/Pydantic model) into OperationResume."""
    if payload is None:
        raise ValueError("interrupt() returned None; missing resume payload")

    if isinstance(payload, (bytes, bytearray)):
        payload = payload.decode("utf-8", errors="replace")

    if isinstance(payload, Mapping) and "resume" in payload:
        payload = payload["resume"]

    if isinstance(payload, OperationResume):
        return payload

    if isinstance(payload, BaseModel):
        return OperationResume.model_validate(payload.model_dump())

    if isinstance(payload, str):
        text = payload.strip()
        if not text:
            raise ValueError("interrupt() returned an empty resume string")
        return OperationResume.model_validate_json(text)

    if isinstance(payload, Mapping):
        return OperationResume.model_validate(dict(payload))

    return OperationResume.model_validate(payload)
