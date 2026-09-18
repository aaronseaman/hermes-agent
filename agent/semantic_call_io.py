"""semantic_call inputs → chat messages, and response → validated output.

Inputs are projected explicitly by the caller; nothing from a parent transcript is added, and
the caller's ``instructions`` are sent byte-for-byte (a capability's prompt stays cache-stable).
The output schema rides as ``response_format`` and is validated HOST-side with jsonschema.
"""

from __future__ import annotations

import base64
import json
from typing import Any, Dict, List, Optional, Sequence, Tuple


def _image_part(block: Dict[str, Any]) -> Dict[str, Any]:
    url = block.get("url")
    if not url and block.get("data") is not None:
        data = block["data"]
        encoded = base64.b64encode(data).decode("ascii") if isinstance(data, (bytes, bytearray)) else str(data)
        url = f"data:{block.get('mime_type') or 'image/png'};base64,{encoded}"
    if not isinstance(url, str) or not url:
        raise ValueError("image input needs url or data")
    return {"type": "image_url", "image_url": {"url": url}}


def _content_part(block: Any) -> Dict[str, Any]:
    """str → text; ``{"type": "text"|"image", ...}`` → that part; any other JSON value → its JSON text."""
    if isinstance(block, str):
        return {"type": "text", "text": block}
    if isinstance(block, dict) and block.get("type") == "text":
        return {"type": "text", "text": str(block.get("text") or "")}
    if isinstance(block, dict) and block.get("type") == "image":
        return _image_part(block)
    return {"type": "text", "text": json.dumps(block, ensure_ascii=False, sort_keys=True, default=str)}


def build_messages(instructions: str, inputs: Sequence[Any]) -> Tuple[List[Dict[str, Any]], bool]:
    """``(messages, has_image)``: an optional system message, then ONE user message.

    Text-only inputs join into a plain string (the shape every chat wire accepts); any image
    makes the user content a parts list.
    """
    if isinstance(inputs, (str, bytes)) or not isinstance(inputs, Sequence):
        raise ValueError("inputs must be a list of input blocks")
    parts = [_content_part(block) for block in inputs]
    if not parts:
        raise ValueError("inputs must not be empty")
    has_image = any(p["type"] == "image_url" for p in parts)
    content: Any = parts if has_image else "\n\n".join(p["text"] for p in parts)
    messages: List[Dict[str, Any]] = [{"role": "system", "content": instructions}] if instructions else []
    messages.append({"role": "user", "content": content})
    return messages, has_image


def response_format(schema: Dict[str, Any], name: str, strict: bool) -> Dict[str, Any]:
    return {"type": "json_schema", "json_schema": {"name": name, "strict": strict, "schema": schema}}


def response_text(response: Any) -> str:
    """The assistant message content (``""`` when absent)."""
    return response.choices[0].message.content or ""


def check_schema(schema: Any, *, require_validator: bool = True) -> Dict[str, Any]:
    """The schema dict, or ValueError. A schema the host will enforce needs jsonschema installed:
    without it nothing could be validated, so that fails closed."""
    from tools.delegation_output_schema import coerce_output_schema
    if require_validator:
        _require_jsonschema()
    coerced, error = coerce_output_schema(schema)
    if error or coerced is None:
        raise ValueError(error or "output_schema must be a JSON Schema object")
    return coerced


def validate_output(text: str, schema: Dict[str, Any]) -> Tuple[Optional[Any], List[str]]:
    """``(parsed, [])`` when ``text`` holds JSON valid under ``schema``, else ``(None, errors)``."""
    from tools.delegation_output_schema import extract_json_candidate, validate_output as _validate
    _require_jsonschema()
    ok, errors = _validate(text, schema)
    return (json.loads(extract_json_candidate(text)), []) if ok else (None, errors)


def _require_jsonschema() -> None:
    try:
        import jsonschema  # noqa: F401
    except ImportError as exc:  # fail closed: an unvalidated "structured" output is not one
        raise RuntimeError("semantic_call output_schema needs the jsonschema package") from exc
