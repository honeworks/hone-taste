"""`TextClient` over the OpenAI Python SDK: OpenAI, Ollama (`/v1`), vLLM and other compatible servers.

    pip install hone-taste[openai]

>>> from hone_taste.adapters.openai import OpenAITextClient
>>> client = OpenAITextClient("gemma3:12b", base_url="http://127.0.0.1:11434/v1", api_key="ollama")
>>> client.model
'gemma3:12b'
"""

from __future__ import annotations

import base64
import json
import mimetypes
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

from hone_taste.errors import MissingExtra, ModelUnavailable
from hone_taste.ports import TextResult, TraceContext

PARAMS = ("temperature", "top_p", "seed", "max_tokens", "stop")  # other params are ignored (TextClient port)
THINK = re.compile(r"<think>.*?</think>", re.DOTALL)
FENCE = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL)


def _image_part(part: Mapping[str, Any]) -> dict[str, Any]:
    if "path" in part:
        mime = mimetypes.guess_type(str(part["path"]))[0] or "image/png"
        data = base64.b64encode(Path(part["path"]).read_bytes()).decode()
    else:
        mime, data = part.get("mime", "image/png"), part["data_b64"]
    return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{data}"}}


def _message(message: Mapping[str, Any]) -> dict[str, Any]:
    content = message["content"]
    if isinstance(content, str):
        return {"role": message["role"], "content": content}
    parts = [_image_part(p) if p.get("type") == "image" else dict(p) for p in content]
    return {"role": message["role"], "content": parts}


def _parse_json(text: str, schema: Mapping[str, Any]) -> tuple[Any, str | None]:
    """The JSON object in `text` (ignoring <think> blocks and code fences), checked for required keys."""
    body = THINK.sub("", text).strip()
    fenced = FENCE.match(body)
    try:
        parsed: Any = json.loads(fenced.group(1) if fenced else body)
    except json.JSONDecodeError:
        return None, "response is not valid JSON"
    keys: set[str] = set(cast(dict[str, Any], parsed)) if isinstance(parsed, dict) else set()
    missing = [k for k in schema.get("required", []) if k not in keys]
    if missing:
        return None, f"response misses required keys {missing}"
    return cast(Any, parsed), None


class OpenAITextClient:
    """A `TextClient` calling `chat.completions` of an OpenAI-compatible server.

    Pass `client=` (an `openai.OpenAI`) or `base_url` / `api_key` (default: the `OPENAI_API_KEY` env var).
    `defaults` are request params used on every call (e.g. `temperature=0`). Transport errors raise the
    SDK's exceptions; a response that is not valid JSON for a schema request sets `error`.
    """

    def __init__(
        self,
        model: str,
        *,
        client: Any = None,
        base_url: str | None = None,
        api_key: str | None = None,
        **defaults: Any,
    ) -> None:
        if client is None:
            try:
                import openai  # noqa: PLC0415 - optional extra, imported only when this adapter is used
            except ImportError as exc:
                raise MissingExtra(
                    "the OpenAI adapter needs the openai SDK: pip install hone-taste[openai]"
                ) from exc
            client = openai.OpenAI(base_url=base_url, api_key=api_key)
        self.model = model
        self.client = client
        self.defaults = defaults

    def complete(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        schema: Mapping[str, Any] | None = None,
        trace: TraceContext | None = None,
        **params: Any,
    ) -> TextResult:
        request = {k: v for k, v in {**self.defaults, **params}.items() if k in PARAMS}
        if schema is not None:
            request["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "answer", "schema": schema},
            }
        try:
            response = self.client.chat.completions.create(
                model=self.model, messages=[_message(m) for m in messages], **request
            )
        except Exception as exc:  # SDK / network errors -> a RuntimeError subclass (TextClient port)
            raise ModelUnavailable(f"{self.model}: {type(exc).__name__}: {exc}") from exc
        choice = response.choices[0]
        text = choice.message.content or ""
        parsed, error = _parse_json(text, schema) if schema is not None else (None, None)
        usage = response.usage
        tokens = (
            {"input_tokens": usage.prompt_tokens, "output_tokens": usage.completion_tokens} if usage else {}
        )
        return TextResult(
            text=text,
            parsed=parsed,
            error=error,
            model=response.model or self.model,
            finish_reason=choice.finish_reason,
            usage=tokens,
        )
