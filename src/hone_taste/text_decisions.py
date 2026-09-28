"""Answer DecisionClient questions with any TextClient (one JSON-schema completion per `decide` call).

This is how the audience panel runs on a plain chat model (OpenAI SDK, Ollama, LangChain ...). The
answers are the model's own statements, so `calibrated` is always `False`.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from hone_taste.ports import Question, TextClient, TraceContext, get_field

SYSTEM = (
    "You answer questions about the given material. Follow each question's instructions, give a short "
    "rationale, and reply with JSON matching the schema only."
)


def _answer_schema(question: Question) -> dict[str, Any]:
    kind = question["type"]
    if kind == "yes_no":
        return {"type": "boolean"}
    if kind == "choice":
        return {"type": "string", "enum": list(question["options"])}
    low, high = question.get("scale", (1, 5))
    return {"type": "number", "minimum": low, "maximum": high}


def _schema(questions: Mapping[str, Question]) -> dict[str, Any]:
    properties = {
        name: {
            "type": "object",
            "properties": {"rationale": {"type": "string"}, "answer": _answer_schema(q)},
            "required": ["rationale", "answer"],
        }
        for name, q in questions.items()
    }
    return {"type": "object", "properties": properties, "required": list(questions)}


def _describe(name: str, question: Question) -> str:
    lines = [f"[{name}] ({question['type']}) {question['instructions']}"]
    if question["type"] == "choice":
        lines.append(f"Options: {', '.join(question['options'])}")
    if question["type"] == "score":
        low, high = question.get("scale", (1, 5))
        lines.append(f"Answer with a number from {low} to {high}.")
    anchors: Mapping[str, str] = question.get("anchors") or {}
    if anchors:
        lines.append("Anchors: " + "; ".join(f"{k} = {v}" for k, v in anchors.items()))
    return "\n".join(lines)


def _to_answer(question: Question, reply: Any) -> dict[str, Any]:
    """Convert one parsed `{"answer": ..., "rationale": ...}` object into a DecisionClient answer."""
    kind = question["type"]
    answer: dict[str, Any] = {"type": kind, "calibrated": False, "rationale": None, "error": None}
    if not isinstance(reply, Mapping) or "answer" not in reply:
        return {**answer, "value": None, "error": "no answer in the model output"}
    raw: Any = get_field(reply, "answer")
    answer["rationale"] = str(get_field(reply, "rationale") or "") or None
    if kind == "yes_no" and isinstance(raw, bool):
        return {**answer, "value": float(raw), "probabilities": {"yes": float(raw), "no": 1.0 - raw}}
    if kind == "choice" and raw in question["options"]:
        return {**answer, "value": 1.0, "choice": raw}
    if kind == "score" and isinstance(raw, int | float) and not isinstance(raw, bool):
        low, high = question.get("scale", (1, 5))
        value = min(1.0, max(0.0, (float(raw) - low) / (high - low)))
        return {**answer, "value": value, "raw": float(raw)}
    return {**answer, "value": None, "error": f"invalid {kind} answer: {raw!r}"}


@dataclass(slots=True)
class TextDecisionClient:
    """A `DecisionClient` built on a `TextClient`.

    >>> from hone_taste.testing import FakeTextClient
    >>> text = FakeTextClient(['{"q": {"answer": 4, "rationale": "catchy"}}'])
    >>> answers = TextDecisionClient(text).decide("la", {"q": {"type": "score", "instructions": "Catchy?"}})
    >>> answers["q"]["value"]
    0.75
    """

    client: TextClient
    params: Mapping[str, Any] | None = None  # extra params for complete(), e.g. {"temperature": 0}

    def decide(
        self,
        state: str | Mapping[str, Any],
        questions: Mapping[str, Question],
        *,
        images: Sequence[str] = (),
        trace: TraceContext | None = None,
    ) -> dict[str, Any]:
        material = state if isinstance(state, str) else json.dumps(state, ensure_ascii=False, indent=1)
        prompt = f"Material:\n{material}\n\nQuestions:\n" + "\n\n".join(
            _describe(n, q) for n, q in questions.items()
        )
        content: Any = prompt
        if images:
            content = [{"type": "text", "text": prompt}, *({"type": "image", "path": p} for p in images)]
        messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": content}]
        result = self.client.complete(
            messages, schema=_schema(questions), trace=trace, **dict(self.params or {})
        )
        parsed, error = _parse(result)
        if error:
            return {
                n: {"type": q["type"], "value": None, "calibrated": False, "error": error}
                for n, q in questions.items()
            }
        return {name: _to_answer(q, parsed.get(name)) for name, q in questions.items()}


def _parse(result: Any) -> tuple[dict[str, Any], str]:
    """The parsed JSON object from a TextResult, or an error message."""
    if get_field(result, "error"):
        return {}, str(get_field(result, "error"))
    parsed = get_field(result, "parsed")
    if parsed is None:
        try:
            parsed = json.loads(get_field(result, "text") or "")
        except json.JSONDecodeError:
            return {}, "model output is not JSON"
    if not isinstance(parsed, dict):
        return {}, "model output is not a JSON object"
    return parsed, ""  # pyright: ignore[reportUnknownVariableType]
