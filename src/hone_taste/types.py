"""The score value object, the Scorer protocol and the one concrete scorer class.

Every scorer in hone-taste is a `FunctionScorer`: metadata plus a plain function that turns an input into
a `Score`. The class checks the input kind first and turns any exception into `Score(None, error=...)`,
so "could not score" is always `None`, never `0`.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from hone_taste._tracing import start_span
from hone_taste.errors import ConfigError


@dataclass(frozen=True, slots=True)
class Score:
    """A 0-1 score (higher = better / more human-like). `value is None` means "could not score".

    >>> Score(0.8, reason="few AI-isms").value
    0.8
    """

    value: float | None
    confidence: float | None = None
    reason: str = ""
    details: Mapping[str, Any] = field(default_factory=dict[str, Any])
    error: str = ""


@runtime_checkable
class Scorer(Protocol):
    """Anything callable on one input that returns a `Score` and describes itself."""

    name: str
    family: str
    accepts: frozenset[str]
    license: str
    source_data: str

    def __call__(self, input: Any, /) -> Score: ...


def _is_path(value: Any) -> bool:
    return isinstance(value, str | os.PathLike)


def _has_str_keys(value: Any, *keys: str) -> bool:
    return isinstance(value, Mapping) and all(isinstance(value.get(k), str) for k in keys)  # type: ignore[union-attr]


def _is_text_with_prompt(value: Any) -> bool:
    return _has_str_keys(value, "prompt", "response")


def _is_image_with_prompt(value: Any) -> bool:
    return _has_str_keys(value, "prompt") and _is_path(value.get("image"))


INPUT_KINDS: dict[str, Callable[[Any], bool]] = {
    "text": lambda value: isinstance(value, str),
    "text_with_prompt": _is_text_with_prompt,
    "audio": _is_path,
    "image": _is_path,
    "image_with_prompt": _is_image_with_prompt,
}


def input_kind_error(accepts: frozenset[str], value: Any) -> str:
    """Return "" when `value` fits one of the `accepts` kinds, else an error message."""
    if any(INPUT_KINDS[kind](value) for kind in accepts):
        return ""
    return f"accepts {', '.join(sorted(accepts))}; got {type(value).__name__}"


def score_safely(scorer: Callable[[Any], Score], input: Any) -> Score:
    """Call any scorer (ours or a user's); an exception becomes `Score(None, error=...)` for this input."""
    try:
        return scorer(input)
    except Exception as exc:  # a scorer never raises for one bad input; the error is the result
        return Score(None, error=f"{type(exc).__name__}: {exc}")


def _raw_attribute(raw: Any) -> float | str | None:
    """A number as it is, several named raw values (e.g. SongEval dimensions) as JSON."""
    if isinstance(raw, Mapping):
        return json.dumps(raw)
    return raw if isinstance(raw, int | float) else None


def span_attributes(scorer: FunctionScorer, score: Score) -> dict[str, Any]:
    """The `hone.taste.*` attributes of one score; `None` values are left out."""
    raw, reference = score.details.get("raw"), score.details.get("reference_set")
    attributes = {
        "hone.taste.scorer": scorer.name,
        "hone.taste.family": scorer.family,
        "hone.taste.model_id": scorer.model_id or None,
        "hone.taste.license": scorer.license,
        "hone.taste.value": score.value,
        "hone.taste.confidence": score.confidence,
        "hone.taste.raw": _raw_attribute(raw),
        "hone.taste.reference_set": reference if isinstance(reference, str) else None,
        "hone.taste.reason": score.reason or None,
        "hone.taste.error": score.error or None,
        "hone.taste.details": json.dumps(score.details, ensure_ascii=False, default=str),
    }
    return {key: value for key, value in attributes.items() if value is not None}


def recorded(
    scorer: FunctionScorer,
    score: Callable[..., Score],
    *inputs: Any,
    trace: Mapping[str, str] | None = None,
    attributes: Mapping[str, str] | None = None,
) -> Score:
    """`score(*inputs)` inside a `hone.taste.score` span, after checking every input's kind."""
    with start_span("hone.taste.score", trace=trace) as span:
        mismatch = next((e for e in (input_kind_error(scorer.accepts, i) for i in inputs) if e), "")
        result = Score(None, error=mismatch) if mismatch else score_safely(lambda _: score(*inputs), None)
        span["attributes"].update({**span_attributes(scorer, result), **(attributes or {})})
        if result.error:
            span["status"] = {"code": "error", "message": result.error}
    return result


@dataclass(slots=True)
class FunctionScorer:
    """A scorer built from a plain function `score(input) -> Score` plus metadata.

    >>> s = FunctionScorer("len", "human_likeness", frozenset({"text"}), lambda t: Score(len(t) / 10))
    >>> s("hello").value
    0.5
    >>> s(42).error
    'accepts text; got int'
    """

    name: str
    family: str
    accepts: frozenset[str]
    score: Callable[[Any], Score]
    license: str = "Apache-2.0"
    source_data: str = ""
    on_close: Callable[[], None] | None = None
    model_id: str = ""

    def __post_init__(self) -> None:
        unknown = self.accepts - INPUT_KINDS.keys()
        if unknown or not self.accepts:
            raise ConfigError(f"unknown input kinds {sorted(unknown)}; choose from {sorted(INPUT_KINDS)}")

    def __call__(self, input: Any, /, *, trace: Mapping[str, str] | None = None) -> Score:
        """Score one input and record a `hone.taste.score` span (design/current.md §7)."""
        return recorded(self, self.score, input, trace=trace)

    def close(self) -> None:
        """Release heavy resources (loaded models); the next call loads them again."""
        if self.on_close is not None:
            self.on_close()
