"""Ports hone-taste owns (version 1, design/current.md §6). Any object with these methods fits.

`DecisionClient` / `TextClient` back the audience panel; `GpuLease` wraps heavy model loading and
inference; `RecordSink` receives the `hone.taste.score` spans.
"""

from __future__ import annotations

from collections.abc import Generator, Mapping, Sequence
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass, field
from typing import Any, Protocol, cast

PORTS_VERSION = "1"

TraceContext = Mapping[str, str]
Question = Mapping[
    str, Any
]  # {"type": "yes_no"|"choice"|"score", "instructions": str, "scale", "anchors", ...}
Answer = Any  # a Mapping or an attribute object with type, value, choice, raw, rationale, error, ...


def get_field(obj: Any, key: str, default: Any = None) -> Any:
    """Read `key` from a Mapping or an attribute-style object (ports accept both)."""
    if isinstance(obj, Mapping):
        return cast(Mapping[str, Any], obj).get(key, default)
    return getattr(obj, key, default)


@dataclass(frozen=True, slots=True)
class TextResult:
    """What `TextClient.complete` returns."""

    text: str
    parsed: Any = None
    error: str | None = None
    model: str = ""
    finish_reason: str | None = None
    usage: Mapping[str, int] = field(default_factory=dict[str, int])
    span_id: str | None = None


class TextClient(Protocol):
    """Prompt in, text or a schema-validated object out.

    Transport errors raise; model-quality problems (e.g. invalid JSON) set `error` on the result.
    """

    def complete(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        schema: Mapping[str, Any] | None = None,
        trace: TraceContext | None = None,
        **params: Any,
    ) -> Any: ...


class DecisionClient(Protocol):
    """Structured questions about a state; returns one answer per question name."""

    def decide(
        self,
        state: str | Mapping[str, Any],
        questions: Mapping[str, Question],
        *,
        images: Sequence[str] = (),
        trace: TraceContext | None = None,
    ) -> Mapping[str, Answer]: ...


class GpuLease(Protocol):
    """Reserve GPU memory around model loading and inference."""

    def lease(
        self, name: str, vram_gb: float, *, timeout_s: float | None = None, trace: TraceContext | None = None
    ) -> AbstractContextManager[None]: ...


class NullGpuLease:
    """The default `GpuLease`: reserves nothing (fine when one process owns the GPU).

    >>> with NullGpuLease().lease("songeval", 3.0):
    ...     pass
    """

    @contextmanager
    def lease(
        self, name: str, vram_gb: float, *, timeout_s: float | None = None, trace: TraceContext | None = None
    ) -> Generator[None]:
        yield


class RecordSink(Protocol):
    """Receives spans in the shared span format (design/current.md §7)."""

    def emit(self, span: Mapping[str, Any]) -> None: ...
    def flush(self) -> None: ...
    def close(self) -> None: ...
