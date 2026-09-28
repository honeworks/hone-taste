"""`for_select`: use any hone-taste scorer as a candidate scorer (e.g. in hone-select), importing nothing."""

from __future__ import annotations

import contextlib
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, cast

from hone_taste._records import NullSink, recording
from hone_taste._tracing import RESOURCE, current_trace, use_trace
from hone_taste.errors import ConfigError
from hone_taste.ports import get_field
from hone_taste.types import Score, Scorer, score_safely

FILE_KINDS = {"audio", "image", "image_with_prompt"}
PACKAGE_VERSION = RESOURCE["hone.package.version"]


@dataclass(frozen=True, slots=True)
class SelectScorer:
    """Calls `scorer` on the input read from a hone-select candidate (`id`, `data`, `files`, `meta`)."""

    scorer: Scorer
    field: str | None = None
    file: str | None = None
    prompt_field: str | None = None
    record: bool = True

    def __post_init__(self) -> None:
        if self.field is not None and self.file is not None:
            raise ConfigError("for_select reads either a data field or a file, not both")
        if self.file is not None and not FILE_KINDS & set(getattr(self.scorer, "accepts", FILE_KINDS)):
            raise ConfigError(
                f"{self.scorer.name} does not take files (accepts {sorted(self.scorer.accepts)}); "
                "use field= to read candidate.data instead"
            )

    @property
    def name(self) -> str:
        return self.scorer.name

    @property
    def version(self) -> str:
        """hone-taste's version (hone-select records it as the scorer version)."""
        return PACKAGE_VERSION

    def read_input(self, candidate: Any) -> Any:
        """`files[file]`, `data[field]` or `data`; with `prompt_field`, a prompt + response / image dict."""
        data = get_field(candidate, "data")
        if self.file is not None:
            value = get_field(candidate, "files")[self.file]
        else:
            value = data if self.field is None else data[self.field]
        if self.prompt_field is None:
            return value
        return {"prompt": data[self.prompt_field], ("image" if self.file else "response"): value}

    def __call__(self, candidate: Any, /, *, trace: Mapping[str, str] | None = None) -> Score:
        """Score one candidate; `trace` (a trace context) links the span to the caller's trace."""
        try:
            value = self.read_input(candidate)
        except (KeyError, IndexError, TypeError) as exc:
            return Score(None, error=f"could not read the input from the candidate: {exc!r}")
        trace = {**(current_trace() if trace is None else trace), "hone.scorer": self.name}
        candidate_id = get_field(candidate, "id")
        if candidate_id is not None:
            trace["hone.candidate_id"] = str(candidate_id)
        sink = contextlib.nullcontext() if self.record else recording(NullSink())
        with use_trace(trace), sink:
            return score_safely(self.scorer, value)


def for_select(
    scorer: Scorer,
    *,
    field: str | None = None,
    file: str | None = None,
    prompt_field: str | None = None,
    record: bool = True,
) -> SelectScorer:
    """A callable `candidate -> Score` for hone-select's `registry`.

    Reads `candidate.data[field]` (or all of `candidate.data`), or `candidate.files[file]` for audio /
    image scorers; `prompt_field` names the prompt in `candidate.data` for `text_with_prompt` /
    `image_with_prompt` scorers. `record=False` turns off hone-taste's own spans (hone-select records
    the score itself).

    >>> import hone_taste as tt
    >>> class Candidate:
    ...     id, data, files, meta = "c1", {"lyrics": "Let us delve into it."}, {}, {}
    >>> tt.for_select(tt.patterns(["delve"]), field="lyrics", record=False)(Candidate()).value
    0.8
    """
    return SelectScorer(scorer, field, file, prompt_field, record)


@dataclass(frozen=True, slots=True)
class SelectPairwise:
    """Calls `panel.pairwise` on the inputs read from two hone-select candidates; `kind = "pairwise"`."""

    reader: SelectScorer

    kind = "pairwise"  # hone-select registers a callable with this attribute as a pairwise judge

    @property
    def name(self) -> str:
        return f"{self.reader.name}_pairwise"

    @property
    def version(self) -> str:
        return PACKAGE_VERSION

    def __call__(
        self, a: Any, b: Any, /, *, trace: Mapping[str, str] | None = None
    ) -> tuple[str, float | None]:
        """`("a" | "b" | "tie", confidence)`; a comparison that failed raises, which hone-select records."""
        first, second = self.reader.read_input(a), self.reader.read_input(b)
        trace = {**(current_trace() if trace is None else trace), "hone.scorer": self.name}
        sink = contextlib.nullcontext() if self.reader.record else recording(NullSink())
        with use_trace(trace), sink:
            compare = cast(Callable[[Any, Any], Score], getattr(self.reader.scorer, "pairwise"))  # noqa: B009
            result = compare(first, second)
        if result.value is None:
            raise RuntimeError(result.error)
        return result.details["choice"], result.confidence


def for_select_pairwise(
    panel: Scorer, *, field: str | None = None, file: str | None = None, record: bool = True
) -> SelectPairwise:
    """A hone-select pairwise judge `(a, b) -> ("a" | "b" | "tie", confidence)` from an audience panel.

    hone-select asks it in both orders for near ties, which also cancels a remaining position bias.

    >>> import hone_taste as tt
    >>> class Candidate:
    ...     def __init__(self, text):
    ...         self.id, self.data, self.files, self.meta = text, {"text": text}, {}, {}
    >>> panel = tt.audience(["a fan"], "Which is catchier?", tt.testing.FakeDecisionClient())
    >>> judge = tt.for_select_pairwise(panel, field="text", record=False)
    >>> judge.kind, judge(Candidate("la la"), Candidate("na na"))
    ('pairwise', ('a', 1.0))
    """
    if not hasattr(panel, "pairwise"):
        raise ConfigError(f"{panel.name} cannot compare two inputs; pass a panel from tt.audience(...)")
    return SelectPairwise(SelectScorer(panel, field, file, None, record))
