"""Audience panel: an LLM plays several target personas; each rates the input on a scale with a reason.

Caveat (also in every result's details): the personas are one model pretending. They share its taste and
blind spots and tend to be too positive, so weigh the panel lower than taste models until
`tt.agreement(...)` shows it matches real picks. Rated alone, candidates often all get the top of the
scale (`details["ceiling"]`); `panel.pairwise(a, b)` asks each persona to compare two instead
(design/changes/0002).
"""

from __future__ import annotations

import mimetypes
import os
import re
import statistics
import tomllib
import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hone_taste._tracing import current_trace
from hone_taste.combine import aggregate_function
from hone_taste.errors import ConfigError
from hone_taste.ports import DecisionClient, TextClient, get_field
from hone_taste.text_decisions import TextDecisionClient
from hone_taste.types import FunctionScorer, Score, recorded

CAVEAT = "one LLM playing personas; weigh lower than taste models until checked against real picks"
IMAGE_STATE = "(the item to judge is the attached image)"
PAIR_IMAGE_STATE = "(the two items to compare are the attached images: the first is A, the second is B)"
OPTIONS = ("A", "B", "no preference")
PREFERS_FIRST = {"A": 1.0, "B": 0.0, "no preference": 0.5}  # how much the answer favours what came first


def load_personas(personas: Sequence[str] | str | Path) -> list[str]:
    """A list as given, or `personas = [...]` from a TOML file."""
    if isinstance(personas, str | Path):
        try:
            names: Sequence[Any] = tomllib.loads(Path(personas).read_text(encoding="utf-8")).get(
                "personas", []
            )
        except (OSError, tomllib.TOMLDecodeError) as exc:
            raise ConfigError(f"cannot read personas from {personas}: {exc}") from exc
    else:
        names = personas
    found = [str(p).strip() for p in names if str(p).strip()]
    if not found:
        raise ConfigError(
            "audience() needs at least one persona (a list, or `personas = [...]` in a TOML file)"
        )
    return found


def as_decision_client(client: DecisionClient | TextClient) -> DecisionClient:
    """Use a DecisionClient as is; wrap a TextClient (anything with `complete`) in a TextDecisionClient."""
    if hasattr(client, "decide"):
        return client  # type: ignore[return-value]
    if hasattr(client, "complete"):
        return TextDecisionClient(client)  # type: ignore[arg-type]
    raise ConfigError("client must have decide() (DecisionClient) or complete() (TextClient)")


def _family(model: str) -> str:
    match = re.match(r"[a-z]+", model.lower().rsplit("/", 1)[-1])
    return match.group(0) if match else model.lower()


def _same_family_warning(client: Any, generator_model: str | None) -> str:
    judge_model = get_field(client, "model") or get_field(client, "model_id")
    if not generator_model or not isinstance(judge_model, str):
        return ""
    if _family(judge_model) != _family(generator_model):
        return ""
    message = f"panel model {judge_model!r} is from the same family as the generator {generator_model!r}"
    warnings.warn(message, UserWarning, stacklevel=3)
    return message


def _is_image(input: Any) -> bool:
    """A path object, or a string naming an existing image file; any other string is text to judge."""
    if isinstance(input, os.PathLike):
        return True
    guessed = mimetypes.guess_type(input)[0] or ""
    return guessed.startswith("image/") and Path(input).is_file()


def _ask(client: DecisionClient, question: Mapping[str, Any], input: Any) -> dict[str, Any]:
    """One persona's rating: `{"value", "raw", "rationale", "error"}`; errors never raise."""
    state, images = (IMAGE_STATE, (str(input),)) if _is_image(input) else (input, ())
    try:
        answers = client.decide(state, {"rating": question}, images=images, trace=current_trace())
        answer = answers.get("rating")
    except Exception as exc:  # one persona failing must not sink the panel
        return {"value": None, "raw": None, "rationale": "", "error": f"{type(exc).__name__}: {exc}"}
    if answer is None:
        return {"value": None, "raw": None, "rationale": "", "error": "not answered"}
    value: float | None = get_field(answer, "value")
    error = get_field(answer, "error") or ("" if value is not None else "no value")
    return {
        "value": None if error else value,
        "raw": get_field(answer, "raw"),
        "rationale": get_field(answer, "rationale") or "",
        "error": error,
    }


def audience(
    personas: Sequence[str] | str | Path,
    question: str,
    client: DecisionClient | TextClient,
    *,
    scale: tuple[int, int] = (1, 5),
    aggregate: str = "mean",
    anchors: Mapping[str, str] | None = None,
    generator_model: str | None = None,
) -> AudiencePanel:
    """A scorer asking each persona `question` (one `decide()` call per persona) and aggregating.

    `client` is any DecisionClient, or a TextClient (wrapped in `TextDecisionClient`). Text inputs are the
    state; an image path is attached as an image. For audio, pass a text description (lyrics, SongEval
    scores). `generator_model` enables a warning when the panel's model family matches the generator's.
    `details["ceiling"]` is true when every persona gave the top rating; `panel.pairwise(a, b)` compares
    two inputs instead of rating one.

    >>> from hone_taste.testing import FakeDecisionClient
    >>> panel = audience(["a blues fan", "a producer"], "Would you keep listening?", FakeDecisionClient())
    >>> result = panel("some lyric")
    >>> result.value, result.details["disagreement"], len(result.details["personas"])
    (0.5, 0.0, 2)
    >>> result.details["ceiling"]
    False
    """
    people = load_personas(personas)
    low, high = scale
    if high <= low:
        raise ConfigError(f"scale must be (low, high) with high > low; got {scale}")
    apply = aggregate_function(aggregate)
    decision_client = as_decision_client(client)
    warning = _same_family_warning(client, generator_model)
    anchors = dict(
        anchors or {str(low): "not at all", str((low + high) // 2): "somewhat", str(high): "absolutely"}
    )

    def score(input: Any) -> Score:
        results: list[dict[str, Any]] = []
        for persona in people:
            spec = {
                "type": "score",
                "instructions": f"You are {persona}.\n{question}",
                "scale": [low, high],
                "anchors": anchors,
            }
            results.append({"persona": persona, **_ask(decision_client, spec, input)})
        known = [r["value"] for r in results if r["value"] is not None]
        details: dict[str, Any] = {
            "personas": results,
            "disagreement": statistics.pstdev(known) if len(known) > 1 else None,
            "ceiling": bool(known) and min(known) >= 1.0,  # every persona gave the top: no preference shown
            "aggregate": aggregate,
            "question": question,
            "caveat": CAVEAT,
        }
        if warning:
            details["warning"] = warning
        value = apply([(v, 1.0) for v in known])
        if value is None:
            errors = "; ".join(f"{r['persona'][:40]}: {r['error']}" for r in results)
            return Score(None, details=details, error=f"every persona failed ({errors})")
        return Score(value, reason=f"{aggregate} of {len(known)}/{len(people)} personas", details=details)

    def compare(a: Any, b: Any) -> Score:
        choice_question = f"{question}\nYou are shown two items, A and B. Which do you prefer?"
        results = [
            {
                "persona": persona,
                **_prefer(decision_client, f"You are {persona}.\n{choice_question}", a, b, i),
            }
            for i, persona in enumerate(people)
        ]
        return _comparison(results, question, warning)

    return AudiencePanel(
        "audience",
        "audience",
        frozenset({"text", "image"}),
        score,
        license="depends on the client's model",
        source_data="LLM-simulated personas (no human ratings)",
        compare=compare,
    )


def _pair_state(first: Any, second: Any) -> tuple[str, tuple[str, ...]]:
    if _is_image(first) and _is_image(second):
        return PAIR_IMAGE_STATE, (str(first), str(second))
    return f"A:\n{first}\n\nB:\n{second}", ()


def _prefer(client: DecisionClient, instructions: str, a: Any, b: Any, index: int) -> dict[str, Any]:
    """One persona's preference between `a` and `b`; odd personas see `b` first (against position bias).

    `prefers_a` is 1 (a), 0 (b) or 0.5 (no preference); errors never raise."""
    swapped = index % 2 == 1
    state, images = _pair_state(b, a) if swapped else _pair_state(a, b)
    spec = {"type": "choice", "instructions": instructions, "options": list(OPTIONS)}
    result: dict[str, Any] = {"shown_first": "b" if swapped else "a", "prefers_a": None, "rationale": ""}
    try:
        answer = client.decide(state, {"preference": spec}, images=images, trace=current_trace()).get(
            "preference"
        )
    except Exception as exc:  # one persona failing must not sink the comparison
        return {**result, "error": f"{type(exc).__name__}: {exc}"}
    choice = get_field(answer, "choice") if answer is not None else None
    error = (get_field(answer, "error") if answer is not None else "not answered") or (
        "" if choice in OPTIONS else f"invalid choice {choice!r}"
    )
    if error:
        return {**result, "error": error}
    first = PREFERS_FIRST[str(choice)]
    rationale = get_field(answer, "rationale") or ""
    return {**result, "prefers_a": 1.0 - first if swapped else first, "rationale": rationale, "error": ""}


def _comparison(results: list[dict[str, Any]], question: str, warning: str) -> Score:
    known = [r["prefers_a"] for r in results if r["prefers_a"] is not None]
    details: dict[str, Any] = {"personas": results, "mode": "compare", "question": question, "caveat": CAVEAT}
    if warning:
        details["warning"] = warning
    if not known:
        errors = "; ".join(f"{r['persona'][:40]}: {r['error']}" for r in results)
        return Score(None, details={**details, "choice": None}, error=f"every persona failed ({errors})")
    value = statistics.fmean(known)
    choice = "a" if value > 0.5 else "b" if value < 0.5 else "tie"
    reason = f"{choice}: {len(known)}/{len(results)} personas compared (1 = all prefer a)"
    return Score(value, confidence=abs(value - 0.5) * 2, reason=reason, details={**details, "choice": choice})


@dataclass(slots=True)
class AudiencePanel(FunctionScorer):
    """The scorer `audience()` returns: rates one input when called; `pairwise(a, b)` compares two."""

    compare: Callable[[Any, Any], Score] | None = None

    def pairwise(self, a: Any, b: Any, /, *, trace: Mapping[str, str] | None = None) -> Score:
        """Each persona sees both inputs (labelled A and B; every second persona in swapped order) and
        says which it prefers. `value` = share preferring `a` (0.5 = tie), `details["choice"]` is
        "a", "b" or "tie", `confidence` = |value - 0.5| * 2. Recorded as a `hone.taste.score` span with
        `hone.taste.panel.mode = "compare"`.

        >>> from hone_taste.testing import FakeDecisionClient
        >>> panel = audience(["a fan", "a critic"], "Which chorus is catchier?", FakeDecisionClient())
        >>> result = panel.pairwise("la la la", "na na na")  # the fake always answers "A": a position bias
        >>> result.value, result.details["choice"]
        (0.5, 'tie')
        """
        compare = self.compare
        if compare is None:
            raise ConfigError("this panel was built without a comparison; build it with tt.audience(...)")
        return recorded(self, compare, a, b, trace=trace, attributes={"hone.taste.panel.mode": "compare"})
