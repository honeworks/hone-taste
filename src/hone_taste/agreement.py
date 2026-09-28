"""Check scorers against real human picks: how often each one agrees, and which weights fit the picks."""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Mapping, Sequence
from typing import Any

from hone_taste.errors import ConfigError
from hone_taste.fitting import Table, Weights, fit_weights, item_key, score_table
from hone_taste.types import Scorer


@dataclasses.dataclass(frozen=True, slots=True)
class ScorerAgreement:
    """How one scorer's preferences match the human picks it could score."""

    rate: float | None  # share of pairs where the scorer prefers the human's pick (ties count half)
    correlation: float | None  # Kendall-style: (agree - disagree) / compared, in [-1, 1]
    compared: int  # pairs where both items got a value
    unscored: int  # pairs skipped because a value was missing


@dataclasses.dataclass(frozen=True, slots=True)
class AgreementReport:
    """Per-scorer agreement with human picks plus weights fitted to those picks. `str()` prints a table."""

    scorers: dict[str, ScorerAgreement]
    suggested: Weights
    pairs: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "pairs": self.pairs,
            "scorers": {name: dataclasses.asdict(a) for name, a in self.scorers.items()},
            "suggested_weights": dataclasses.asdict(self.suggested),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    def __str__(self) -> str:
        def fmt(x: float | None) -> str:
            return "-" if x is None else f"{x:.2f}"

        width = max(len(name) for name in ["scorer", *self.scorers])
        header = f"{'scorer':<{width}}  agree   corr  compared  weight"
        lines = [f"Agreement with {self.pairs} human picks", header]
        for name, a in self.scorers.items():
            weight = self.suggested.weights[name]
            cells = f"{fmt(a.rate):>5}  {fmt(a.correlation):>5}  {a.compared:>8}  {weight:>6.2f}"
            lines.append(f"{name:<{width}}  {cells}")
        lines.append(f"fitted weights explain {self.suggested.accuracy:.0%} of the picks")
        return "\n".join(lines)


def _agreement(table: Table, name: str, pairs: Sequence[tuple[Any, Any]]) -> ScorerAgreement:
    agree = disagree = ties = unscored = 0
    for winner, loser in pairs:
        a, b = table[item_key(winner)][name], table[item_key(loser)][name]
        if a is None or b is None:
            unscored += 1
        elif a > b:
            agree += 1
        elif a < b:
            disagree += 1
        else:
            ties += 1
    compared = agree + disagree + ties
    if compared == 0:
        return ScorerAgreement(None, None, 0, unscored)
    return ScorerAgreement((agree + ties / 2) / compared, (agree - disagree) / compared, compared, unscored)


def agreement(scorers: Mapping[str, Scorer], human_pairs: Sequence[tuple[Any, Any]]) -> AgreementReport:
    """Compare each scorer with human picks given as (winner, loser) pairs; every item is scored once.

    >>> from hone_taste.types import FunctionScorer, Score
    >>> length = FunctionScorer("len", "human_likeness", frozenset({"text"}), lambda t: Score(len(t) / 20))
    >>> report = agreement({"len": length}, [("a longer one", "short"), ("tiny", "a bit longer")])
    >>> report.scorers["len"].rate, report.scorers["len"].correlation
    (0.5, 0.0)
    """
    if not scorers or not human_pairs:
        raise ConfigError("agreement needs at least one scorer and one (winner, loser) pair")
    table = score_table(scorers, [item for pair in human_pairs for item in pair])
    per_scorer = {name: _agreement(table, name, human_pairs) for name in scorers}
    return AgreementReport(per_scorer, fit_weights(table, list(scorers), human_pairs), len(human_pairs))


@dataclasses.dataclass(frozen=True, slots=True)
class Spread:
    """How far apart one scorer's values over a batch of candidates are (see `spread`)."""

    range: float | None  # max - min of the known values; None with fewer than two
    flat: bool  # every known value within `tolerance` of the others: the scorer told nothing apart
    at_ceiling: bool  # every known value within `tolerance` of 1.0
    known: int
    unscored: int


def spread(scores: Sequence[Any], *, tolerance: float = 0.05) -> Spread:
    """Did a scorer tell a batch of candidates apart? Takes `Score`s (or plain values / `None`).

    A panel that rates every candidate the same adds cost but no information: drop or down-weight it for
    this batch when `flat` is true (design/changes/0002).

    >>> from hone_taste.types import Score
    >>> s = spread([Score(1.0), Score(1.0), Score(0.98), Score(None, error="x")])
    >>> s.flat, s.at_ceiling, s.known, s.unscored
    (True, True, 3, 1)
    >>> spread([0.2, 0.9]).flat
    False
    """
    values = [getattr(s, "value", s) for s in scores]
    known = [float(v) for v in values if v is not None]
    width = max(known) - min(known) if len(known) > 1 else None
    return Spread(
        range=width,
        flat=width is not None and width <= tolerance,
        at_ceiling=bool(known) and min(known) >= 1.0 - tolerance,
        known=len(known),
        unscored=len(values) - len(known),
    )
