"""Combine several scorers into one, and the aggregate functions shared with the audience panel."""

from __future__ import annotations

import statistics
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from hone_taste.errors import ConfigError
from hone_taste.types import FunctionScorer, Score, Scorer, score_safely

# Each aggregate takes (value, weight) pairs with at least one entry and no None values.
Weighted = Sequence[tuple[float, float]]


def weighted_mean(pairs: Weighted) -> float:
    return sum(v * w for v, w in pairs) / sum(w for _, w in pairs)


AGGREGATES: dict[str, Callable[[Weighted], float]] = {
    "weighted_mean": weighted_mean,
    "mean": weighted_mean,  # the audience panel's name; personas all weigh 1
    "min": lambda pairs: min(v for v, _ in pairs),
    "median": lambda pairs: statistics.median(v for v, _ in pairs),
}


def aggregate_function(name: str) -> Callable[[Sequence[tuple[float | None, float]]], float | None]:
    """The named aggregate, applied to (value, weight) pairs after dropping `None` values and zero weights;
    it returns `None` when nothing is left.

    >>> aggregate_function("weighted_mean")([(1.0, 1.0), (None, 5.0), (0.0, 3.0)])
    0.25
    """
    try:
        method = AGGREGATES[name]
    except KeyError:
        raise ConfigError(f"unknown aggregate {name!r}; choose one of {sorted(AGGREGATES)}") from None

    def apply(pairs: Sequence[tuple[float | None, float]]) -> float | None:
        known = [(v, w) for v, w in pairs if v is not None and w > 0]
        return method(known) if known else None

    return apply


def _check_weights(weights: Mapping[str, float], scorers: Mapping[str, Scorer]) -> None:
    missing = sorted(set(weights) - set(scorers))
    if missing:
        raise ConfigError(
            f"weights name scorers that were not passed: {missing}; available: {sorted(scorers)}"
        )
    if any(w < 0 for w in weights.values()) or sum(weights.values()) <= 0:
        raise ConfigError(f"weights must be >= 0 with a positive total; got {dict(weights)}")


def combine(
    weights: Mapping[str, float], scorers: Mapping[str, Scorer], *, aggregate: str = "weighted_mean"
) -> FunctionScorer:
    """One scorer from several: each scorer runs on the input; missing values drop out and the
    remaining weights are renormalized. `value` is `None` only when no scorer produced a value.

    >>> from hone_taste.types import FunctionScorer, Score
    >>> a = FunctionScorer("a", "human_likeness", frozenset({"text"}), lambda t: Score(1.0))
    >>> b = FunctionScorer("b", "human_likeness", frozenset({"text"}), lambda t: Score(None, error="x"))
    >>> combine({"a": 1, "b": 3}, {"a": a, "b": b})("hi").value
    1.0
    """
    _check_weights(weights, scorers)
    weights = dict(weights)
    apply = aggregate_function(aggregate)
    used = {name: scorers[name] for name in weights}

    def score(input: Any) -> Score:
        parts = {name: score_safely(scorer, input) for name, scorer in used.items()}
        value = apply([(parts[n].value, weights[n]) for n in parts])
        details = {
            "parts": {
                n: {"value": s.value, "weight": weights[n], "error": s.error} for n, s in parts.items()
            },
            "aggregate": aggregate,
        }
        if value is None:
            errors = "; ".join(f"{n}: {s.error or 'no value'}" for n, s in parts.items())
            return Score(None, details=details, error=f"no scorer produced a value ({errors})")
        reason = ", ".join(f"{n}={s.value:.2f}" for n, s in parts.items() if s.value is not None)
        return Score(value, reason=f"{aggregate} of {reason}", details=details)

    accepts = frozenset[str]().union(*(s.accepts for s in used.values()))
    licenses = ", ".join(sorted({s.license for s in used.values()}))
    sources = "; ".join(f"{n}: {s.source_data}" for n, s in used.items() if s.source_data)
    return FunctionScorer("combined", "combined", accepts, score, license=licenses, source_data=sources)
