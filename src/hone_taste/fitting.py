"""Fit per-person weights over scorers from pairwise picks (Bradley-Terry / logistic regression).

Each pick (winner, loser) gives one example: the per-scorer score differences `s_k(winner) - s_k(loser)`,
each divided by that scorer's spread over the scored items so the L2 penalty treats scorers alike. The
model is `P(winner preferred) = sigmoid(sum_k w_k * x_k)` without intercept, fitted by gradient descent in
pure Python (a handful of scorers and picks; no NumPy needed). Negative weights are clipped to 0 and the
rest normalized to sum 1, so the result can drive a weighted mean.
"""

from __future__ import annotations

import json
import math
import statistics
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from hone_taste.errors import ConfigError
from hone_taste.types import Scorer, score_safely

L2 = 1.0
STEPS = 2000
LEARNING_RATE = 0.5

Table = dict[str, dict[str, float | None]]  # item key -> scorer name -> value


def item_key(item: Any) -> str:
    """A stable identity for an item (text, path, JSON-able dict) used to cache scores and compare picks."""
    return json.dumps(item, sort_keys=True, default=str)


def score_table(scorers: Mapping[str, Scorer], items: Iterable[Any]) -> Table:
    """Score every distinct item once with every scorer."""
    table: Table = {}
    for item in items:
        key = item_key(item)
        if key not in table:
            table[key] = {name: score_safely(scorer, item).value for name, scorer in scorers.items()}
    return table


@dataclass(frozen=True, slots=True)
class Weights:
    """Fitted weights (>= 0, summing to 1) plus how well they explain the picks."""

    weights: dict[str, float]
    accuracy: float  # share of picks the fitted model gets right (ties count half)
    log_loss: float
    picks: int
    coefficients: dict[str, float] = field(default_factory=dict[str, float])  # before clipping, raw scale
    note: str = ""


def _spread(table: Table, name: str) -> float:
    values = [v for row in table.values() if (v := row[name]) is not None]
    spread = statistics.pstdev(values) if len(values) > 1 else 0.0
    return spread if spread > 0 else 1.0


def _sigmoid(z: float) -> float:
    return 1 / (1 + math.exp(-z)) if z >= 0 else math.exp(z) / (1 + math.exp(z))


def _gradient_descent(rows: list[list[float]]) -> list[float]:
    w = [0.0] * len(rows[0])
    for _ in range(STEPS):
        grad = [L2 * wk for wk in w]
        for x in rows:
            p_wrong = _sigmoid(-sum(wk * xk for wk, xk in zip(w, x, strict=True)))
            grad = [g - xk * p_wrong for g, xk in zip(grad, x, strict=True)]
        w = [wk - LEARNING_RATE * g / len(rows) for wk, g in zip(w, grad, strict=True)]
    return w


def _quality(rows: list[list[float]], w: list[float]) -> tuple[float, float]:
    margins = [sum(wk * xk for wk, xk in zip(w, x, strict=True)) for x in rows]
    accuracy = sum(1.0 if m > 0 else 0.5 if m == 0 else 0.0 for m in margins) / len(rows)
    log_loss = sum(-math.log(max(_sigmoid(m), 1e-12)) for m in margins) / len(rows)
    return accuracy, log_loss


def fit_weights(table: Table, names: Sequence[str], pairs: Sequence[tuple[Any, Any]]) -> Weights:
    """Fit weights over the scorers `names` so that they prefer each pair's winner.

    `table` holds the scores of every picked item (see `score_table`); extra rows only widen the
    measured spread of each scorer.

    >>> table = {item_key(t): {"len": len(t) / 10} for t in ["long text", "short", "longer text"]}
    >>> fit_weights(table, ["len"], [("long text", "short"), ("longer text", "short")]).weights
    {'len': 1.0}
    """
    if not names or not pairs:
        raise ConfigError("fit needs at least one scorer and one pick")
    spread = {name: _spread(table, name) for name in names}

    def difference(winner: Any, loser: Any, name: str) -> float:
        a, b = table[item_key(winner)][name], table[item_key(loser)][name]
        return 0.0 if a is None or b is None else (a - b) / spread[name]  # unknown -> no evidence

    rows = [[difference(w, lo, name) for name in names] for w, lo in pairs]
    w = _gradient_descent(rows)
    accuracy, log_loss = _quality(rows, w)
    coefficients = {name: wk / spread[name] for name, wk in zip(names, w, strict=True)}
    positive = {name: max(0.0, c) for name, c in coefficients.items()}
    total = sum(positive.values())
    if total == 0:
        equal = {name: 1 / len(names) for name in names}
        return Weights(equal, accuracy, log_loss, len(pairs), coefficients, "no scorer predicts the picks")
    weights = {name: c / total for name, c in positive.items()}
    return Weights(weights, accuracy, log_loss, len(pairs), coefficients)
