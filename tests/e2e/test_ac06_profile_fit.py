"""AC-6: picks where scorer X predicts the winner give X the larger weight; weights persist and reload."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

import hone_taste as tt


@dataclass
class Lookup:
    """A user-written scorer that reads its values from a table."""

    name: str
    values: dict[str, float]
    family: str = "human_likeness"
    accepts: frozenset[str] = frozenset({"text"})
    license: str = "Apache-2.0"
    source_data: str = "synthetic"

    def __call__(self, input: Any, /) -> tt.Score:
        return tt.Score(self.values[input])


ITEMS = [f"lyric {i}" for i in range(8)]
X = {item: i / 7 for i, item in enumerate(ITEMS)}  # the person's taste: higher x wins
Y = {item: [0.2, 0.9, 0.4, 0.1, 0.8, 0.3, 0.7, 0.5][i] for i, item in enumerate(ITEMS)}  # unrelated


def again_fit(pairs: list[tuple[Any, Any]], scorers: dict[str, Lookup], tmp_path: Path) -> tt.Weights:
    fresh = tt.profile("fresh", path=tmp_path / "fresh.json")
    for winner, loser in pairs:
        fresh.add_pick(winner=winner, loser=loser)
    return fresh.fit(scorers, items=ITEMS)


def test_ac6_profile_fit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HONE_HOME", str(tmp_path / "home"))
    scorers = {"x": Lookup("x", X), "y": Lookup("y", Y)}
    me = tt.profile("ana")
    for a, b in [(0, 3), (5, 1), (7, 2), (4, 6), (2, 1), (6, 0), (3, 5), (1, 7)]:
        winner, loser = (ITEMS[a], ITEMS[b]) if X[ITEMS[a]] > X[ITEMS[b]] else (ITEMS[b], ITEMS[a])
        me.add_pick(winner=winner, loser=loser)

    weights = me.fit(scorers, items=ITEMS)
    assert weights.weights["x"] > weights.weights["y"]
    assert weights.weights["x"] > 0.8
    assert sum(weights.weights.values()) == pytest.approx(1.0)
    assert weights.accuracy == 1.0
    assert weights.picks == 8

    assert again_fit(me.pairs, scorers, tmp_path) == weights  # deterministic

    path = tmp_path / "home" / "taste" / "profiles" / "ana.json"
    assert path.exists()
    again = tt.profile("ana")
    assert again.fitted == weights
    assert len(again.picks) == 8

    mine = again.as_scorer(scorers)
    assert mine.family == "personal"
    w = weights.weights
    assert mine(ITEMS[7]).value == pytest.approx(w["x"] * X[ITEMS[7]] + w["y"] * Y[ITEMS[7]])
    assert mine(ITEMS[7]).value > mine(ITEMS[0]).value  # type: ignore[operator]
