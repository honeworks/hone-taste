"""AC-15: agreement report on synthetic picks: rates, correlation and suggested weights."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

import hone_taste as tt


@dataclass
class Lookup:
    name: str
    values: dict[str, float | None]
    family: str = "human_likeness"
    accepts: frozenset[str] = frozenset({"text"})
    license: str = "Apache-2.0"
    source_data: str = "synthetic"
    calls: int = 0

    def __call__(self, input: Any, /) -> tt.Score:
        self.calls += 1
        return tt.Score(self.values[input])


def test_ac15_agreement(tmp_path: Path) -> None:
    human = [("a", "b"), ("c", "d"), ("e", "f"), ("g", "h")]
    good = Lookup("good", {"a": 0.9, "b": 0.1, "c": 0.8, "d": 0.2, "e": 0.7, "f": 0.3, "g": 0.4, "h": 0.6})
    bad = Lookup("bad", {"a": 0.1, "b": 0.9, "c": 0.2, "d": 0.8, "e": 0.5, "f": 0.5, "g": 0.6, "h": 0.4})
    partial = Lookup(
        "partial", {"a": 0.9, "b": None, "c": 0.8, "d": 0.2, "e": 0.3, "f": 0.7, "g": None, "h": 0.1}
    )

    report = tt.agreement({"good": good, "bad": bad, "partial": partial}, human)

    assert report.pairs == 4
    assert report.scorers["good"].rate == pytest.approx(0.75)
    assert report.scorers["good"].correlation == pytest.approx(0.5)
    assert report.scorers["bad"].rate == pytest.approx(0.375)  # 1 agree, 1 tie, 2 disagree
    assert report.scorers["bad"].correlation == pytest.approx(-0.25)
    assert report.scorers["partial"].compared == 2
    assert report.scorers["partial"].unscored == 2
    assert report.scorers["partial"].rate == pytest.approx(0.5)
    weights = report.suggested.weights
    assert weights["good"] > weights["bad"]
    assert weights["good"] > weights["partial"]
    assert sum(weights.values()) == pytest.approx(1.0)
    assert weights["bad"] == 0.0
    assert report.suggested.coefficients["bad"] < 0
    assert report.suggested.accuracy == pytest.approx(0.75)
    assert good.calls == 8  # every item scored once, shared by the rates and the fit
    same = tt.profile("check", path=tmp_path / "check.json")
    for winner, loser in human:
        same.add_pick(winner=winner, loser=loser)
    assert same.fit({"good": good, "bad": bad, "partial": partial}) == report.suggested  # the same fit

    data = json.loads(report.to_json())
    assert data["scorers"]["good"]["rate"] == pytest.approx(0.75)
    assert data["suggested_weights"]["weights"]["good"] == pytest.approx(weights["good"])
    printed = str(report)
    rows = {line.split()[0]: line.split()[1:] for line in printed.splitlines()[2:-1]}
    assert set(rows) == {"good", "bad", "partial"}
    assert rows["good"] == ["0.75", "0.50", "4", f"{weights['good']:.2f}"]
    assert len({len(line) for line in printed.splitlines()[1:-1]}) == 1  # columns line up
    assert "4 human picks" in printed
