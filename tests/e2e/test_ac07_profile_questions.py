"""AC-7: profile.questions returns the closest-margin pairs first, within budget, without repeats."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import hone_taste as tt


@dataclass
class Lookup:
    name: str
    values: dict[str, float]
    family: str = "human_likeness"
    accepts: frozenset[str] = frozenset({"text"})
    license: str = "Apache-2.0"
    source_data: str = "synthetic"

    def __call__(self, input: Any, /) -> tt.Score:
        return tt.Score(self.values[input])


def test_ac7_profile_questions(tmp_path: Path) -> None:
    values = {"a": 0.1, "b": 0.5, "c": 0.52, "d": 0.9}
    scorers = {"s": Lookup("s", values)}
    me = tt.profile("q", path=tmp_path / "q.json")

    first = me.questions(["a", "b", "c", "d", "b"], scorers, budget=2)
    assert first == [("b", "c"), ("c", "d")]  # margins 0.02, then 0.38

    me.add_pick(winner="c", loser="b")
    everything = me.questions(list(values), scorers, budget=10)
    assert len(everything) == 5  # 6 pairs minus the one already picked
    assert ("b", "c") not in everything
    assert ("c", "b") not in everything
    assert len({frozenset(p) for p in everything}) == len(everything)
    margins = [abs(values[a] - values[b]) for a, b in everything]
    assert margins == sorted(margins)

    assert me.questions(list(values), scorers, budget=0) == []
