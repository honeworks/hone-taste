"""AC-10: combine is a weighted mean with None handling (weights renormalized)."""

from dataclasses import dataclass
from typing import Any

import pytest

import hone_taste as tt


@dataclass
class Fixed:
    """A user-written scorer: anything matching the tt.Scorer protocol works."""

    name: str
    result: float | None
    family: str = "human_likeness"
    accepts: frozenset[str] = frozenset({"text"})
    license: str = "Apache-2.0"
    source_data: str = "none"

    def __call__(self, input: Any, /) -> tt.Score:
        return tt.Score(self.result, error="" if self.result is not None else "could not score")


def test_ac10_combine() -> None:
    scorers = {"slop": Fixed("slop", 0.9), "panel": Fixed("panel", 0.3), "taste": Fixed("taste", None)}
    assert isinstance(scorers["slop"], tt.Scorer)

    combined = tt.combine({"slop": 1.0, "panel": 3.0, "taste": 4.0}, scorers)
    result = combined("some lyric")
    # taste failed -> its weight drops out; (0.9 * 1 + 0.3 * 3) / 4 = 0.45
    assert result.value == pytest.approx(0.45)
    assert result.details["parts"]["taste"]["error"] == "could not score"

    both_fail = tt.combine({"taste": 1.0}, scorers)("x")
    assert both_fail.value is None
    assert both_fail.error

    assert tt.combine({"slop": 1.0, "panel": 1.0}, scorers, aggregate="min")("x").value == pytest.approx(0.3)

    class Broken(Fixed):
        def __call__(self, input: Any, /) -> tt.Score:
            raise RuntimeError("model server down")

    partial = tt.combine({"slop": 1.0, "broken": 1.0}, {**scorers, "broken": Broken("broken", None)})("x")
    assert partial.value == pytest.approx(0.9)  # a raising part drops out like a None
    assert partial.details["parts"]["broken"]["value"] is None
    assert partial.details["parts"]["broken"]["error"] == "RuntimeError: model server down"

    zero = tt.combine({"z": 1.0}, {"z": Fixed("z", 0.0)})("x")
    assert zero.value == 0.0  # zero stays zero, not None
