"""AC-21: a saturated panel says so (`ceiling`, `tt.spread`), and a comparative panel tells the same
candidates apart (`panel.pairwise`, and as a hone-select pairwise judge through `for_select_pairwise`)."""

from dataclasses import dataclass, field
from typing import Any

import pytest

import hone_taste as tt

PERSONAS = ["A teenager who shares dances with friends", "A parent who hates earworms"]
IDEAS = ["a dance with a hand-clap hook", "a slow ballad about homework", "a chant about bus stops"]


@dataclass
class Candidate:
    id: str
    data: Any
    files: dict[str, str] = field(default_factory=dict[str, str])
    meta: dict[str, Any] = field(default_factory=dict[str, Any])


def enthusiastic(state: Any, name: str, question: Any) -> dict[str, Any]:
    """Like a one-LLM audience: every idea alone gets the top; asked to compare, it prefers the clap hook."""
    if question["type"] == "score":
        return {"type": "score", "value": 1.0, "raw": 5, "rationale": "fun"}
    first = state.split("\n\nB:\n")[0]
    choice = "A" if "hand-clap" in first else "B"
    return {"type": "choice", "value": 1.0, "choice": choice, "rationale": "clap"}


def test_ac21_comparative_panels() -> None:
    client = tt.testing.FakeDecisionClient(enthusiastic)
    panel = tt.audience(PERSONAS, "Would you share it?", client)

    ratings = [panel(idea) for idea in IDEAS]
    assert all(r.details["ceiling"] for r in ratings)
    report = tt.spread(ratings)
    assert report.flat
    assert report.at_ceiling

    duel = panel.pairwise(IDEAS[1], IDEAS[0])
    assert duel.details["choice"] == "b"
    assert duel.value == 0.0
    assert duel.details["personas"][1]["shown_first"] == "b"

    judge = tt.for_select_pairwise(panel, field="idea", record=False)
    assert judge.kind == "pairwise"
    assert judge.name == "audience_pairwise"
    a, b = Candidate("c1", {"idea": IDEAS[0]}), Candidate("c2", {"idea": IDEAS[2]})
    assert judge(a, b) == ("a", 1.0)
    assert judge(b, a) == ("b", 1.0)

    with pytest.raises(tt.errors.ConfigError, match="cannot compare"):
        tt.for_select_pairwise(tt.slop_score())
