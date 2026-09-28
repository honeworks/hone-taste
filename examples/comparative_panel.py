"""Comparative panels: when every candidate gets the top rating, compare them instead.

What: a panel that rates each candidate alone often gives every candidate the top of the scale. This
      example shows how hone-taste says so (`details["ceiling"]`, `tt.spread`) and how to get a real
      preference: `panel.pairwise(a, b)` shows each persona both candidates and asks which it prefers, and
      `tt.for_select_pairwise(panel, field=...)` makes that a hone-select pairwise judge.
How:  1. build a panel with `tt.audience(...)` as usual (here over a scripted `FakeDecisionClient` that,
         like a real one-LLM audience, loves every idea when asked about one at a time),
      2. rate a batch of candidates and pass the scores to `tt.spread(...)`: `flat` / `at_ceiling` say
         the ratings told the candidates apart not at all,
      3. call `panel.pairwise(a, b)`: every second persona sees the pair in swapped order, so a judge that
         always picks the first one shown ends in a tie, not a false winner,
      4. for hone-select, register `tt.for_select_pairwise(panel, field="idea")`; it returns
         `("a" | "b" | "tie", confidence)` and hone-select asks it in both orders for near ties.
Why:  a saturated rating is a ceiling, not a preference: in a selection it costs calls and weight but adds
      no information. Comparing two candidates side by side is how people judge too, and it costs one call
      per persona per pair, no more than rating for small batches (design/changes/0002).

Run: uv run python examples/comparative_panel.py
"""

from dataclasses import dataclass, field
from typing import Any

import hone_taste as tt
from hone_taste.testing import FakeDecisionClient

PERSONAS = ["A 15-year-old who shares dance videos with friends", "A parent who hates earworms"]
IDEAS = [
    "a dance built around a hand-clap hook",
    "a slow ballad about homework",
    "a chant about waiting for the bus",
]


def one_llm_audience(state, name, question):
    """Stands in for an LLM: alone, every idea is a 5/5; side by side, the hand-clap hook wins."""
    if question["type"] == "score":
        return {"type": "score", "value": 1.0, "raw": 5, "rationale": "sounds fun!"}
    first = state.split("\n\nB:\n")[0]  # the state is "A:\n<first>\n\nB:\n<second>"
    choice = "A" if "hand-clap" in first else "B" if "hand-clap" in state else "no preference"
    return {"type": "choice", "value": 1.0, "choice": choice, "rationale": "you can clap along"}


client = FakeDecisionClient(one_llm_audience)
panel = tt.audience(PERSONAS, "Would you share it with friends?", client)

# 1. Rated alone, every idea hits the ceiling, and tt.spread says the ratings carry no information.
ratings = [panel(idea) for idea in IDEAS]
print("ratings:", [r.value for r in ratings], "ceiling:", [r.details["ceiling"] for r in ratings])
report = tt.spread(ratings)
print("spread:", report)
assert report.flat and report.at_ceiling

# 2. Compared, the personas prefer the hand-clap hook, whichever position it is shown in.
duel = panel.pairwise(IDEAS[1], IDEAS[0])
print(f"pairwise: value={duel.value} choice={duel.details['choice']} reason={duel.reason!r}")
for persona in duel.details["personas"]:
    who, first = persona["persona"][:28], persona["shown_first"]
    print(f"  {who}...: saw {first} first, prefers_a={persona['prefers_a']}")
assert duel.details["choice"] == "b" and duel.value == 0.0
assert [p["shown_first"] for p in duel.details["personas"]] == ["a", "b"]

# Two ideas neither persona prefers: a tie with confidence 0.
even = panel.pairwise(IDEAS[1], IDEAS[2])
assert even.details["choice"] == "tie" and even.confidence == 0.0


# 3. The same panel as a hone-select pairwise judge, over candidate-shaped objects.
@dataclass
class Candidate:  # hone-select passes objects with id, data, files, meta
    id: str
    data: dict[str, Any]
    files: dict[str, str] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)


judge = tt.for_select_pairwise(panel, field="idea")
clap, bus = Candidate("c1", {"idea": IDEAS[0]}), Candidate("c3", {"idea": IDEAS[2]})
print("judge:", judge.name, judge.kind, judge(bus, clap))
assert judge.kind == "pairwise" and judge(bus, clap) == ("b", 1.0)
