"""Check scorers against real human picks before trusting them: the agreement report.

What: `tt.agreement(scorers, human_pairs)` scores both sides of every (winner, loser) pick a person made
      and reports, per scorer, how often it prefers the person's pick (`rate`, ties count half), a
      Kendall-style `correlation` in [-1, 1], how many pairs it could compare and how many it could not
      score. It also fits `suggested` weights over the scorers from the same picks.
How:  1. collect picks as (winner, loser) pairs (from `profile.pairs`, a review tool, a spreadsheet),
      2. `report = tt.agreement({"slop": ..., "panel": ...}, pairs)`,
      3. `print(report)` for a table, `report.to_json()` / `report.to_dict()` to store or compare runs,
      4. read `report.scorers[name]` and feed `report.suggested.weights` into `tt.combine`.
Why:  every stand-in scorer learned somebody's taste (or none); this is how you find out whether it is
      yours. A scorer near 0.5 agreement is noise for this person; one well below 0.5 is actively wrong
      for them. Pitfalls: with a handful of pairs the numbers are rough; a scorer that could not score a
      pair is counted in `unscored`, not as disagreement.

Run: uv run python examples/agreement.py
"""

import json

import hone_taste as tt
from hone_taste.testing import FakeDecisionClient

# Picks a person made: (the one they preferred, the one they did not).
HUMAN_PAIRS = [
    ("Dad's boots by the door still smell like diesel", "Neon echoes whisper through the tapestry of night"),
    (
        "The kettle clicks off and nobody gets up to pour it",
        "Shattered dreams and broken wings into the night",
    ),
    ("Rain on the tin roof, my father's boots by the door", "Baby baby baby, you're my baby tonight"),
    ("Baby, the kettle's cold and your coat is still on the hook", "Into the night we chase our neon dreams"),
]


def short_is_better(state, name, question):
    """A scripted panel with the wrong taste: it prefers shorter lines. It cannot rate texts with 'coat'."""
    if "coat" in state:
        return {"type": "score", "value": None, "error": "refused to answer"}
    return {"type": "score", "value": max(0.0, 1 - len(state) / 60), "rationale": "short and punchy"}


scorers = {
    "slop": tt.slop_score(domain="lyrics"),
    "no_baby": tt.patterns(["baby"]),
    "panel": tt.audience(
        ["A pop radio listener"], "Would you replay it?", FakeDecisionClient(short_is_better)
    ),
}
report = tt.agreement(scorers, HUMAN_PAIRS)

print(report)  # a small table: agreement, correlation, compared pairs, suggested weight
slop, panel = report.scorers["slop"], report.scorers["panel"]
print("panel:", panel)  # one ScorerAgreement: rate, correlation, compared, unscored
assert slop.rate == 0.875 and slop.correlation == 0.75  # 3 picks agree, 1 tie (both lines score 1.0)
assert panel.unscored == 1 and panel.compared == 3  # the pair with the refused text is not held against it
assert panel.rate is not None and panel.rate < 0.5  # the panel's taste is not this person's
assert report.suggested.weights["slop"] > report.suggested.weights["panel"]

# Store the report as JSON to compare runs over time.
stored = json.loads(report.to_json())
assert stored["pairs"] == len(HUMAN_PAIRS)
assert stored["suggested_weights"]["weights"] == report.suggested.weights

# Use the suggested weights directly: one scorer weighted by what agreed with this person.
tuned = tt.combine(report.suggested.weights, scorers)
winner, loser = HUMAN_PAIRS[0]
tuned_winner, tuned_loser = tuned(winner).value, tuned(loser).value
assert tuned_winner is not None and tuned_loser is not None  # None would mean "could not score"
print(f"tuned: winner={tuned_winner:.2f} loser={tuned_loser:.2f}")
assert tuned_winner > tuned_loser
