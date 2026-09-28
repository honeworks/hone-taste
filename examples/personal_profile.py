"""Personal taste: learn how much each scorer should count for one person from a few pairwise picks.

What: `tt.profile(name)` keeps one person's picks (winner, loser), liked / disliked examples and fitted
      weights in a JSON file (default `${HONE_HOME:-.hone}/taste/profiles/<name>.json`). `fit` learns
      weights over your scorers (Bradley-Terry: which scorer differences predict the picks), `as_scorer`
      turns them into one scorer tuned to the person, and `questions` / `ask` spend the person's time only
      on the closest calls.
How:  1. `me = tt.profile("name", path=...)`; record picks with `me.add_pick(winner=..., loser=...)`,
      2. `me.questions(candidates, scorers, budget=2)` -> the pairs the scorers find hardest to call,
      3. `me.ask(pairs)` asks in the terminal (1 / 2 / s = skip / q = quit) and records the answers; in
         scripts and tests pass `io=hone_taste.testing.ScriptedIO([...])`,
      4. `me.fit(scorers, items=...)` -> `Weights` (weights summing to 1, accuracy on the picks),
      5. `me.as_scorer(scorers)` ranks new candidates; `me.prompt_examples()` gives a liked / disliked
         block for panel prompts; `tt.profile(name, path=...)` loads everything back.
Why:  generic taste models and panels encode other people's taste. Five to ten picks are enough to learn
      which of them this person agrees with, far cheaper than asking them to rate everything. Pitfalls:
      with few picks the weights are rough (check `weights.accuracy`, keep adding picks); `ask` refuses to
      run without a terminal instead of hanging a pipeline; every change is saved at once, so point
      `path=` somewhere temporary for experiments.

Run: uv run python examples/personal_profile.py
"""

import tempfile
from pathlib import Path

import hone_taste as tt
from hone_taste.testing import ScriptedIO

NEON = "Neon echoes whisper through the tapestry of night"
BOOTS = "Dad's boots by the door still smell like diesel, baby"
BABY = "Baby baby baby, you're my baby tonight"
KETTLE = "The kettle clicks off and nobody gets up to pour it"
WINGS = "Shattered dreams and broken wings into the night"
COAT = "Baby, the kettle's cold and your coat is still on the hook"
CANDIDATES = [NEON, BOOTS, BABY, KETTLE, WINGS, COAT]

scorers = {"slop": tt.slop_score(domain="lyrics"), "no_baby": tt.patterns(["baby", "tonight"])}
workdir = (
    tempfile.TemporaryDirectory()
)  # a fresh file per run (removed at the end); drop path= for the default
path = Path(workdir.name, "ana.json")

# 1. Picks: this person hates clichés but does not mind a "baby" in a line with a real image.
me = tt.profile("ana", path=path)
me.add_pick(winner=BOOTS, loser=NEON, context="verse 1")
me.add_pick(winner=KETTLE, loser=BABY)

# 2. Which pairs are worth asking about? The closest predicted margins first, never a pair already picked.
pairs = me.questions(CANDIDATES, scorers, budget=2)
for a, b in pairs:
    print(f"ask: {a[:30]!r} vs {b[:30]!r}")
assert len(pairs) == 2
assert all(p not in me.pairs and p[::-1] not in me.pairs for p in pairs)  # never re-asked

# 3. Ask. ScriptedIO answers "1" (the first item) to the first pair and skips the second.
io = ScriptedIO(["1", "s"])
recorded = me.ask(pairs, io=io)
print("the person saw:", io.prompts[0].strip().splitlines())
assert recorded == [pairs[0]] and len(me.picks) == 3
me.add_pick(winner=COAT, loser=WINGS)  # one more pick, e.g. from a review session

# 4. Fit weights over the scorers (items: extra candidates that show each scorer's spread).
weights = me.fit(scorers, items=CANDIDATES)
rounded = {name: round(w, 2) for name, w in weights.weights.items()}
print(f"weights={rounded} accuracy={weights.accuracy:.2f} picks={weights.picks}")
assert weights.weights["slop"] > weights.weights["no_baby"]
assert abs(sum(weights.weights.values()) - 1) < 1e-9

# 5. One scorer tuned to this person; new candidates rank the way their picks did.
mine = me.as_scorer(scorers)
scores = {text: mine(text) for text in CANDIDATES}  # score each candidate once
values = {text: s.value for text, s in scores.items() if s.value is not None}  # None = could not score
ranked = sorted(values, key=lambda text: values[text], reverse=True)
for text in ranked:
    print(f"  {values[text]:.2f}  {text}")
assert mine.name == "profile:ana" and ranked.index(BOOTS) < ranked.index(NEON)

me.examples(liked=[KETTLE], disliked=[NEON])
print(me.prompt_examples())  # paste into a panel question: tt.audience(..., question=f"...\n{block}")

# Everything was saved as it happened; load it back.
again = tt.profile("ana", path=path)
print(f"reloaded: {len(again.picks)} picks, fitted weights {again.fitted}, from {path.name}")
assert again.pairs == me.pairs and again.fitted == weights and again.liked == [KETTLE]
workdir.cleanup()
