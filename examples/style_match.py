"""Style match: does this draft read like the author? Measure it, anchor the judge on real text, calibrate.

What: `tt.stylometry(texts)` fingerprints an author's texts (15 surface features and 80 function-word
      rates; no model) and scores how close a draft is. `tt.style_judge(texts, client)` asks an LLM judge
      that sees only real excerpts of the author. `tt.style_match(texts, client, weights=...)` combines
      the two. `tt.calibrate_style(scorers, real, imitations)` checks every scorer on held-out real texts
      against imitations of them and derives the weights. The judge here is a scripted
      `FakeDecisionClient` that, like a judge given a style description, is dazzled by drama.
How:  1. collect about ten texts by the author (this example makes eight from sentence pools),
      2. hold a few real texts out and make imitations with the same subjects (a generic text, your own
         ghostwriter's version),
      3. `calibrate_style({"stylometry": ..., "style_judge": ...}, real, {"generic": [...], "ghost": [...]})`
         and read the agreement table: a scorer that prefers imitations gets weight 0,
      4. `style_match(texts, client, weights=calibration.weights)`; store the fingerprint with
         `fingerprint(texts).to_dict()` and pass `fingerprint=Fingerprint.from_dict(...)` later.
Why:  a judge given a style *description* rewards imitations of the description: in persona-writer's
      first calibration the style-sheet judge preferred the ghostwritten imitation over the real author
      in 3 of 3 cases, while stylometry ranked all 6 pairs right. Anchor the judge on real text and check
      it against held-out real text before trusting it (design/changes/0004).

Run: uv run python examples/style_match.py
"""

import random

import hone_taste as tt
from hone_taste.testing import FakeDecisionClient

PLAIN = [
    "I wrote a small {t} last week. It's about 200 lines of C.",
    "The first version was wrong. I didn't check the bounds, so it crashed on empty input.",
    "It's not clever. It doesn't need to be.",
    "I measured it (with a plain loop and a clock) and it was fast enough.",
    "There's one catch: the {t} assumes a power-of-two size.",
    "I fuzzed it overnight. It found two bugs, both in my code, not in the idea.",
    "My tests are a single file. They run in under a second.",
    "I'm not sure it's the best design, but it's the one I understand.",
]
DRAMA = [
    "Picture this: it's 2 a.m., the coffee is cold, and my {t} has just crashed — again.",
    "Why does this keep happening? Because we trust abstractions we have never read!",
    "So I did what any stubborn programmer would do — I **rewrote it from scratch**.",
    "The lesson? Never underestimate simple code. Trust me, you'll thank yourself later.",
]
GENERIC = (
    "## Introduction\n\nIn today's fast-paced world, understanding the {t} is essential for every "
    "developer who wants to write efficient, maintainable and scalable code.\n\n- **Performance**: faster "
    "operations\n- **Reliability**: fewer bugs\n\nBy following these guidelines, you will be well on your "
    "way to building robust systems."
)
rng = random.Random(1)  # noqa: S311 - a fixed seed for sample texts, not security


def post(topic: str) -> str:
    return "\n\n".join(" ".join(rng.sample(PLAIN, 3)).format(t=topic) for _ in range(4))


reference = [post(t) for t in ("hash table", "allocator", "parser", "lock", "ring buffer", "test runner")]
held_out = ["image decoder", "shell script"]
real = [post(t) for t in held_out]
imitations = {
    "generic": [GENERIC.format(t=t) for t in held_out],
    "ghost": [" ".join(DRAMA).format(t=t) for t in held_out],
}


def dazzled(state, name, question):
    """Stands in for a judge that rewards drama (as one given a style description does)."""
    drama = min(1.0, (state.count("—") + state.count("?") + state.count("**")) / 5)
    value = 1 - drama if name == "exaggerates" else drama
    return {"type": question["type"], "value": value, "raw": 1 + 4 * value, "rationale": "scripted"}


client = FakeDecisionClient(dazzled)
scorers = {"stylometry": tt.stylometry(reference), "style_judge": tt.style_judge(reference, client)}

# 1. Calibrate on held-out real texts: which scorer prefers the real author?
calibration = tt.calibrate_style(scorers, real, imitations)
print(calibration)
assert calibration.report.scorers["stylometry"].rate == 1.0
assert calibration.weights["style_judge"] == 0.0  # it preferred the imitations: no weight

# 2. The combined scorer with the calibrated weights ranks a new real post first.
match = tt.style_match(reference, client, weights=calibration.weights)
draft_real, draft_ghost = post("linker map"), " ".join(DRAMA).format(t="linker map")
real_score, ghost_score = match(draft_real), match(draft_ghost)
print(f"style_match: real={real_score.value:.2f} ghost={ghost_score.value:.2f}")
print("furthest features of the ghost:", scorers["stylometry"](draft_ghost).reason)
assert (real_score.value or 0) > (ghost_score.value or 0)

# 3. Store the fingerprint (JSON) and score later without the reference texts' statistics again.
stored = tt.fingerprint(reference).to_dict()
again = tt.stylometry(tt.Fingerprint.from_dict(stored))
assert again(draft_real).value == scorers["stylometry"](draft_real).value
