"""Quickstart: call a scorer, read the `Score` it returns, and see what "could not score" looks like.

What: every hone-taste scorer is a plain callable, `scorer(input) -> Score`. A `Score` has a 0-1 `value`
      (higher = better / more human-like), a `confidence`, a short `reason`, per-aspect `details` and an
      `error`. `value is None` means the scorer could not score this input; it is never a silent 0.
How:  1. build a scorer with a factory (`tt.slop_score()`),
      2. call it on a text and read `value`, `confidence`, `reason` and `details["hits"]`,
      3. look at the scorer's metadata (`name`, `family`, `accepts`, `license`, `source_data`),
      4. give it input it cannot score (a number, an empty text) and read `error` instead of `value`.
Why:  this is the one contract everything else builds on: `combine`, panels, profiles, `agreement` and the
      hone-select bridge all take and return these objects. Pitfall: `if score.value:` treats 0.0 like
      `None`; always test `score.value is None` before you use the number.

Run: uv run python examples/quickstart.py
"""

import hone_taste as tt

slop = tt.slop_score()  # scorers are cheap to build and hold no state between calls

# 1. Score a text. value 0-1: 1 = no overused LLM phrasing found.
text = "It's not just a product, but a journey through the rich tapestry of innovation."
cliched = slop(text)
plain = slop("The bus was late again, so I walked home in the rain and made soup.")
# confidence is low for short texts: 14 words say less than 100 (full confidence).
print(f"cliched: value={cliched.value:.2f} confidence={cliched.confidence:.2f} reason={cliched.reason!r}")
print(f"plain:   value={plain.value:.2f} reason={plain.reason!r}")
assert cliched.value is not None and plain.value is not None
assert cliched.value < plain.value

# 2. `details` says why: every hit with its character span, so you can show or fix it.
for hit in cliched.details["hits"]:
    start, end = hit["span"]
    print(f"  {text[start:end]!r}: {hit}")
assert any(hit.get("word") == "tapestry" for hit in cliched.details["hits"])

# 3. Metadata travels with the scorer: whose taste it encodes and under which license.
print(f"scorer: name={slop.name} family={slop.family} accepts={sorted(slop.accepts)}")
print(f"        license={slop.license!r}")
assert isinstance(slop, tt.Scorer)  # the Protocol is runtime-checkable

# 4. Could not score -> value None with an error, never 0 and never an exception.
wrong_kind = slop(42)  # the slop scorer accepts "text" only
empty = slop("   ")  # no words to judge
print(f"wrong input: value={wrong_kind.value} error={wrong_kind.error!r}")
print(f"empty text:  value={empty.value} error={empty.error!r}")
assert wrong_kind.value is None and wrong_kind.error.startswith("accepts text")
assert empty.value is None and empty.error

# Every call above was also recorded as a span under ${HONE_HOME:-.hone}/taste/ (see records_and_tracing.py).
