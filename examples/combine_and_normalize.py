"""Combine several scorers into one, and map raw model outputs to 0-1 with a reference set.

What: `tt.combine(weights, scorers)` builds one scorer from several: each part scores the input, parts that
      could not score (`value is None`) drop out and the remaining weights are renormalized. The value is
      `None` only when no part produced one. `hone_taste.normalize` turns raw numbers (logits, 1-5
      ratings) into 0-1 values using the raw values of known-good and known-bad examples (a reference set).
How:  1. `tt.combine({"slop": 2, "panel": 1}, {"slop": ..., "panel": ...})`, optionally `aggregate="min"`
         or `"median"`; read the parts in `details["parts"]`,
      2. make one part fail (an LLM client that is down) and see the weights renormalize,
      3. build a `normalize.ReferenceSet` (or load one: `normalize.reference_set(domain, name, path=...)`),
      4. map raw values with `normalize.normalized(raw, ref)`: "linear" (5th-95th percentile -> 0-1,
         clamped, the default) or "percentile" (share of reference values below); `details["raw"]` keeps
         the raw number.
Why:  no single scorer is trustworthy on its own (Goodhart: optimizing hard against one finds outputs that
      fool it), so real selections combine families. Missing values must not count as 0, or a flaky
      server would sink good candidates. Raw model outputs have no fixed scale; a reference set from your
      own domain gives them one. Pitfall: `min` punishes any weak part; use it only for must-pass aspects.

Run: uv run python examples/combine_and_normalize.py
"""

import json
import tempfile
from pathlib import Path

import hone_taste as tt
from hone_taste import normalize
from hone_taste.errors import ConfigError
from hone_taste.testing import FakeDecisionClient

slop = tt.slop_score(domain="lyrics")
banned = tt.patterns(["baby", "tonight"], max_hits=2)
text = "Baby, the neon lights are calling tonight"

# 1. Weighted mean of the parts (weights need not sum to 1; they are normalized).
both = tt.combine({"slop": 2, "banned": 1}, {"slop": slop, "banned": banned})
result = both(text)
slop_value, banned_value = slop(text).value, banned(text).value
assert result.value is not None and slop_value is not None and banned_value is not None
print(f"combined={result.value:.3f}  reason={result.reason!r}")
for name, part in result.details["parts"].items():
    print(f"  {name}: value={part['value']:.3f} weight={part['weight']}")
assert abs(result.value - (2 * slop_value + 1 * banned_value) / 3) < 1e-9

strict = tt.combine({"slop": 1, "banned": 1}, {"slop": slop, "banned": banned}, aggregate="min")
weakest = strict(text).value
print(f"min aggregate={weakest:.3f}")
assert weakest == min(slop_value, banned_value)


# 2. A part that cannot score drops out: here the panel's LLM server is down.
def server_down(state, name, question):
    raise ConnectionError("connection refused")


panel = tt.audience(["A blues fan"], "Would you keep listening?", client=FakeDecisionClient(server_down))
with_panel = tt.combine(
    {"slop": 2, "banned": 1, "panel": 5}, {"slop": slop, "banned": banned, "panel": panel}
)
fallback = with_panel(text)
panel_error = fallback.details["parts"]["panel"]["error"]
print(f"panel down: combined={fallback.value:.3f} (panel error: {panel_error[:40]}...)")
assert fallback.value == result.value  # the panel's weight is left out, not counted as 0

nothing = tt.combine({"panel": 1}, {"panel": panel})(text)
print(f"no part scored: value={nothing.value} error={nothing.error[:50]!r}...")
assert nothing.value is None and nothing.error.startswith("no scorer produced a value")

# 3. A reference set: raw values some model gave known-good and known-bad examples.
ref = normalize.ReferenceSet("text", "my_reward_model", (-4.0, -2.5, -1.0, 0.5, 2.0, 3.5, 6.0, 8.0))
with tempfile.TemporaryDirectory() as folder:  # the same set as a file you can ship with your project
    path = Path(folder, "my_reward_model.json")
    path.write_text(
        json.dumps({"good": [3.5, 6.0, 8.0, 2.0], "bad": [-4.0, -2.5, -1.0, 0.5], "source": "demo"})
    )
    loaded = normalize.reference_set("text", "my_reward_model", path=path)
assert sorted(loaded.values) == sorted(ref.values)

# 4. Map raw values to 0-1; the raw value is always kept.
for raw in (-10.0, 1.0, 5.0, 20.0):
    linear = normalize.normalized(raw, ref)
    pct = normalize.normalized(raw, ref, mode="percentile")
    print(f"raw={raw:6.1f}  linear={linear.value:.2f}  percentile={pct.value:.2f}  details={linear.details}")
    assert linear.value is not None and 0.0 <= linear.value <= 1.0
    assert linear.details["raw"] == raw
assert normalize.normalized(-10.0, ref).value == 0.0 and normalize.normalized(20.0, ref).value == 1.0
assert normalize.normalized(float("nan"), ref).value is None  # a broken raw value is "could not score"

# The packaged reference set of the default reward model, and what happens for one that does not exist.
packaged = normalize.reference_set("text", "reward_model")
print(f"packaged {packaged.label}: {len(packaged.values)} values ({packaged.source[:40]}...)")
try:
    normalize.reference_set("images", "my_new_model")
except ConfigError as exc:
    print("missing set:", str(exc)[:60], "...")
else:
    raise AssertionError("expected a ConfigError")
