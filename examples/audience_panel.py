"""An audience panel: an LLM plays several target personas and each rates the input with a reason.

What: `tt.audience(personas, question, client)` asks every persona the same question through a
      `DecisionClient` (one `decide()` call per persona, a `score` question on `scale` with `anchors`) and
      aggregates the ratings. `details` keeps each persona's value and rationale, the disagreement
      (standard deviation) and a caveat. Here the client is `hone_taste.testing.FakeDecisionClient`, a
      scripted stand-in, so the example runs offline; swap in any real client without other changes.
How:  1. write the personas as short, concrete descriptions (a list, or `personas = [...]` in a TOML file),
      2. pass the question, the client and optionally `scale`, `anchors` and `aggregate` ("mean", "min",
         "median"),
      3. call the panel on a text (or an image path, which is attached as an image),
      4. read `value`, `details["personas"]`, `details["disagreement"]`; inspect `client.calls` to see
         exactly what was asked.
Why:  a panel is the cheapest stand-in for "would these people like it?" when no taste model covers the
      domain. It is one LLM pretending, so it shares that model's taste and tends to be too positive:
      weight it lower until `tt.agreement` shows it matches real picks. One failing persona never sinks
      the panel: its error is kept and the others are aggregated; `value` is `None` only if all fail.

Run: uv run python examples/audience_panel.py
"""

import tempfile
from pathlib import Path

import hone_taste as tt
from hone_taste.testing import FakeDecisionClient

PERSONAS = [
    "A 25-year-old who streams blues-rock and skips songs within 20 seconds",
    "A 55-year-old live-blues fan who hates cliches",
]


def scripted_listener(state, name, question):
    """Stands in for an LLM: `state` is the text, `question["instructions"]` starts with the persona."""
    young = "25-year-old" in question["instructions"]
    cliched = any(word in state.lower() for word in ("neon", "tapestry", "echoes"))
    raw = 2 if cliched else (4 if young else 5)  # on the 1-5 scale
    low, high = question["scale"]
    reason = "stock phrases" if cliched else "a concrete, lived-in image"
    return {"type": "score", "value": (raw - low) / (high - low), "raw": raw, "rationale": reason}


client = FakeDecisionClient(scripted_listener)
panel = tt.audience(
    PERSONAS,
    question="Would you keep listening past the first chorus?",
    client=client,
    scale=(1, 5),
    anchors={"1": "skip at once", "3": "maybe", "5": "play it again"},
)

fresh = panel("Dad's work boots by the door still smell like diesel and rain")
cliched = panel("Neon echoes whisper through the tapestry of night")
print(f"fresh:   value={fresh.value:.3f} disagreement={fresh.details['disagreement']:.3f}")
print(f"cliched: value={cliched.value:.3f} reason={cliched.reason!r}")
for rating in fresh.details["personas"]:
    print(f"  {rating['persona'][:30]}...: raw={rating['raw']} ({rating['rationale']})")
print("caveat:", fresh.details["caveat"])
assert fresh.value == (0.75 + 1.0) / 2 and cliched.value == 0.25
assert fresh.details["disagreement"] == 0.125  # population stdev of the persona values

# What the client was asked: one call per persona, a score question with the scale and anchors.
asked = client.calls[0]["questions"]["rating"]
print("asked:", {k: asked[k] for k in ("type", "scale", "anchors")})
assert len(client.calls) == 4 and asked["instructions"].startswith(f"You are {PERSONAS[0]}.")

# Personas from a TOML file, and the min aggregate (the least convinced persona decides).
with tempfile.TemporaryDirectory() as folder:
    toml = Path(folder, "personas.toml")
    toml.write_text(
        'personas = [\n  "A 25-year-old who streams blues-rock and skips songs within 20 seconds",\n'
        '  "A 55-year-old live-blues fan who hates cliches",\n]\n'
    )
    strict = tt.audience(toml, "Would you keep listening?", client, aggregate="min")
    assert strict("Dad's boots by the door").value == 0.75


# One persona's call fails: its error is kept, the rest are aggregated.
def flaky(state, name, question):
    if "55-year-old" in question["instructions"]:
        raise TimeoutError("the model did not answer in 30 s")
    return scripted_listener(state, name, question)


partial = tt.audience(PERSONAS, "Would you keep listening?", FakeDecisionClient(flaky))("Dad's boots")
failed = [p for p in partial.details["personas"] if p["error"]]
print(f"one persona down: value={partial.value:.2f} reason={partial.reason!r} error={failed[0]['error']!r}")
assert partial.value == 0.75 and partial.reason == "mean of 1/2 personas"

# An image path is attached as an image; the state then says so.
with tempfile.TemporaryDirectory() as folder:
    cover = Path(folder, "cover.png")
    cover.write_bytes(b"\x89PNG\r\n\x1a\n")  # any existing .png / .jpg path is sent as an image
    viewer = FakeDecisionClient()  # default answers: the middle of the scale
    tt.audience(["A record-store owner"], "Would you pick this cover up?", viewer)(str(cover))
print("image call:", viewer.calls[0]["state"], viewer.calls[0]["images"][0].endswith("cover.png"))
assert viewer.calls[0]["images"] == [str(cover)]
