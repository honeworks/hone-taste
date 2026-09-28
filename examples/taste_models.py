"""Taste models: lazy loading, the GPU lease, `close()`, the license gate, and a fake backend for tests.

What: `tt.songeval`, `tt.audiobox`, `tt.reward_model` and `tt.image_preference` wrap small models trained
      on many human ratings. Each loads its weights on the first call (not when built), inside a
      `GpuLease`, keeps them until `scorer.close()`, and maps raw outputs to 0-1. `model=` swaps the real
      backend for any object with `load()`, `predict(input) -> {name: raw}` and `unload()`; here that is
      `hone_taste.testing.FakeTasteModel`, so no torch, no weights and no GPU are needed.
How:  1. share one `events` list between `FakeTasteModel` and `FakeGpuLease` to see the order of things,
      2. models with unclear license terms (SongEval, PickScore) raise `LicenseNotAccepted` until you pass
         `accept_license=True` (read the terms at the URL in the message first),
      3. build the scorer: nothing loads yet; call it: lease -> load -> predict -> release,
      4. call again: no second load; `close()` unloads; the next call loads again,
      5. read the honesty fields in `details`: `model_id`, `license`, `source_data` (whose ratings).
Why:  heavy models must not load at import or construction time, must share the GPU politely (pass
      hone-models' lease as `gpu=` in real runs) and must be freed when a batch is done. Real use:
      `tt.songeval(accept_license=True, gpu=my_lease)` with `pip install "hone-taste[songs]"`. Pitfalls:
      a wrong input kind returns `Score(None, error=...)` without touching the model; the loaded model is
      cached per scorer object, so build a scorer once and reuse it.

Run: uv run python examples/taste_models.py
"""

import hone_taste as tt
from hone_taste.errors import LicenseNotAccepted, MissingExtra
from hone_taste.testing import FakeGpuLease, FakeTasteModel

# 1. A fake SongEval backend: five dimensions on SongEval's 1-5 scale.
events = []
backend = FakeTasteModel(
    {"coherence": 4.0, "musicality": 3.5, "memorability": 4.5, "clarity": 3.0, "naturalness": 4.0},
    events=events,
)
lease = FakeGpuLease(events=events)

# 2. The license gate comes first: SongEval's terms are unclear (non-commercial parts).
# LicenseNotAccepted is a MissingExtra: catch MissingExtra to handle "cannot use this model" in one place.
try:
    tt.songeval(model=backend, gpu=lease)
except MissingExtra as exc:
    print(f"gate ({type(exc).__name__}): {exc}")
    assert isinstance(exc, LicenseNotAccepted)
else:
    raise AssertionError("expected the license gate")

# 3. Build it (after reading the terms): nothing is loaded yet.
song = tt.songeval(
    model=backend, gpu=lease, accept_license=True, weights={"memorability": 2, "musicality": 1}
)
assert events == []

first = song("song.wav")  # an audio path; the fake does not read it
print(f"songeval: value={first.value:.3f} dimensions={first.details['dimensions']}")
print("events after the first call:", events)
assert events == ["lease songeval", "load", "predict", "release songeval"]
assert first.value == (2 * 0.875 + 1 * 0.625) / 3  # memorability 4.5 -> 0.875, musicality 3.5 -> 0.625

# 4. Cached until close(); the next call loads again.
song("another.wav")
song.close()
song("third.wav")
print("events after two more calls and a close:", events[4:])
assert events[4:] == [
    "lease songeval",
    "predict",
    "release songeval",
    "unload",
    "lease songeval",
    "load",
    "predict",
    "release songeval",
]
song.close()

# A wrong input kind never reaches the model or the GPU.
before = len(events)
assert song({"prompt": "x"}).value is None and len(events) == before

# 5. Whose taste and which license travel with every score.
print("learned from:", first.details["source_data"][:70], "...")
assert first.details["model_id"] == "ASLP-lab/SongEval" and "unclear" in first.details["license"]

# A reward model (clear license, no gate): raw reward -> 0-1 through the packaged reference set.
rewards = FakeTasteModel(lambda item: {"reward": 9.0 if "because" in item["response"].lower() else -3.0})
judge = tt.reward_model(model=rewards)
good = judge({"prompt": "Why is the sky blue?", "response": "Because air scatters blue light more."})
bad = judge({"prompt": "Why is the sky blue?", "response": "Blue."})
print(
    f"reward_model: good={good.value:.2f} (raw {good.details['raw']}) bad={bad.value:.2f} "
    f"reference={good.details['reference_set']}"
)
assert good.value is not None and bad.value is not None
assert good.value > bad.value and good.details["raw"] == 9.0
assert judge("just a text").error.startswith("accepts text_with_prompt")  # needs prompt + response
judge.close()
