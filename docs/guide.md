# Guide

Every scorer is a callable `scorer(input) -> Score` with metadata (`name`, `family`, `accepts`,
`license`, `source_data`). `Score.value` is 0-1, higher is better (or more human-like); `None` means
"could not score" and `error` says why. It is never silently 0.

Inputs are typed by `accepts`: `"text"` (a `str`), `"text_with_prompt"` (`{"prompt", "response"}`),
`"audio"` / `"image"` (a path), `"image_with_prompt"` (`{"prompt", "image"}`). A wrong input gives
`Score(None, error="accepts ...")` without calling the model.

All code blocks on this page run as tests (with fakes where a model would be needed). Each section links
to a longer, explained example in [`examples/`](../examples/README.md).

## B. Human-likeness

### Slop score
A port of [slop-score](https://github.com/sam-paech/slop-score) (MIT): overused LLM words and phrases
(60%), "not X, but Y" contrasts (25%) and overused trigrams (15%), per 1,000 words.

```python
import hone_taste as tt

email = tt.slop_score(domain="email")  # presets: general, lyrics, email
s = email("I hope this email finds you well. Let's delve into the details.")
print(s.value, s.details["rates_per_1k_words"])
for hit in s.details["hits"]:
    print(hit)  # {"phrase" | "word" | "pattern" | "trigram": ..., "span": [start, end]}
```
Example: [`slop_and_patterns.py`](../examples/slop_and_patterns.py) (presets, own word lists, patterns).

### Your own patterns
```python
import re
import hone_taste as tt

banned = tt.patterns(["as an AI", re.compile(r"\bin conclusion\b", re.I)], max_hits=3)
s = banned("In conclusion, as an AI I cannot say.")
assert s.value == 1 - 2 / 3
```

### Binoculars (extra `detect`)
`tt.binoculars()` compares two related language models; `value = 1 - p(machine)`, with
`details["raw_score"]` and the `threshold` used. The default threshold was calibrated upstream for a larger
model pair, so treat the value as a signal and recalibrate on your own texts when it matters.

## A. Taste models

Heavy models load on the first call, inside a `GpuLease` (`gpu=`, default: no lease), stay cached in
the scorer, and are freed by `scorer.close()`. Tests and dry runs inject a fake backend with `model=`:

```python
import hone_taste as tt
from hone_taste.testing import FakeGpuLease, FakeTasteModel

events = []
fake = FakeTasteModel(
    {"coherence": 4.0, "musicality": 3.5, "memorability": 4.5, "clarity": 3.0, "naturalness": 4.0},
    events=events,
)
song = tt.songeval(
    model=fake,
    gpu=FakeGpuLease(events=events),
    accept_license=True,
    weights={"memorability": 2, "musicality": 1},
)
s = song("song.wav")
print(s.value, s.details["dimensions"])
song.close()
print(events)  # ['lease songeval', 'load', 'predict', 'release songeval', 'unload']
```

- `tt.songeval(...)`: five SongEval dimensions (1-5 scale mapped to 0-1). License unclear, so
  `accept_license=True` is required; downloads a 100 MB checkpoint to `${HONE_HOME:-.hone}/taste/models/`.
  Whole four-minute songs can run out of memory on an 8 GB GPU: `max_seconds=90` scores the middle 90 s,
  and after an out-of-memory error retries on half the length (down to 30 s); `details["excerpt_s"]`
  says what was scored. Without `max_seconds` the whole song is scored once, as before.
- `tt.audiobox(...)`: Meta Audiobox Aesthetics CE, CU, PC, PQ (0-10 mapped to 0-1); value = mean of CE,
  CU, PQ by default.
- `tt.reward_model(...)`: `{"prompt", "response"}` in; the raw reward is mapped to 0-1 with a reference set.
- `tt.image_preference("pickscore" | "hpsv3", reference=...)`: `{"prompt", "image"}` in.

`tt.models()` lists each model's license and whose ratings it learned from. Examples:
[`taste_models.py`](../examples/taste_models.py) (lazy loading, lease, license gate) and
[`model_registry.py`](../examples/model_registry.py).

### Objective audio checks (extra `songs`)

"Does this song annoy the ear?" has objective parts no taste model reports. `tt.audio_checks()` measures
a song on the CPU, without a model (about 3 s per song): integrated loudness and loudness range (ITU-R
BS.1770 / EBU), true peak, clipping, harshness (2-5 kHz against 250 Hz-2 kHz), sibilance, mud (200-500
Hz), spectral centroid and glitches (noise bursts and clicks); with `bpm=`, `key=` or `meter=` also tempo,
key and meter against the request (half / double / dotted tempo readings and the relative key count as a
match), and with `mastered=True` the delivery checks (-14 LUFS +- 1.5, true peak below -1 dBTP).
`value` is the share of checks passed; `details` hold every measurement, each verdict and the names in
`failed`. `tt.audio_report(path)` returns only the measurements.

<!-- not executed: needs an audio file -->
```python
checks = tt.audio_checks({"lra_min": 2.0}, bpm=120, key="A minor")  # override any default limit
result = checks("take-3.wav")
print(result.value, result.details["failed"], result.details["measurements"]["integrated_lufs"])
```

The default limits (`hone_taste.scorers.audio_checks.LIMITS`) are first guesses from about ten
generated songs: calibrate them on your own. Example: [`song_checks.py`](../examples/song_checks.py).

### Character consistency (extra `images`)

"Is this the same character as in the reference?" is a different question from "would people like this
image?": a beautiful picture of the wrong person should score low. `tt.character_consistency` compares an
image's DINOv2-small embedding (Apache-2.0, about 90 MB, fine on a CPU) with the reference and maps the
cosine from `floor` (default 0.25) to 1. A character sheet that shows the character several times side by
side is split with `panels=n`, and the candidate is compared with the closest strip. `model=` takes any
embedder with `load()`, `embed(path, part=(k, n)) -> vector` and `unload()`:

```python
import hone_taste as tt
from hone_taste.testing import FakeEmbedder

fake = FakeEmbedder({"sheet.png#0": [1.0, 0.0], "sheet.png#1": [0.0, 1.0], "new.png": [0.1, 0.9]})
same = tt.character_consistency("sheet.png", panels=2, model=fake)
s = same("new.png")
print(s.value, s.details["cosine"], s.details["panel"])  # close to the second strip
```

Embeddings measure visual similarity, not identity; check the reference and the floor against a few of
your own picks with `tt.agreement`. DINOv2 is used rather than CLIP because CLIP-style embeddings reward
the same caption or scene more than the same person. Example:
[`character_consistency.py`](../examples/character_consistency.py).

### Reference sets and normalization
Raw model outputs (logits) are mapped to 0-1 with a reference set: raw values of known-good and
known-bad examples. `linear` maps the 5th-95th percentile range to 0-1; `percentile` gives the share of
reference values below. The raw value is always kept in `details["raw"]`.

```python
from hone_taste import normalize

ref = normalize.ReferenceSet("text", "my_rm", (-3.0, 0.5, 2.0, 8.0, 12.0))
print(normalize.normalized(5.0, ref).value, normalize.normalized(5.0, ref, mode="percentile").value)
```

A reference set file is JSON: `{"good": [...], "bad": [...], "source": "..."}`; pass its path as
`reference=` or to `normalize.reference_set(domain, name, path=...)`. Example:
[`combine_and_normalize.py`](../examples/combine_and_normalize.py).

## C. Audience panel

One `decide()` call per persona; value = mean (or `min` / `median`) of the persona ratings;
`details["personas"]` holds each rating with its rationale and `details["disagreement"]` the spread.

```python
import hone_taste as tt
from hone_taste.testing import FakeDecisionClient


def rule(state, name, question):  # a scripted client: fans of rain songs rate higher
    raw = 5 if "rain" in state else 2
    return {"type": "score", "value": (raw - 1) / 4, "raw": raw, "rationale": "scripted"}


panel = tt.audience(
    ["A 25-year-old who skips songs within 20 seconds", "A 55-year-old blues fan who hates cliches"],
    question="Would you keep listening past the first chorus?",
    client=FakeDecisionClient(rule),
)
s = panel("Rain on the tin roof")
print(s.value, s.details["disagreement"], [p["rationale"] for p in s.details["personas"]])
```

Examples: [`audience_panel.py`](../examples/audience_panel.py) and
[`bring_your_own_client.py`](../examples/bring_your_own_client.py).

### When every candidate gets the top rating

A panel rates each candidate alone, so a one-LLM audience often gives every candidate the top of the
scale. Then the value is a ceiling, not a preference, and the panel adds cost but no information to a
selection. Two things help:

- **Say it.** `details["ceiling"]` is true when every persona gave the top rating, and
  `tt.spread(scores)` reports whether a scorer told a batch of candidates apart at all (`flat`,
  `at_ceiling`, `range`), so you can drop or down-weight it for that batch.
- **Compare instead.** `panel.pairwise(a, b)` shows each persona both candidates, labelled A and B, and
  asks which it prefers. Every second persona sees them in swapped order, so a judge that always picks the
  first one shown ends in a tie instead of a false winner. `value` is the share of personas preferring
  `a` (0.5 is a tie), `details["choice"]` is `"a"`, `"b"` or `"tie"`.

```python
import hone_taste as tt
from hone_taste.testing import FakeDecisionClient


def judge(state, name, question):  # rates everything 5/5 alone; compared, prefers the clap hook
    if question["type"] == "score":
        return {"type": "score", "value": 1.0, "raw": 5}
    first = state.split("\n\nB:\n")[0]
    return {"type": "choice", "value": 1.0, "choice": "A" if "clap" in first else "B"}


panel = tt.audience(["A teenager", "A parent"], "Would you share it?", FakeDecisionClient(judge))
ideas = ["a hand-clap dance", "a homework ballad"]
print(tt.spread([panel(i) for i in ideas]))  # flat=True, at_ceiling=True
duel = panel.pairwise(ideas[1], ideas[0])
print(duel.value, duel.details["choice"])  # 0.0 b
```

In hone-select, `tt.for_select_pairwise(panel, field="idea")` is a pairwise judge
(`kind = "pairwise"`, returns `(choice, confidence)`); hone-select asks it in both orders for near ties.
A comparison costs one call per persona per pair. Example:
[`comparative_panel.py`](../examples/comparative_panel.py).

Any `TextClient` also works (it is wrapped in `tt.TextDecisionClient`), e.g. the OpenAI adapter pointed
at Ollama:

<!-- not executed: needs a running server -->
```python
from hone_taste.adapters.openai import OpenAITextClient

client = OpenAITextClient("gemma3:12b", base_url="http://127.0.0.1:11434/v1", api_key="ollama")
panel = tt.audience(["A blues fan"], "Would you keep listening?", client, generator_model="qwen3:8b")
```

## D. Personal taste

A profile stores one person's picks at `${HONE_HOME:-.hone}/taste/profiles/<name>.json` and learns how
much each scorer should count for them (Bradley-Terry fit, pure Python).

```python
import hone_taste as tt

slop = tt.slop_score(domain="lyrics")
banned = tt.patterns(["baby"])
lines = [
    "Neon echoes whisper through the tapestry of night",
    "Dad's boots by the door still smell like diesel",
    "Baby baby baby, you're my baby tonight",
    "The kettle clicks off and nobody gets up",
]
me = tt.profile("guide-demo")
me.add_pick(winner=lines[1], loser=lines[0])
me.add_pick(winner=lines[3], loser=lines[2])
weights = me.fit({"slop": slop, "banned": banned}, items=lines)
print(weights.weights, weights.accuracy)
mine = me.as_scorer({"slop": slop, "banned": banned})
print(me.questions(lines, {"slop": slop, "banned": banned}, budget=2))  # the closest calls to ask next
```

`me.ask(pairs)` asks in the terminal (1 / 2 / s to skip / q to quit) and records the picks; pass
`io=hone_taste.testing.ScriptedIO([...])` in scripts and tests. `me.prompt_examples()` returns a short
"liked / disliked" block to put into panel prompts. Example:
[`personal_profile.py`](../examples/personal_profile.py).

### Style: does this read like the author?

`tt.stylometry(texts)` fingerprints an author's texts (about ten; at least two) and scores how close a
text's measurable style is: 15 surface features (sentence and paragraph length, I / we / you,
contractions, questions, dashes, parentheses, word length, headings, lists, code, bold) and Burrows'
Delta over 80 function words. No model, deterministic, blind to meaning. `tt.style_judge(texts, client)`
is an LLM judge that sees only real excerpts of the author, and `tt.style_match(texts, client,
weights=...)` combines both.

```python
import hone_taste as tt
from hone_taste.testing import FakeDecisionClient

posts = [
    "I wrote a small parser. It's 200 lines. It isn't clever (it doesn't need to be).",
    "I measured the allocator. It was fast enough. I didn't tune it further.",
    "My tests are one file. They run in a second. I've kept it that way for years.",
]
style = tt.stylometry(posts)
print(style("I fixed the lock. It's simpler now.").value, style("Unlock synergies at scale!").value)
match = tt.style_match(posts, FakeDecisionClient(), weights={"stylometry": 0.7, "style_judge": 0.3})
stored = tt.fingerprint(posts).to_dict()  # JSON; later: tt.stylometry(tt.Fingerprint.from_dict(stored))
```

**Calibrate before trusting a judge.** A judge given a style *description* rewards imitations of the
description: in persona-writer's first calibration, a style-sheet judge preferred the ghostwritten
imitation over the real author in 3 of 3 held-out posts, while stylometry ranked all 6 pairs right. Hold a
few real texts out, make imitations on the same subjects, and let `tt.calibrate_style` set the weights:

<!-- not executed: needs real held-out posts, imitations and a judge -->
```python
scorers = {"stylometry": tt.stylometry(posts), "style_judge": tt.style_judge(posts, judge_client)}
calibration = tt.calibrate_style(scorers, real_posts, {"generic": generic, "ghost": ghostwritten})
print(calibration)  # agreement with the real texts, mean values per group, derived weights
match = tt.style_match(posts, judge_client, weights=calibration.weights)
```

Each scorer is weighted by its correlation with the real texts (negative counts as 0). Example:
[`style_match.py`](../examples/style_match.py).

## Combining and checking
```python
import hone_taste as tt

slop, banned = tt.slop_score(), tt.patterns(["delve"])
both = tt.combine({"slop": 3, "banned": 1}, {"slop": slop, "banned": banned})  # None values drop out
report = tt.agreement(
    {"slop": slop, "banned": banned}, [("Plain words about rain.", "Let us delve into the tapestry of rain.")]
)
print(report)  # agreement rate, Kendall-style correlation, suggested weights
print(report.to_json())
```
Examples: [`combine_and_normalize.py`](../examples/combine_and_normalize.py),
[`custom_scorer.py`](../examples/custom_scorer.py) (your own scorer) and
[`agreement.py`](../examples/agreement.py).

## hone-select bridge
```python
import hone_taste as tt


class Candidate:  # hone-select passes objects with id, data, files, meta
    id, data, files, meta = "c1", {"lyrics": "Rain on the tin roof"}, {"audio": "c1.wav"}, {}


score_lyrics = tt.for_select(tt.slop_score(domain="lyrics"), field="lyrics")
print(score_lyrics(Candidate()).value)
```
`file="audio"` reads `candidate.files["audio"]` instead; `prompt_field="prompt"` builds
`{"prompt", "response"}` / `{"prompt", "image"}` inputs; `record=False` leaves recording to hone-select.
`tt.for_select_pairwise(panel, field=...)` turns an audience panel into a pairwise judge (see
[above](#when-every-candidate-gets-the-top-rating)).
Example: [`select_bridge.py`](../examples/select_bridge.py).
