# hone-taste: the design today (v0.1.0)

This is the design as it stands, written for people who want to understand, use or change the package.
Why it exists is in [`README.md`](README.md); how it got here is in [`changes/`](changes/) and
[`decisions.md`](decisions.md). User documentation lives in [`../docs/`](../docs/guide.md).

## 1. Purpose

Ready-made scorers that **stand in for a human judge**. They answer "would people like this?" and "does
this feel human-made?" for text, songs and images, without asking a person to rate things many times.
Four families, cheapest first:

| Family | Idea | Scorers | Needs a human? |
|---|---|---|---|
| A. Taste models | small models trained on many human ratings predict what people would say; image embeddings compare with a reference | `songeval`, `audiobox`, `reward_model`, `image_preference`, `character_consistency` | no |
| B. Human-likeness | overused AI phrasing (slop), an AI-text detector, custom pattern lists | `slop_score`, `patterns`, `binoculars` | no |
| C. Audience panel | a language model plays several target personas; each rates the output with a reason | `audience` | no |
| D. Personal taste | learn one person's taste from 5-10 picks, or an author's style from their texts | `profile`, `stylometry`, `style_match` | a few picks or texts |

Scorers work on their own, combine with each other, and can be handed to any selection step that takes a
callable (section 5).

### Honesty rules

These are part of the design, not just advice. They appear in the user docs and, where they apply, in each
score's `details`.

- Each scorer states **whose taste** it learned (dataset, raters) and its **license**.
- Detectors are **signals, never gates by default**: they have known false positives on formal, templated
  and non-native writing.
- Personas are one language model pretending; weight them lower until they agree with real picks.
- Optimizing hard against one taste model finds outputs that fool it (Goodhart); combine families.
- A judge given a style description rewards imitations of the description; style judges see real text
  only and are checked against held-out real text (section 4.6.1).

## 2. Concepts

### Score

```python
@dataclass(frozen=True)
class Score:
    value: float | None          # 0-1, higher = better / more human-like; None = could not score
    confidence: float | None = None
    reason: str = ""
    details: Mapping[str, Any] = {}   # per-aspect values, hits, raw values, whose taste, license
    error: str = ""              # why value is None
```

`value is None` always means "could not score" and always comes with an `error`. It is never silently 0.

### Scorer

A scorer is any callable `scorer(input) -> Score` that describes itself:

| Attribute | Meaning |
|---|---|
| `name` | e.g. `slop_lyrics`, `songeval`, `audience`, `profile:alice` |
| `family` | `taste_model`, `human_likeness`, `audience`, `personal` or `combined` |
| `accepts` | the input kinds it takes (below) |
| `license` | license of the scorer or the model behind it |
| `source_data` | whose ratings or which data it learned from |

Every scorer hone-taste builds is one concrete class (`FunctionScorer`): metadata plus a plain
`score(input) -> Score` function, an optional `close()` and a `trace=` keyword. An exception while scoring
one input becomes `Score(None, error="<Type>: <message>")`; configuration errors are raised when the
scorer is built. Users may write their own scorer as any object with the attributes above.

### Input kinds

| Kind | Python value |
|---|---|
| `text` | `str` |
| `text_with_prompt` | mapping with `str` `prompt` and `response` |
| `audio` | a path (`str` or `os.PathLike`) |
| `image` | a path |
| `image_with_prompt` | mapping with `str` `prompt` and a path `image` |

Kinds are recognised by type only, with no filesystem access. An input that fits none of the scorer's
kinds gives `Score(None, error="accepts ...")` without calling the model; a missing file is reported by
the scorer as an error score.

## 3. Public API

```python
import hone_taste as tt

# A. taste models (optional extras; loaded lazily; `model=` injects a backend)
tt.songeval(device=None, gpu=None, *, weights=None, checkpoint=None, accept_license=False, model=None,
            max_seconds=None)                                                                         # "songs"; audio
tt.audio_checks(limits=None, *, bpm=None, key=None, meter=None, mastered=False)                        # "songs"; audio, no model
tt.audio_report(path, *, with_music=True) -> dict                                                    # "songs"
tt.audiobox(gpu=None, *, weights=None, accept_license=False, model=None)                                 # "songs"; audio
tt.reward_model(model_id=None, device=None, gpu=None, *, reference=None, normalization="linear",
                accept_license=False, model=None)                                                   # "text"; text_with_prompt
tt.image_preference(kind="pickscore" | "hpsv3", device=None, gpu=None, *, reference=None,
                    normalization="linear", accept_license=False, model=None)                           # "images"; image_with_prompt
tt.character_consistency(reference, *, panels=1, floor=0.25, device=None, gpu=None, model=None)          # "images"; image

# B. human-likeness
tt.slop_score(domain="general" | "lyrics" | "email", wordlists=None)              # core
tt.patterns(banned: list[str | re.Pattern], name="patterns", *, max_hits=5)       # core
tt.binoculars(observer=..., performer=..., device=None, gpu=None, *, threshold=..., accept_license=False,
              model=None)                                                        # "detect"

# C. audience panel
tt.audience(personas: list[str] | path, question: str, client: DecisionClient | TextClient, *,
            scale=(1, 5), aggregate="mean", anchors=None, generator_model=None) -> AudiencePanel
panel.pairwise(a, b) -> Score               # each persona compares two inputs (value = share preferring a)
tt.spread(scores, *, tolerance=0.05) -> Spread   # did a scorer tell a batch of candidates apart?
tt.TextDecisionClient(text_client)          # a DecisionClient over any TextClient

# D. personal taste
me = tt.profile(name, *, path=None) -> Profile
me.add_pick(winner, loser, *, context=None)
me.examples(liked=[...], disliked=[...])
me.fit(scorers, items) -> Weights            # re-weight other scorers from the picks
me.as_scorer(scorers) -> Scorer              # weighted combination tuned to the person
me.questions(candidates, scorers, budget=5)  # the most informative pairs to ask next
me.ask(pairs, *, io=ConsoleIO())             # asks in the terminal; records picks
me.prompt_examples()                         # a short liked / disliked block for panel prompts

# D. style: does a text read like this author?
tt.fingerprint(texts) -> Fingerprint         # JSON-ready: .to_dict(), Fingerprint.from_dict()
tt.stylometry(texts | fingerprint, *, name="stylometry")                      # core, no model
tt.style_judge(texts, client, *, author="the author", excerpts=5)             # judge on real excerpts
tt.style_match(texts, client, *, fingerprint=None, weights=None, author=..., excerpts=5)
tt.calibrate_style(scorers, real, imitations) -> StyleCalibration   # real vs imitation pairs, weights

# combining, normalization, agreement
tt.combine(weights, scorers, *, aggregate="weighted_mean")
tt.normalize.reference_set(domain, name, *, path=None), tt.normalize.normalized(raw, ref, *, mode="linear" | "percentile")
tt.agreement(scorers, human_pairs) -> AgreementReport

# the candidate-scorer shape (section 5)
tt.for_select(scorer, *, field=None, file=None, prompt_field=None, record=True) -> SelectScorer
tt.for_select_pairwise(panel, *, field=None, file=None, record=True) -> SelectPairwise

# registry, records, ports
tt.models(), tt.model_info(name)
tt.recording(sink), tt.current_trace(), tt.SqliteSpanSink, tt.JsonlSpanSink, tt.MemorySink, tt.NullSink
tt.DecisionClient, tt.TextClient, tt.TextResult, tt.GpuLease, tt.NullGpuLease, tt.RecordSink, tt.PORTS_VERSION
tt.errors: HoneTasteError, ConfigError, MissingExtra, LicenseNotAccepted, ModelUnavailable
tt.testing: FakeDecisionClient, FakeTextClient, FakeTasteModel, FakeEmbedder, FakeGpuLease, ScriptedIO, contracts
```

A short tour:

```python
slop = tt.slop_score(domain="lyrics")
s = slop("It's not just a song, but a journey through the tapestry of time.")
s.value              # low (0-1, higher = more human-like)
s.details["hits"]    # [{"pattern": "not X, but Y", "span": [5, 35], ...}, {"word": "tapestry", ...}]

panel = tt.audience(
    ["A 25-year-old who streams blues-rock and skips songs within 20 seconds",
     "A 55-year-old live-blues fan who hates cliches"],
    question="Would you keep listening past the first chorus?",
    client=my_decision_client)          # any DecisionClient or TextClient
panel(lyrics).details["personas"]       # per-persona value and rationale; details["disagreement"]

me = tt.profile("alice")
me.add_pick(winner=lyric_a, loser=lyric_b)
me.fit({"slop": slop, "panel": panel}, items=recent_lyrics)
mine = me.as_scorer({"slop": slop, "panel": panel})
```

## 4. Scorers

### 4.1 Slop score (core)

A port of the published [Slop Score](https://github.com/sam-paech/slop-score) idea: a weighted composite of
overused LLM words and phrases (60%), "not X, but Y" contrast patterns (25%) and overused trigrams (15%).

- The upstream word and trigram lists (MIT) ship unchanged as package data with their notice;
  `wordlists=<folder>` loads newer upstream files from a checkout.
- All three components are rates per 1,000 words. Each is scaled by its own reference maximum
  (80 words, 9 contrasts, 2.5 trigrams per 1,000 words, about twice the worst model average on the
  upstream leaderboard, because single short texts vary much more than averages) and clamped:
  `value = 1 - sum(w_c * min(1, rate_c / max_c))`.
- Trigrams are matched on content words (stopwords removed), the way the upstream lists are written.
  Contrast patterns are three surface regexes; the upstream part-of-speech stage is not ported.
- Domain presets (`general`, `lyrics`, `email`) add words and phrases only ("neon", "echoes of";
  "I hope this email finds you well"); the weights are the same for every domain. A word inside a
  matched phrase counts once.
- `details`: every hit with its kind (`word`, `phrase`, `pattern`, `trigram`) and character `span`, and
  the per-component rates. `confidence = min(1, words / 100)`. Empty text is an error score.

### 4.2 Patterns (core)

Banned literals and regexes. Literals match whole phrases case-insensitively; compiled regexes are used as
given. `value = 1 - min(1, hits / max_hits)` (`max_hits` defaults to 5); every hit is listed with its
pattern, match and span.

### 4.3 Taste models (extras, lazy)

| Scorer | Model | Input | Output |
|---|---|---|---|
| `songeval` | SongEval scoring head + MuQ encoder | audio | five dimensions (coherence, musicality, memorability, clarity of structure, naturalness of vocal breathing and phrasing), 1-5 mapped to 0-1; value = weighted mean (equal by default) |
| `audiobox` | Audiobox Aesthetics | audio | CE, CU, PC, PQ, 0-10 mapped to 0-1; value = mean of CE, CU, PQ by default (production complexity is not better-when-higher) |
| `reward_model` | Skywork-Reward-V2-Qwen3-0.6B by default | text with prompt | raw reward mapped to 0-1 through a reference set |
| `image_preference` | PickScore (default) or HPSv3 | image with prompt | raw score mapped to 0-1 through a reference set |

Every taste model is a thin scorer around a **backend**: any object with `load()`, `predict(input) ->
{name: raw}` and `unload()`.

- The first call loads the backend; every call runs inside `gpu.lease(name, vram_gb, trace=...)` (default
  `NullGpuLease`); `close()` unloads it and the next call loads again. The loaded model is cached in the
  scorer object, so create a scorer once and reuse it.
- Every factory takes `model=` so tests and users can inject a backend (`testing.FakeTasteModel`).
  Without `model=`, the factory checks that its extra is installed and raises `MissingExtra` naming it.
  Real backends import torch and friends only inside `load()`.
- `details` carry `model_id`, `license` and `source_data` (whose taste), and the raw values.
- SongEval is not pip-installable: its small scoring head is re-implemented (with a source notice in
  `THIRD_PARTY_NOTICES.md`) and loaded strictly, so a changed upstream checkpoint fails loudly. The
  checkpoint is downloaded once from upstream to `${HONE_HOME:-.hone}/taste/models/songeval/`.
- `reward_model` and `image_preference` need a reference set; a non-default `model_id` must bring its own.
  A packaged set exists for the default reward model; `image_preference` needs `reference=` for now.

#### Song quality checks

([0005](changes/0005-song-quality-scorers.md))

- **SongEval on long songs.** `songeval(max_seconds=n)` scores the middle `n` seconds (at least 30).
  After an error whose message says "out of memory", it retries inside the same lease on half the last
  length, never below 30 s; then the error stands. Without `max_seconds` the whole song is scored once
  and an out-of-memory error is an error score, exactly as before. The backend gets the limit as `predict(path, max_seconds=n)` (the real one loads only
  that window with librosa and frees GPU memory after a failed pass). `details["excerpt_s"]` is the
  length scored, `None` for the whole song.
- **`audio_checks`** (family `taste_model`, no model, CPU; numpy / scipy / librosa from the `songs`
  extra, imported only when a song is measured): measures loudness (BS.1770 K-weighted, gated
  integrated LUFS; EBU loudness range), true peak (4x oversampled), clipped share (runs of 3+ samples at
  full scale), harshness and sibilance (90th percentile of 1 s band-energy ratios against 250 Hz-2 kHz),
  mud share (median 200-500 Hz share), spectral centroid, glitches per minute (spectral-flux spikes, robust
  z > 12, on noise-like frames) and, when a request asks for them, tempo (librosa beat tracker), key
  (Krumhansl-Schmuckler on mean chroma) and a rough meter family. Checks: `no_clipping`, `not_harsh`,
  `not_sibilant`, `not_muddy`, `no_glitches`, `dynamics` (3-18 LU), plus `tempo` (4 %, half / double /
  dotted readings allowed), `key` (exact or relative), `meter` (only a clear 3 vs 4 can fail) and, with
  `mastered=True`, `loudness` (-14 +- 1.5 LUFS) and `true_peak` (<= -1 dBTP). `value` = share passed;
  `details`: `measurements`, `checks`, `failed`, `limits`. `limits=` overrides named entries of `LIMITS`;
  an unknown name is a `ConfigError`.
- MuQ-Eval and a phoneme error rate for sung words are not built: they need the owner's GPU run to
  confirm the models and their licences.

#### Character consistency

`character_consistency(reference, panels=n)` answers "is this the same character as in the reference?"
([0003](changes/0003-character-consistency.md)), for an `image` path.

- An **image embedder** (`load()`, `embed(path, part=(k, n)) -> vector`, `unload()`) embeds each
  reference once (on the first call), split into `n` equal vertical strips when `panels=n` (a character
  sheet), and the candidate. `details`: the best `cosine`, the matching `panel` (index over every strip
  of every reference) and the `floor`; `value = clamp((cosine - floor) / (1 - floor))`, default floor
  0.25 (unrelated pictures of people land about there, from teacher-persona's renders).
- Default embedder: DINOv2-small (registry entry `dinov2_small`, Apache-2.0, extra `images`), the
  L2-normalized pooled output. It runs through the same lazy load, lease and `close()` as the taste
  models; family `taste_model`, although it learned from no ratings (its `source_data` says so).
- Honesty: embeddings measure visual similarity, not identity (`details["caveat"]`).
- The judge variant (a vision `DecisionClient` asked a face / hair / accessories / outfit checklist over
  the reference and the candidate) is not built yet: it waits for several images per decision in
  hone-select (its change 0005).

#### Registry and license gate

`hone_taste/data/models.toml` lists every wrapped model: id, URL, license, whether the license is clear,
whose ratings it learned from, a VRAM estimate and its extra (`tt.models()`, `tt.model_info()`,
`hone-taste models`, and the README table, which a test keeps in sync).

When a model's terms are **not clear**, the scorer still exists but raises `LicenseNotAccepted` (a
`MissingExtra`) unless called with `accept_license=True`. In v0.1 that applies to **SongEval** (README:
CC-BY-NC-SA-4.0; repository metadata: Apache-2.0; its MuQ encoder: CC-BY-NC-4.0) and **PickScore** (no
license on the model card). A model the user picks by id (`reward_model(model_id=...)`, a custom
Binoculars pair) is not gated: the registry cannot know its terms, so `license` points to its model card.
Whether shipping these wrappers is acceptable at all is **awaiting owner review**
([decisions.md](decisions.md#awaiting-owner-review)).

### 4.4 AI-text detector (extra `detect`)

`binoculars` implements the Binoculars method (perplexity ratio of an observer / performer model pair).
The default pair is Qwen2.5-0.5B and its instruct variant (Apache-2.0, fits an 8 GB GPU).
`p(machine) = sigmoid((threshold - score) / 0.05)`, `value = 1 - p(machine)`; `details` carry `raw_score`
and the threshold. The default threshold is the upstream low-false-positive one, calibrated for a larger
model pair, so it is approximate here and configurable. Family `human_likeness`;
`confidence = min(1, tokens / 128)`.

### 4.5 Audience panel (core, needs a client)

- One `decide()` call per persona, each with one `score` question named `rating`
  (`"You are <persona>.\n<question>"`, the scale and anchors). No batching: putting all personas into one
  prompt would let them bleed into each other.
- `anchors=` defaults to low "not at all", middle "somewhat", high "absolutely". Persona files are TOML
  with `personas = [...]`.
- `client` is a `DecisionClient` or a `TextClient`; a `TextClient` is wrapped in `TextDecisionClient` (one
  JSON-schema completion per `decide`, answers marked `calibrated=False`).
- Text is the state; an image path is attached with `images=[path]`. Audio is not accepted directly:
  pass a text description (lyrics, SongEval scores).
- Aggregates: `mean`, `min`, `median`. `details["personas"]` holds each persona's value and rationale;
  `details["disagreement"]` is the population standard deviation (`None` with fewer than two answers).
- A persona whose call fails keeps its error in `details`; the aggregate uses the rest; `value` is `None`
  only if every persona failed.
- `generator_model=` enables a warning (a `UserWarning` and `details["warning"]`) when the panel's model
  is from the same family as the generator (leading letters of the model name, without provider prefix).
- `details["ceiling"]` is true when every answering persona gave the top of the scale: the value is then a
  ceiling, not a preference ([0002](changes/0002-audience-panels-that-compare.md)). `tt.spread(scores)`
  reports over a batch of candidates whether a scorer told them apart: `range` (max - min), `flat`
  (range within `tolerance`), `at_ceiling` (every value within `tolerance` of 1), `known`, `unscored`.
- **Comparing.** `audience()` returns an `AudiencePanel` (a `FunctionScorer` with one more method).
  `panel.pairwise(a, b)` makes one `decide()` call per persona with a `choice` question named
  `preference` (options `A`, `B`, `no preference`) over the state `"A:\n<a>\n\nB:\n<b>"`, or, when
  both inputs are image paths, the two images attached in order. Every second persona sees the pair in
  swapped order, so a judge that always picks the first one shown ends in a tie. Each persona's
  `prefers_a` is 1, 0 or 0.5; `value` = their mean; `details["choice"]` = `"a"` (value > 0.5), `"b"` or
  `"tie"`; `confidence = |value - 0.5| * 2`. Errors per persona as for ratings. Recorded as a
  `hone.taste.score` span with `hone.taste.panel.mode = "compare"`. A batch mode (`panel.rank(items)`)
  is not built: pairwise covers hone-select's tie-breaks, the case that needed it.

### 4.6 Personal profiles (core)

- Stored as JSON at `${HONE_HOME:-.hone}/taste/profiles/<name>.json` (picks, examples, fitted weights,
  dates), written atomically on every change. Names use `[A-Za-z0-9_.-]`. Items must be JSON-compatible.
- `fit` is a Bradley-Terry / logistic regression in pure Python: one example per pick, features
  `(s_k(winner) - s_k(loser)) / spread_k`, no intercept, L2 = 1.0, deterministic gradient steps.
  Coefficients are mapped back to the raw scale, negatives clipped to 0 and normalized to sum 1, so they
  drive `combine`'s weighted mean. A missing value counts as no evidence. If no scorer predicts the picks,
  the weights are equal with a note. `Weights` also reports accuracy and log loss on the picks.
- `questions` ranks candidate pairs by the smallest predicted margin (fitted weights, else equal), ties
  broken by the largest disagreement between scorers; it never repeats a picked pair, skips unscored
  items and respects the budget.
- `ask(pairs, io=...)` takes any object with `ask(prompt) -> str`; replies `1`, `2`, `s` (skip), `q`
  (quit). `ConsoleIO` refuses to run when stdin is not a terminal; `testing.ScriptedIO` scripts it.
- `as_scorer` is a weighted combination named `profile:<name>` with family `personal`.

### 4.6.1 Style match (core; the judge needs a client)

"Does this read like this author?" ([0004](changes/0004-style-match-scorer.md)). The honesty rule: **a
judge given a style description rewards imitations of the description; anchor it on real text and check
it against held-out real text before trusting it.** (persona-writer's style-sheet judge preferred its own
imitations over the real author in 3 of 3 held-out posts; stylometry ranked all 6 pairs right.)

- `fingerprint(texts)` (at least two texts; about ten give a stable one): `(mean, spread)` of 15 surface
  features per text (words per sentence and its spread, sentences per paragraph, I / we / you rates,
  contractions, questions, dashes, parentheses, word length, headings, list items, code share, bold) and
  of the rates of 80 English function words, plus the author's typical Burrows' Delta. Code blocks are
  left out of the prose statistics. A spread is floored at 25 % of the mean (and 0.05).
- `stylometry(...)`: family `personal`, `accepts = {"text"}`. `value = 0.5 * surface + 0.5 * words`,
  where `surface` = mean over features of `max(0, 1 - |z| / 3)` and `words = exp(-max(0, Delta -
  typical) / typical)`. `details`: `surface`, `function_words`, `delta`, `typical_delta`, the four
  `furthest` features with their z-scores, `reference_texts`; `confidence = min(1, words / 400)`. A text
  with no words is an error score.
- `style_judge(...)`: family `audience`. Up to `excerpts` prose paragraphs of 30-150 words, round robin
  over the reference texts, are the only reference the judge sees (never a style description). One
  `decide()` call with a 1-5 `voice` score (exaggeration counts as a mismatch), a `plain` yes/no and an
  `exaggerates` yes/no (inverted); value = mean of what was answered; texts clipped to 2,500 words.
- `style_match(...)`: `combine` of the two halves (default weights 0.5 / 0.5), named `style_match`,
  family `personal`; if the judge fails, the value is stylometry's and the error stays in
  `details["parts"]`.
- `calibrate_style(scorers, real, imitations)`: `imitations` maps group names (e.g. `generic`, `ghost`)
  to texts aligned with `real`; the pairs `(real[i], group[i])` go through `agreement` (every text scored
  once per scorer). `weights` = each scorer's correlation clipped at 0, normalized (equal with a `note`
  when none agrees); `means` per group and scorer; `str()` prints the table, `to_json()`.

### 4.7 Normalization and reference sets

A reference set holds raw values of known-good and known-bad examples for one domain and scorer
(`references/<domain>/<scorer>.json`, or a user's file: `{"good": [...], "bad": [...], "source": ...}`).
`linear` maps the 5th-95th percentile range to 0-1 (clamped); `percentile` gives the share of reference
values below. The raw value is always kept in `details["raw"]` and the set in `details["reference_set"]`.

### 4.8 Combining

`combine(weights, scorers, aggregate=...)` with `weighted_mean` (alias `mean`), `min` or `median`. `None`
values and zero weights drop out and the remaining weights are renormalized; nothing left gives `None`.
A combined scorer has family `combined`; its `license` and `source_data` list those of its parts.

### 4.9 Agreement check

`agreement(scorers, human_pairs)` reports per scorer, over the pairs where both items got a value: the
agreement `rate` (ties count half), a Kendall-style `correlation` ((agree - disagree) / compared), and the
numbers `compared` and `unscored`. Suggested weights come from the same fit as profiles. The report prints
as a table and has `to_dict()` / `to_json()`.

## 5. The candidate-scorer shape

Selection engines usually call scorers with a *candidate* object, not the raw input. `for_select` turns
any hone-taste scorer into such a callable, without importing any selection package.

- A candidate exposes `id`, `data`, `files` (name to path) and `meta`. The bridge reads `data[field]` (or
  all of `data`), or `files[file]` for audio and image scorers; with `prompt_field` it builds
  `{"prompt", "response"}` or `{"prompt", "image"}` inputs.
- It returns the `Score`, which has the fields such callers read (`value`, `confidence`, `reason`,
  `details`, `error`), and exposes `name`, `accepts` and `version`.
- It accepts an optional `trace=` keyword so the caller can link hone-taste's spans to its own trace;
  `record=False` sends hone-taste's spans to `NullSink` when the caller records scores itself.
- A candidate missing the field or file gives an error score. `file=` on a scorer that takes no file
  kind is a `ConfigError`.

`for_select_pairwise(panel, field=... | file=...)` is the same for a pairwise judge: it reads the input of
two candidates, calls `panel.pairwise` and returns `("a" | "b" | "tie", confidence)`; it has
`kind = "pairwise"`, `name` (`<panel>_pairwise`) and `version`, and raises when the comparison failed
(hone-select records a failed judge as a tie with the error).

Using it with hone-select is described in [`../docs/guide.md`](../docs/guide.md#hone-select-bridge).

## 6. Ports

hone-taste talks to the outside only through these `typing.Protocol`s. Anything with the right methods
fits; no subclassing, no registration. Payloads are plain JSON-compatible data or objects with the listed
attributes (both are accepted), unknown extra keys are ignored, and `PORTS_VERSION = "1"`.

| Port | Shape | Used by | Shipped |
|---|---|---|---|
| `TextClient` | `complete(messages, *, schema=None, trace=None, **params) -> TextResult` | panels (wrapped) | `adapters.openai.OpenAITextClient` (extra `openai`), `testing.FakeTextClient` |
| `DecisionClient` | `decide(state, questions, *, images=(), trace=None) -> {name: Answer}` | `audience` | `TextDecisionClient`, `testing.FakeDecisionClient` |
| `GpuLease` | `lease(name, vram_gb, *, timeout_s=None, trace=None)` context manager, reentrant per name | taste models (`gpu=`) | `NullGpuLease` (default), `testing.FakeGpuLease` |
| `RecordSink` | `emit(span)`, `flush()`, `close()`; never raises into the caller | recording | SQLite, JSONL, memory and null sinks |
| taste-model backend | `load()`, `predict(input) -> {name: raw}`, `unload()` | taste models (`model=`) | `testing.FakeTasteModel` |
| image embedder | `load()`, `embed(path, part=(0, 1)) -> Sequence[float]`, `unload()` | `character_consistency` (`model=`) | DINOv2-small, `testing.FakeEmbedder` |

- `TextResult` has `text`, `parsed` (the object validated against `schema`, or `None`), `error`, `model`,
  `finish_reason`, `usage` and `span_id`. Model-quality problems set `error`; transport and configuration
  errors raise a `RuntimeError` subclass, which the panel turns into an error for that persona.
- A `score` question has `type`, `instructions`, `scale` (default `[1, 5]`) and optional `anchors`; its
  answer has `type`, `value` (normalized 0-1, or `None` with `error`), `raw`, `confidence`, `calibrated`
  and `rationale`. A missing answer counts as an error.
- **Trace context** is a mapping of strings: a W3C `traceparent` plus optional `hone.run_id`, `hone.item`,
  `hone.step`, `hone.candidate_id`, `hone.scorer` and `hone.lens.finding_id`. Every scorer call takes
  `trace=`; `current_trace()` returns the active context (a `ContextVar`), so nested scorers inherit it.
  The panel passes it to its client and taste models to their lease. Without a context a new trace starts.
- `hone_taste.testing.contracts` holds a checker per port (`check_text_client`, `check_decision_client`,
  `check_gpu_lease`, `check_record_sink`); the fakes pass them, and so should any implementation.

## 7. Records

Every scorer call writes one span `hone.taste.score` (kind `internal`) in the OpenTelemetry-shaped span
format shared by the honeworks packages (schema version 1). Nested scorers (`combine`, panels, profiles)
write child spans in the same trace. A user's own scorer object records nothing unless wrapped.

| Attribute | Meaning |
|---|---|
| `hone.schema_version` | `"1"` |
| `hone.taste.scorer`, `hone.taste.family` | scorer name; `taste_model`, `human_likeness`, `audience`, `personal`, `combined` |
| `hone.taste.model_id`, `hone.taste.license` | taste models: the model actually run; the license |
| `hone.taste.value` | 0-1; left out (never 0) when the scorer could not score, and the span status is `error` |
| `hone.taste.raw`, `hone.taste.reference_set` | the raw value (several named raw values as JSON); the reference set used |
| `hone.taste.confidence`, `hone.taste.reason`, `hone.taste.error` | from the `Score` |
| `hone.taste.details` | `Score.details` as JSON |
| `hone.taste.panel.mode` | `"compare"` on a panel's pairwise comparison; absent on ratings |
| `hone.run_id`, `hone.item`, `hone.step`, `hone.candidate_id`, `hone.scorer`, `hone.lens.finding_id` | copied from the trace context when present |

- **Where:** the sink in a `ContextVar` set by `tt.recording(sink)`; by default a SQLite store at
  `${HONE_HOME:-.hone}/taste/spans.db`, opened on the first span. The store has the shared tables
  `meta`, `spans`, `blobs` and `changes`, runs in WAL mode with a busy timeout (the switch to WAL is
  retried so several processes can create one store at once) and commits every span (spans are few and
  small; nothing is buffered that a crash could lose).
- **Never breaks scoring:** a sink failure is reported once on stderr and counted in `sink.failures`.
- **Content capture:** with `HONE_CAPTURE_CONTENT=0` (or `capture_content=False` on a sink),
  `hone.taste.details` and `hone.taste.reason` become `{"sha256", "len"}`.
- **Secrets:** `sk-...` and `Bearer ...` strings, and the values of environment variables whose names end
  in `API_KEY`, `TOKEN`, `SECRET` or `PASSWORD`, are replaced by `***` before anything is written.

## 8. Command line (extra `cli`)

```text
hone-taste score --scorer slop|binoculars|songeval|audiobox FILES [--domain] [--accept-license] [--json]
hone-taste profile ask --name NAME --items DIR [--scorer ...] [--budget N]
hone-taste agreement --name NAME [--scorer ...] [--json]
hone-taste models
```

`profile ask` compares the `.txt` / `.md` files in a folder and refuses to run without a terminal.
`agreement` uses the profile's picks as the human pairs. Exit codes: 0 ok, 1 some file could not be
scored, 2 usage or configuration error. `reward_model` and `image_preference` need prompt pairs, so they
are Python-only. Without the extra, the `hone-taste` command says to install `hone-taste[cli]`.

## 9. Modules

| Module | Job |
|---|---|
| `types.py` | `Score`, the `Scorer` protocol, input kinds, `FunctionScorer` |
| `scorers/` | `slop`, `patterns`, `songeval`, `audiobox`, `reward_model`, `image_preference`, `binoculars`, `character` (consistency), `audio_checks` / `audio_measure`; `model.py` wraps backends (lazy load, lease, `close()`) |
| `audience.py`, `text_decisions.py` | the panel; `TextDecisionClient` |
| `profile.py`, `fitting.py`, `agreement.py` | profiles, the pure-Python fit, the agreement report and `spread` |
| `scorers/stylometry.py`, `style.py` | fingerprints and stylometry; the excerpt judge, `style_match`, `calibrate_style` |
| `combine.py`, `normalize.py` | aggregates; reference sets and mappings |
| `bridge.py` | `for_select` |
| `registry.py`, `data/models.toml` | model metadata, the license gate |
| `ports.py`, `testing/`, `adapters/openai.py` | ports, fakes and contract checkers, the OpenAI-compatible adapter |
| `_tracing.py`, `_records.py` | trace context and span sinks |
| `cli.py` | the `hone-taste` command |

The core depends only on the standard library and pydantic and never imports an optional extra or
another honeworks package.

## 10. Acceptance cases

The behaviour the package guarantees. Each has a test in `tests/e2e/` named `test_ac<N>_*`; cases marked
**[real]** also run against real models in `tests/gpu/`.

| AC | Scenario | Expected |
|---|---|---|
| AC-1 | Slop on a planted AI-ish paragraph vs a human paragraph | the AI-ish one scores lower; hits listed with spans; deterministic |
| AC-2 | Slop domain presets | the lyrics preset catches lyric cliches; the email preset catches email cliches |
| AC-3 | Patterns scorer with a regex and a literal | hits and value correct |
| AC-4 | Audience panel with `FakeDecisionClient` | one question per persona, anchors passed, aggregate, disagreement and rationales in details |
| AC-5 | Panel client error for one persona | that persona's result has the error; the aggregate uses the rest; `value=None` only if all fail |
| AC-6 | Profile: add picks, fit weights on synthetic data where scorer X predicts the picks | X gets the larger weight; weights persisted and reloaded |
| AC-7 | `profile.questions` | closest-margin pairs first, within budget, no repeats |
| AC-8 | `profile.ask` with scripted IO | picks recorded; without a terminal and without scripted IO, a clear error |
| AC-9 | Normalization with a reference set | values in [0, 1]; percentile mode; raw kept |
| AC-10 | `combine` | weighted mean with `None` handling (renormalized) |
| AC-11 | `for_select` | reads `candidate.data[field]` / `candidate.files[file]`, returns a score object, works with a minimal fake candidate |
| AC-12 | Input kind mismatch | `Score(None, error=...)`, no exception |
| AC-13 | Taste-model wrappers with an injected fake backend | loads lazily, the GPU lease is held around load and inference, `close()` releases |
| AC-14 | License gate | an unclear-license model without `accept_license=True` raises a license error |
| AC-15 | Agreement report on synthetic picks | rates and suggested weights correct |
| AC-16 | Records | `hone.taste.score` spans with the attribute names in section 7 |
| AC-17 | Import boundaries | the core imports without torch or transformers installed |
| AC-18 **[real]** | Audience panel through the OpenAI-compatible adapter to a local model server | the contract checker passes; a cliched lyric scores lower than a fresh one on a two-persona panel (tolerant assertion) |
| AC-19 **[real, optional]** | `reward_model` (smallest Skywork-V2) and `songeval` on a short generated WAV | values in [0, 1]; skipped with a reason if weights cannot be downloaded or licenses block |
| AC-20 | Examples | every `examples/*.py` runs offline, opens with a What / How / Why docstring, uses only the public API and is listed in `examples/README.md` |
| AC-21 | Saturated and comparative panels | a panel rating every candidate at the top sets `ceiling` and `tt.spread` reports `flat` / `at_ceiling`; `panel.pairwise` prefers the right candidate in both orders; `for_select_pairwise` works as a pairwise judge on fake candidates |
| AC-22 | Character consistency with a fake embedder | a candidate close to one panel of a split reference sheet scores high and names that panel; a different character scores below the floor; works through `for_select(file=...)`; an unreadable image is an error score |
| AC-23 | Style match on synthetic texts | stylometry ranks held-out real posts above generic and exaggerated imitations; a judge that rewards the imitation gets weight 0 in `calibrate_style`; `style_match` with those weights ranks real first; a failed judge falls back to stylometry; a fingerprint round-trips through JSON |
| AC-24 | Song quality | SongEval with `max_seconds=90` and a backend that runs out of memory above 60 s scores a 45 s excerpt and says so; `audio_checks` scores a clean synthetic song above a clipped, hissy one and lists the failed checks |

Also guaranteed by tests: every Python block in the README and `docs/` runs, the README license table
matches the registry, and the README quickstart runs from a fresh wheel install without extras.

## 11. Examples

[`examples/`](../examples/README.md) has one runnable, explained file per public concept, using only the
public API and the package's fakes (no network, no GPU). At least these exist:

| File | Concept |
|---|---|
| `quickstart.py` | `Score`, calling a scorer, `None` never 0 |
| `slop_and_patterns.py` | `slop_score` presets and `patterns` |
| `combine_and_normalize.py` | `combine`, reference sets, percentiles, `None` handling |
| `audience_panel.py` | `audience` over a `FakeDecisionClient` |
| `comparative_panel.py` | the ceiling flag, `spread`, `panel.pairwise`, `for_select_pairwise` |
| `personal_profile.py` | picks, `fit`, `as_scorer`, `questions` / `ask` with scripted IO |
| `agreement.py` | the agreement report and suggested weights |
| `style_match.py` | `stylometry`, `style_judge`, `calibrate_style`, `style_match`, stored fingerprints |
| `taste_models.py` | taste-model wrappers with `FakeTasteModel`, lazy loading, GPU lease, license gate |
| `song_checks.py` | `audio_checks`, `audio_report`, SongEval's out-of-memory retries |
| `character_consistency.py` | `character_consistency` over a split reference sheet with `FakeEmbedder` |
| `select_bridge.py` | `for_select` with a candidate-shaped object |
| `lyrics_taste.py` | everything together: choosing lyric drafts with slop, a listener panel and a few picks |

## 12. Known limitations (v0.1.0)

- Binoculars' default threshold was calibrated upstream for a larger model pair; with the small default
  pair it is approximate (on the real test texts it still separated human from machine text).
- SongEval / MuQ weights are non-commercial or unclear, and PickScore states no license: hence the gate.
- The reward-model reference set comes from 12 hand-written prompts: enough to map to 0-1, not a
  calibrated scale. There is no packaged set for PickScore or HPSv3; `image_preference` needs
  `reference=`.
- PickScore and HPSv3 backends have not been run on real weights (HPSv3 needs about 18 GB of VRAM and its
  own `hpsv3` package, so it is not in an extra).
- Loaded models are cached per scorer object, not per process.
- The `audio_checks` limits are first guesses from about ten generated songs; SongEval's excerpt
  retries have been tested with fakes only.
- Stylometry's function words are English; the excerpt-anchored `style_judge` has not yet been
  calibrated on real held-out posts (the first real calibration measured the style-sheet judge).
- `character_consistency`'s floor (0.25) comes from one app's renders of one stylised character, and the
  real DINOv2 backend is covered only by a `-m gpu` test; there is no judge-based consistency check yet.
- The `songs` extra pins `transformers<5` because MuQ breaks on transformers 5.

## 13. Non-goals

Training taste models from scratch; a "make AI text undetectable" tool; video taste; a web page for
picks (the command line asks); batched panel questions; MuseCritic and FSPO-style few-shot personal
reward models (possible later additions).
