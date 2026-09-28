# Decisions

Implementation choices too small for a change record, made while building 0.1.0 (September 2026). The
numbers (D-001 ...) are stable, so the changelog and code comments can refer to them. The design they
belong to is [`changes/0001-initial-design.md`](changes/0001-initial-design.md).

Each entry says what the question was, what was chosen and why. **Review:** says whether a human has
confirmed it.

## Awaiting owner review

> These decisions were made by the builders and have **not yet been confirmed by the owner**.

| Entry | Question for the owner | Current state |
|---|---|---|
| [D-006](#d-006-slop-lists-are-shipped-from-upstream-mit) | **Licensing.** Confirm that the upstream slop-score word and trigram lists are MIT and may be redistributed. | shipped unchanged with the MIT notice |
| [D-003](#d-003-family-and-license-of-combined-scorers) | Accept `combined` as a fifth value of `hone.taste.family` (the shared span format lists four), and add it to that format's description? | `combine(...)` scorers record family `combined` |

When the owner decides, the entry's **Review:** line records the outcome; a decision that changes what
the package promises gets a change record.

## D-001: Input kinds and the accepts check

- **Question:** how is each input kind recognised, so that a mismatch gives `Score(None, error=...)`?
- **Options:** content sniffing (extension, file exists); plain type checks.
- **Choice:** type checks only: `text` = `str`; `text_with_prompt` = mapping with `str` `prompt` and
  `response`; `audio` / `image` = `str` or `os.PathLike`; `image_with_prompt` = mapping with `str`
  `prompt` and a path `image`. A missing file is reported by the scorer itself as an error score.
- **Why:** the simplest deterministic rule; no filesystem access before scoring.
- **Review:** not needed.

## D-002: One concrete scorer class; exceptions become error scores

- **Choice:** `tt.Scorer` is a Protocol; every scorer hone-taste builds is a `FunctionScorer` (metadata, a
  plain `score(input) -> Score` function, optional `on_close`). An exception while scoring one input
  becomes `Score(None, error="<Type>: <message>")`; configuration errors are raised when the scorer is
  built.
- **Why:** one place for the shared behaviour (input check, errors, records); no class hierarchy.
- **Review:** not needed.

## D-003: Family and license of combined scorers

- **Question:** the shared span format lists the families `taste_model`, `human_likeness`, `audience` and
  `personal`; `tt.combine(...)` fits none of them.
- **Choice:** family `combined`; `license` = the distinct licenses of the parts, comma-joined;
  `source_data` = the parts' sources. `profile.as_scorer` uses `personal`.
- **Why:** a combination is not any one family; hiding it under one of the four would mislead analysis.
- **Affects:** readers of `hone.taste.family` (hone-lens) see a fifth value.
- **Review:** **awaiting owner review** (accept the fifth value and add it to the shared span format).

## D-004: Aggregates

- **Choice:** one table shared by `combine` and the audience panel: `weighted_mean` (alias `mean`), `min`,
  `median` (unweighted). `None` values and zero weights drop out before aggregating; nothing left gives
  `None`.
- **Review:** not needed.

## D-005: Doctests run in the default suite

- **Choice:** `--doctest-modules` with `src` in `testpaths`, so docstring examples stay correct.
- **Review:** not needed.

## D-006: Slop lists are shipped from upstream (MIT)

- **Question:** ship the upstream lists if their license allows redistribution, otherwise our own.
- **Finding:** [sam-paech/slop-score](https://github.com/sam-paech/slop-score) states that the slop lists
  in `data/` are MIT; only its word-frequency files are Apache-2.0 / CC-BY-SA, and hone-taste does not
  use them.
- **Choice:** ship `slop_list.json` and `slop_list_trigrams.json` unchanged (upstream commit
  `289264ab2a0d`) with the MIT notice in `src/hone_taste/data/slop/NOTICE.md`;
  `slop_score(wordlists=<folder>)` loads newer upstream files from a checkout.
- **Review:** **awaiting owner review** (confirm the MIT finding for the redistributed data).

## D-007: Slop score details

- All three components are rates per 1,000 words (upstream uses per 1,000 characters for contrasts).
- Each component is scaled by its own reference maximum and clamped, then weighted 60 / 25 / 15:
  `value = 1 - sum(w_c * min(1, rate_c / max_c))`. The maxima (80 words, 9 contrasts, 2.5 trigrams per
  1,000 words) are about twice the worst model average on the upstream leaderboard, because single short
  texts vary much more than averages.
- Trigrams are matched on content words (stopwords removed), which is how the upstream lists are written
  ("took deep breath").
- Contrast patterns: three surface regexes simplified from upstream stage 1; the part-of-speech stage 2
  is not ported.
- Presets add words and phrases only; weights are the same for every domain (no evidence for other
  weights). A word inside a matched phrase is counted once.
- `confidence = min(1, words / 100)`; empty text gives `Score(None, error="no words to score")`.
- Hits name their kind (`word`, `phrase`, `pattern`, `trigram`), each with `span` (and `match` for
  phrases and patterns).
- **Review:** not needed.

## D-008: Patterns scorer

- **Choice:** literals match whole phrases case-insensitively (`(?<!\w)literal(?!\w)`); compiled regexes
  are used as given. `max_hits` defaults to 5. Hits report the literal or regex source as `pattern`.
- **Review:** not needed.

## D-009: Audience panel shape

- **Question:** batching, anchors, persona files and the "same family" check were left open.
- **Choice:**
  - One `decide()` call per persona with one `score` question named `rating`
    (`"You are <persona>.\n<question>"`, `scale`, `anchors`). No batched mode: every client can take one
    question, and batching all personas into one prompt would let them bleed into each other.
  - `anchors=` keyword (default: low "not at all", middle "somewhat", high "absolutely").
  - Persona files are TOML with `personas = [...]`.
  - `client` may be a `DecisionClient` or a `TextClient`; a `TextClient` is wrapped in the public
    `TextDecisionClient` (one JSON-schema completion per `decide`, `calibrated=False`).
  - The "same family" check takes `generator_model=`; the family is the leading letters of the model name
    without provider prefix (`gemma4-12b:latest` gives `gemma`), compared with the client's `model` /
    `model_id` attribute. A match raises a `UserWarning` and adds `details["warning"]`.
  - Text is the state; an image path is attached via `images=[path]`. Audio is not accepted directly:
    pass a text description (lyrics, SongEval scores).
  - `disagreement` = population standard deviation of the persona values; `None` with fewer than two.
- **Review:** not needed.

## D-010: Adapters shipped

- **Choice:** `hone_taste.adapters.openai.OpenAITextClient` (extra `openai`): a `TextClient` over the
  OpenAI SDK (`chat.completions`, `response_format=json_schema`; strips `<think>` blocks and code fences
  before parsing). No extra for hone-models: its clients and GPU lease fit the ports structurally, so
  users pass its objects directly.
- **Affects:** hone-models' clients should pass `hone_taste.testing.contracts` (they do, in the
  cross-package tests).
- **Review:** not needed.

## D-011: Profile fit is pure Python

- **Question:** Bradley-Terry / logistic regression in pure Python or with NumPy?
- **Choice:** pure Python (`fitting.py`): one example per pick with features
  `(s_k(winner) - s_k(loser)) / spread_k` (spread = population standard deviation of scorer k over all
  scored items, so the L2 penalty treats scorers alike), no intercept, L2 = 1.0, 2,000 full-batch gradient
  steps. Coefficients are mapped back to the raw scale, negatives clipped to 0 and normalized to sum 1,
  so the result drives `combine`'s weighted mean. A missing value gives that feature 0 (no evidence).
  When no scorer has a positive coefficient: equal weights with `note = "no scorer predicts the picks"`.
  Fit quality = accuracy (ties count half) and mean log loss on the picks.
- **Why:** a handful of scorers and picks; no new core dependency; deterministic.
- **Review:** not needed.

## D-012: Profile storage and ask

- **Choice:** `${HONE_HOME:-.hone}/taste/profiles/<name>.json`, written atomically (temp file and rename)
  on every change; names limited to `[A-Za-z0-9_.-]+`. Items must be JSON-compatible (paths stored as
  strings) and are identified by their sorted-keys JSON. `ask(pairs, io=...)` takes any object with
  `ask(prompt) -> str` (`PickIO`); replies `1` / `2` / `s` / `q`; three unclear replies skip the pair.
  `ConsoleIO` refuses to run when stdin is not a terminal; `hone_taste.testing.ScriptedIO` is the fake.
- **Review:** not needed.

## D-013: Ranking the questions

- **Question:** "smallest predicted margin or the largest disagreement between scorers": how exactly?
- **Choice:** predicted utility = weighted mean of the scorers (fitted weights when they cover exactly
  these scorers, else equal); pairs sorted by |margin| ascending, ties broken by the largest spread of
  per-scorer differences. Pairs with an unscored item and pairs already picked (either order) are skipped.
- **Review:** not needed.

## D-014: Agreement metrics

- **Choice:** per scorer, over pairs where both items got a value: `rate` = (agree + ties / 2) / compared;
  `correlation` = (agree - disagree) / compared (Kendall's tau over the human pairs); `compared`,
  `unscored`. Suggested weights = the profile fit (D-011) on the same score table (every item scored
  once). `AgreementReport` prints as a plain table and has `to_dict()` / `to_json()`.
- **Review:** not needed.

## D-015: Records and the `for_select` bridge

- **Choice:**
  - Every `FunctionScorer` call (all built-in scorers, `combine`, `audience`, `profile.as_scorer`) emits
    one `hone.taste.score` span (kind `internal`); nested scorers become child spans through the trace
    `ContextVar`. A user's own scorer object records nothing unless it is wrapped (`combine` and
    `for_select` only add context).
  - Attributes: the shared names (`hone.taste.scorer`, `family`, `license`, `value`, `raw`,
    `reference_set`, `details` as JSON) plus `hone.taste.confidence`, `hone.taste.reason`,
    `hone.taste.error`. A missing value is left out, never 0; an errored score sets status `error`.
  - Sink: a `ContextVar` set by `tt.recording(sink)`; default `SqliteSpanSink` at
    `${HONE_HOME:-.hone}/taste/spans.db`, opened lazily on the first span. One commit per span (spans are
    few and small; no background buffer to lose on a crash).
  - Content capture off hashes `hone.taste.details` and `hone.taste.reason`. Secrets: `sk-...` and
    `Bearer ...` patterns, plus the value of any environment variable whose name ends in `API_KEY`,
    `TOKEN`, `SECRET` or `PASSWORD`, become `***`.
  - `tt.for_select(scorer, *, field, file, prompt_field, record=True)` returns a `SelectScorer`
    (`name` = the scorer's name). With `prompt_field` it builds `{"prompt", "response"}` (field) or
    `{"prompt", "image"}` (file). It adds `hone.scorer` and `hone.candidate_id` to the trace context and
    accepts an optional `trace=` keyword so a caller can link hone-taste's spans to its trace;
    `record=False` routes the spans to `NullSink`. A candidate missing the field or file gives an error
    score.
- **Review:** not needed.

## D-016: Taste-model wrappers, registry and license gate

- **Question:** how are heavy models wrapped, injected, gated and normalized?
- **Choice:**
  - One seam: a backend is any object with `load()`, `predict(input) -> {name: raw}`, `unload()`
    (`TasteModel`). `scorers/model.py:model_scorer` wraps it: the first call loads it, every call runs
    inside `gpu.lease(name, vram_gb, trace=...)` (default `NullGpuLease`), `close()` unloads and the next
    call loads again. The loaded model is cached in the scorer object (create a scorer once and reuse it;
    two scorers of the same model hold two copies). Every factory takes `model=` so tests and users inject
    `testing.FakeTasteModel`; `testing.FakeGpuLease` logs lease and release into the same `events` list.
  - Real backends (`SongEvalModel`, `RewardModel`, `PickScoreModel`, `HPSv3Model`, `AudioboxModel`,
    `BinocularsModel`) import torch and friends only in `load()` and are excluded from default-suite
    coverage; the real-model suite exercises them. A factory without `model=` checks that its extra is
    installed and raises `MissingExtra` naming the extra.
  - Registry `data/models.toml` (`tt.models()`, `tt.model_info()`): model id, URL, license,
    `license_clear`, whose ratings (`source_data`), VRAM estimate, extra. Two models have unclear terms
    and raise `LicenseNotAccepted` (a `MissingExtra`) unless `accept_license=True`: **PickScore** (no
    license on the model card) and **SongEval** (README: CC-BY-NC-SA-4.0 for the project; repository
    metadata: Apache-2.0; its MuQ encoder is CC-BY-NC-4.0, non-commercial).
  - SongEval is not pip-installable: its small scoring head (standard attention and linear layers, needed
    to load the checkpoint) is re-implemented with a source notice; the checkpoint is downloaded once from
    the upstream repository to `${HONE_HOME:-.hone}/taste/models/songeval/` (atomically); MuQ comes from
    the `muq` package. Dimensions 1-5 map to 0-1; value = weighted mean (equal by default).
  - Audiobox: axes 0-10 map to 0-1; default value = mean of CE, CU, PQ (production complexity is not
    better-when-higher; weights configurable).
  - `reward_model` / `image_preference`: raw values map to 0-1 through a reference set
    (`references/text/reward_model.json`, `references/images/<kind>.json`) or `reference=`; a
    non-default `model_id` must bring its own reference. HPSv3 needs the separate `hpsv3` package (it pins
    transformers 4.45 and needs about 18 GB of VRAM), so it is not in an extra; its backend is untested
    on the 8 GB development GPU.
  - Binoculars: default pair Qwen2.5-0.5B and -Instruct (Apache-2.0, fits 8 GB; the Falcon-7B pair does
    not). `p(machine) = sigmoid((threshold - score) / 0.05)`, `value = 1 - p`; the default threshold is
    upstream's low-false-positive one (calibrated for Falcon, only approximate here; documented and
    configurable). Family `human_likeness`; `confidence = tokens / 128` (capped).
  - Extras: `text`, `detect` = torch + transformers; `images` adds pillow; `songs` = torch, librosa, muq,
    safetensors, audiobox-aesthetics.
- **Affects:** a GPU lease from hone-models is passed as `gpu=` and passes `contracts.check_gpu_lease`.
- **Review:** accepted by the owner (2026-09-29): publish with the licence gate. SongEval and PickScore
  stay behind `accept_license=True`, no weights are shipped, the README warns prominently that users must
  check the upstream terms themselves, and `THIRD_PARTY_NOTICES.md` records the sources and licences.

## D-017: CLI, reference sets, real-model setup

- **CLI** (`hone-taste`, extra `cli`, typer): `score --scorer slop|binoculars|songeval|audiobox FILES`
  (`--domain`, `--accept-license`, `--json`), `profile ask --name --items DIR [--scorer] [--budget]`
  (items = `.txt` / `.md` files; refuses without a terminal), `agreement --name [--scorer ...] [--json]`
  (the profile's picks are the human pairs), `models` (registry and license table). Exit codes 0 / 1
  (some file could not be scored) / 2 (usage or configuration error). `reward_model` and
  `image_preference` need prompt and response / image pairs, so they are Python-only.
- **Reference sets:** `references/text/reward_model.json` is generated by `scripts/make_reference_sets.py`
  (12 hand-written prompts, one good and one bad answer each, raw rewards of the default model). No
  packaged set for PickScore or HPSv3 yet (their weights were not downloaded: about 4 GB, and about
  18 GB of VRAM); `image_preference` needs `reference=` until one is added.
- **Dependencies:** extra `songs` pins `transformers<5` because MuQ breaks on transformers 5
  (`config._attn_implementation` on its EasyDict config); `text` / `detect` / `images` need
  `transformers>=4.56` (`dtype=` keyword, Qwen3). Audiobox audio is read with `soundfile` and passed as a
  tensor, because torchaudio 2.9+ needs torchcodec to load files. `[tool.uv] constraint-dependencies`
  pins torch in the development lock only; users get any torch 2.2+.
- **Real-model tests** (`tests/gpu`, run through `scripts/gpu-lock.sh`): AC-18 (panel through the
  OpenAI adapter to a local Ollama `/v1` server, timeout 900 s: about 30 s per decision on a shared 8 GB
  GPU), AC-19 (reward model, SongEval), plus Audiobox and Binoculars. The Ollama model is unloaded after
  each test so the torch tests get the GPU. The SongEval checkpoint is kept in the ignored `.hone/`
  between runs.
- **Review:** not needed.

## D-018: Final review follow-ups

- `for_select(scorer, file=...)` raises `ConfigError` when the scorer takes no file kind (`audio`,
  `image`, `image_with_prompt`); otherwise a text scorer would score the path string. The bridge also
  exposes `version` (hone-taste's version), which callers can record as the scorer version.
- Taste-model `details` carry `model_id`, `license` and `source_data` (whose taste, in `details`).
  `hone.taste.raw` holds several named raw values (SongEval, Audiobox) as JSON.
- A user-chosen `reward_model(model_id=...)` or Binoculars pair is **not** license-gated: the user picked
  it and the registry cannot know its terms; `license` says "see the model card(s) of ...", `model_id`
  names the models actually run.
- SongEval's head is loaded strictly (every checkpoint key must match): a changed upstream checkpoint
  fails loudly instead of scoring with random weights. `THIRD_PARTY_NOTICES.md` records the upstream
  license statements.
- The `hone-taste` command prints "install hone-taste[cli]" instead of a traceback when typer is missing.
- **Review:** not needed.

## D-019: Cleanup after the first complete build

- Dropped the unused `respx` dev dependency (the OpenAI adapter tests use the SDK's own mock transport),
  an empty `tests/fixtures/http/`, unused pytest markers and template comments in `pyproject.toml`.
- **Review:** not needed.

## D-020: The examples set

- **Question:** the shape of the examples, which ones beyond the required set, and how their test checks
  them.
- **Choice:**
  - One file per public concept, 14 in all: the required set (see `current.md`, section 11) plus
    `custom_scorer.py` (the `Scorer` protocol), `bring_your_own_client.py` (own `TextClient`,
    `OpenAITextClient` over an SDK-shaped fake), `model_registry.py`, `records_and_tracing.py` and
    `command_line.py`.
  - Plain module-level code, top to bottom (no `main()`), so a reader can copy any part. The docstring has
    `What:`, `How:` (numbered steps matching `# 1.` ... comments in the code), `Why:` (with pitfalls) and
    a `Run:` line. Examples assert the facts they show, so a behaviour change breaks them.
  - Only the public API (`hone_taste`, `hone_taste.testing`, `.errors`, `.normalize`,
    `.adapters.openai`); no private names (checked on the syntax tree by AC-20).
  - AC-20 runs each file as `python examples/<file>.py` in a subprocess, in an empty folder with its own
    `HONE_HOME`; examples that write files use a temporary folder so reruns start clean.
  - `command_line.py` runs `python -m hone_taste.cli`, so it needs no console script on `PATH`; it needs
    the `cli` extra.
  - Examples are linted by ruff (with `S101`, `S603` and the pytest-style `PT` rules off) and
    type-checked by pyright (basic mode): `Score.value` is `float | None`, so the examples show the
    explicit `is not None` check that copied code needs anyway.
- **Found while writing them:** the license-gate message said "license is unclear: unclear: ..." for
  SongEval, and `AgreementReport` columns did not line up; both fixed.
- **Review:** not needed.

## D-021: Trace context carries the finding id

- **Choice:** spans copy `hone.lens.finding_id` from the trace context, like the other shared context
  keys, so scores made during a hone-lens replay can be traced back to the finding.
- **Review:** not needed.

## D-022: Concurrent writers creating one span store

- **Choice:** `SqliteSpanSink` sets `busy_timeout` before switching to WAL and retries the switch briefly.
  When several processes created the same fresh store at once, the switch ignored the timeout and a
  writer dropped its spans ("database is locked").
- **Review:** not needed.

## D-023: Pairwise panels: one subclass, alternating order

- **Question:** where does `panel.pairwise(a, b)` ([0002](changes/0002-audience-panels-that-compare.md))
  live, and how is the order of the two candidates chosen?
- **Choice:** `audience()` returns `AudiencePanel`, a `FunctionScorer` subclass with one extra field
  (the comparison function) and the `pairwise` method; ratings are unchanged. Spans for ratings and
  comparisons are written by one helper (`types.recorded`). Persona `i` sees `b` first when `i` is odd:
  deterministic, no random seed, and a judge that always picks the first shown ends in a tie across two
  personas. `for_select_pairwise` raises on a failed comparison, which hone-select records as a tie with
  the error.
- **Why:** callers asked for `panel.pairwise`; a subclass of depth 1 keeps `audience()`'s return value a
  drop-in `FunctionScorer`. Alternating instead of shuffling keeps runs reproducible.
- **Review:** not needed.

## D-024: Character consistency is a taste-model scorer over an embedder

- **Question:** how does `character_consistency` ([0003](changes/0003-character-consistency.md)) fit the
  existing scorer shapes, and which family does it record?
- **Choice:** a small image-embedder backend (`load`, `embed(path, part)`, `unload`) wrapped into the
  taste-model machinery (`model_scorer`: lazy load, `GpuLease`, `close()`), so nothing new handles
  loading. The reference is embedded once, on the first call, and kept across `close()` (vectors are
  small and the model does not change). Family `taste_model`: the shared span format has no better value
  and a new one would need the same review as D-003; its `source_data` says it learned from no ratings.
  Cosine is pure Python (vectors of a few hundred numbers; no numpy in the core).
- **Review:** not needed.

## D-025: Style match without a style-sheet judge

- **Question:** which parts of persona-writer's style scorers ([0004](changes/0004-style-match-scorer.md))
  does the package take?
- **Choice:** stylometry as persona-writer wrote it (same features, floors and formula; the function
  words are a one-line table), the excerpt-anchored judge, `combine` for `style_match`, and the
  calibration. Not the style-sheet judge: it is the one calibration showed to reward imitations, and a
  package default would invite that mistake. Excerpts are picked without a model (prose paragraphs of
  30-150 words, round robin over the texts) so no build step is needed; `style_judge` has family
  `audience` (an LLM judging), `stylometry` and `style_match` `personal`. `calibrate_style` weights by
  clipped correlation (as persona-writer does) rather than by the Bradley-Terry fit, which rewards the
  scorer that separates the pairs most, not the one that agrees most often.
- **Review:** not needed.

## D-026: Song quality: retry inside the backend wrapper, checks as a model-free scorer

- **Question:** how do SongEval's excerpts and the audio checks ([0005](changes/0005-song-quality-scorers.md))
  fit the existing shapes without breaking callers?
- **Choice:** `max_seconds` defaults to `None`: the whole song, scored once, exactly as before (apps
  that excerpt and retry themselves, like OneShotStudio today, must see the old behaviour; an automatic
  retry broke its test). With `max_seconds`, the out-of-memory retry halves the excerpt; no `on_oom`
  switch (one strategy). The retry lives in a small backend wrapper, so it runs inside the same GPU
  lease; the backend receives `max_seconds` only when it is set, so existing backends keep working. `audio_checks` records
  family `taste_model` like the other audio scorers, with `source_data` saying it uses no model; its
  measurement module (the app's code, split so each module stays under 300 lines) is imported lazily,
  so the core still imports without numpy, and relaxes pyright's unknown-type checks because scipy
  and librosa ship no types.
- **Review:** not needed.
