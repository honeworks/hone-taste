# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/). Why the design changed is recorded in
[design/changes/](design/changes/).

## [0.1.0] - unreleased

The design of this release: [design/changes/0001-initial-design.md](design/changes/0001-initial-design.md).

### Added (song quality)
- `tt.songeval(max_seconds=...)` scores a middle excerpt of long songs and, after an out-of-memory error,
  retries on half the length (down to 30 s); `details["excerpt_s"]`. Without it nothing changes.
- `tt.audio_checks(...)` / `tt.audio_report(path)`: objective audio checks without a model (loudness,
  dynamics, true peak, clipping, harshness, sibilance, mud, glitches; tempo, key and meter against a
  request) in the `songs` extra, which now names numpy and scipy; AC-24 and `examples/song_checks.py`
  ([0005](design/changes/0005-song-quality-scorers.md), renumbered from a second 0004).

### Added (style match)
- `tt.fingerprint` / `tt.stylometry` (core, no model: surface features and Burrows' Delta),
  `tt.style_judge` (an LLM judge anchored on real excerpts only), `tt.style_match` (both, weighted) and
  `tt.calibrate_style` (real vs imitation pairs through `agreement`, weights from the correlations);
  AC-23 and `examples/style_match.py`
  ([0004](design/changes/0004-style-match-scorer.md), renumbered from a second 0003).

### Added (character consistency)
- `tt.character_consistency(reference, panels=n)`: is this image the same character as the reference
  (sheet)? DINOv2-small image embeddings (Apache-2.0, extra `images`, registry entry `dinov2_small`),
  cosine to the closest reference panel mapped from a floor; `testing.FakeEmbedder`; AC-22 and
  `examples/character_consistency.py`
  ([0003](design/changes/0003-character-consistency.md)).

### Added (comparative audience panels)
- `details["ceiling"]` on panel scores (every persona gave the top rating), `tt.spread(scores)` (did a
  scorer tell a batch of candidates apart?), `panel.pairwise(a, b)` (each persona compares two inputs,
  order alternating per persona; span attribute `hone.taste.panel.mode = "compare"`) and
  `tt.for_select_pairwise(panel, field=...)`, a hone-select pairwise judge; AC-21 and
  `examples/comparative_panel.py`
  ([0002](design/changes/0002-audience-panels-that-compare.md)).

### Changed (design docs)
- The design is documented for readers in `design/`: why the package exists, the current design with its
  acceptance cases, the initial change record, the early research and the decision log
  (`design/decisions.md`, with the licensing questions awaiting owner review). `CONTRIBUTING.md` holds
  the contributor conventions.

### Fixed (concurrent writers)
- `SqliteSpanSink` sets `busy_timeout` before switching to WAL and retries the switch briefly: when several
  processes created the same fresh store at once, the switch ignored the timeout and a writer dropped its
  spans ("could not record spans: database is locked").

### Added (examples)
- `examples/`: 14 runnable, explained examples (What / How / Why docstring), one per public concept, and
  `examples/README.md` indexing them in reading order; every example runs offline with the public fakes.
- AC-20 test: runs each example in a subprocess and checks its docstring, the index and public-API-only
  imports; README and docs link the examples.

### Fixed (examples)
- The license-gate message quoted SongEval's license as "unclear: unclear: ..."; it now names the gate and
  quotes the registry text once.
- `print(tt.agreement(...))` lines up its columns for any scorer name length.

### Changed (cross-package tracing)
- Spans copy `hone.lens.finding_id` from the trace context (a shared trace-context attribute), so
  scores made under a hone-lens replay carry the finding id.

### Removed (cleanup pass)
- Unused `respx` dev dependency, the empty `tests/fixtures/http/`, unused pytest markers and template
  placeholder comments in `pyproject.toml`, unused fixture parameters in two tests
  ([D-019](design/decisions.md#d-019-cleanup-after-the-first-complete-build)).

### Added
- `Score`, the `Scorer` protocol, input-kind checks (`accepts`), typed errors.
- `tt.normalize`: reference sets, linear (5th-95th percentile) and percentile modes.
- `tt.combine`: weighted mean / min / median with `None` renormalization.
- `tt.slop_score` (general, lyrics, email presets) with the upstream slop-score lists (MIT).
- `tt.patterns`: banned literals and regexes.
- `tt.audience`: persona panel over any `DecisionClient` or `TextClient`; `tt.TextDecisionClient`.
- Ports (`DecisionClient`, `TextClient`, `GpuLease`, `RecordSink`, `PORTS_VERSION`), `tt.testing` fakes and contract checkers.
- `hone_taste.adapters.openai.OpenAITextClient` (extra `openai`).
- `tt.profile`: personal taste profiles (picks, examples, pure-Python Bradley-Terry fit, `questions`,
  `ask` with `ConsoleIO` / `hone_taste.testing.ScriptedIO`, `as_scorer`, `prompt_examples`).
- `tt.agreement`: per-scorer agreement with human picks, Kendall-style correlation, suggested weights.
- `tt.for_select`: use any scorer as a hone-select scorer (reads `candidate.data[field]` /
  `candidate.files[file]`), with trace propagation and `record=False`.
- Records: every score writes a `hone.taste.score` span (span schema v1) to
  `${HONE_HOME:-.hone}/taste/spans.db`; `tt.recording(sink)` with `SqliteSpanSink`, `JsonlSpanSink`,
  `MemorySink`, `NullSink`; `tt.current_trace()`; content-capture switch and secret redaction.
- Taste models (lazy, `GpuLease`-wrapped, `close()`): `tt.songeval` and `tt.audiobox` (extra `songs`),
  `tt.reward_model` (extra `text`), `tt.image_preference` (PickScore / HPSv3, extra `images`); the
  `tt.binoculars` AI-text detector (extra `detect`); model registry with licenses (`tt.models()`) and the
  `accept_license=True` gate; `tt.NullGpuLease`, `testing.FakeTasteModel`, `testing.FakeGpuLease`.
- `hone-taste` CLI (extra `cli`): `score`, `profile ask`, `agreement`, `models`.
- Packaged reference set for the default reward model; `scripts/make_reference_sets.py`.
- README (quickstart, honesty rules, license table), `docs/` (guide, records, CLI, adapters) and
  `examples/` (lyric selection with fakes, hone-select bridge), all executed by the test suite.
- Real-model suite (`tests/gpu`): audience panel via Ollama `/v1`, reward model, SongEval, Audiobox,
  Binoculars.
- THIRD_PARTY_NOTICES.md (slop-score lists, SongEval head).
