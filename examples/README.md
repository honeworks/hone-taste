# Examples

Each file shows one thing hone-taste does, the way we recommend doing it. It opens with a short
**What / How / Why** docstring, then runs top to bottom with the package's fakes: no network, no GPU, no
model downloads. Every example prints what it shows and `assert`s the key facts, and the test suite runs
all of them (`tests/e2e/test_ac20_examples.py`), so they stay correct. Like any use of hone-taste, running
them records spans under `${HONE_HOME:-.hone}/taste/` (`.hone/` is git-ignored). `command_line.py` also
needs the `cli` extra (`uv sync --all-extras` installs it).

```bash
uv run python examples/quickstart.py          # any example, from the repository root
```

Read them in this order; later ones assume the earlier ideas. The last column links to the matching
section of the design ([`design/current.md`](../design/current.md)).

## Basics

| File | Concept | What it shows | Design |
|---|---|---|---|
| [`quickstart.py`](quickstart.py) | `Score`, calling a scorer | A scorer is a callable returning a `Score`; `value is None` (with `error`) means "could not score", never 0. | [§2](../design/current.md#2-concepts) |
| [`slop_and_patterns.py`](slop_and_patterns.py) | `slop_score`, `patterns` | Domain presets (general, lyrics, email), hits with spans, your own word lists and banned regexes. | [§4.1](../design/current.md#41-slop-score-core), [§4.2](../design/current.md#42-patterns-core) |
| [`combine_and_normalize.py`](combine_and_normalize.py) | `combine`, `normalize` | Weighted combination where missing values drop out; reference sets, linear and percentile mapping. | [§4.7](../design/current.md#47-normalization-and-reference-sets), [§4.8](../design/current.md#48-combining) |
| [`custom_scorer.py`](custom_scorer.py) | the `Scorer` protocol | Write your own scorer as a small dataclass and combine it with the built-in ones. | [§2](../design/current.md#2-concepts) |

## Audience panels

| File | Concept | What it shows | Design |
|---|---|---|---|
| [`audience_panel.py`](audience_panel.py) | `audience` over a `DecisionClient` | Personas, scale and anchors, aggregates, disagreement, a failing persona, image inputs. | [§4.5](../design/current.md#45-audience-panel-core-needs-a-client) |
| [`comparative_panel.py`](comparative_panel.py) | `panel.pairwise`, `spread`, `for_select_pairwise` | When every candidate gets the top rating: the ceiling flag, `tt.spread`, comparing two candidates (order-swapped per persona) and the panel as a hone-select pairwise judge. | [§4.5](../design/current.md#45-audience-panel-core-needs-a-client) |
| [`bring_your_own_client.py`](bring_your_own_client.py) | `TextClient`, `OpenAITextClient`, contracts | Your own client checked by the contract checkers; the OpenAI-compatible adapter over an SDK-shaped fake; the same-family warning. | [§6](../design/current.md#6-ports) |

## Personal taste

| File | Concept | What it shows | Design |
|---|---|---|---|
| [`personal_profile.py`](personal_profile.py) | `profile`: picks, `fit`, `questions`, `ask` | Learn one person's weights from a few picks, ask only about the closest calls, rank with `as_scorer`. | [§4.6](../design/current.md#46-personal-profiles-core) |
| [`style_match.py`](style_match.py) | `stylometry`, `style_judge`, `style_match`, `calibrate_style` | Does a draft read like the author? Fingerprint their texts, anchor the judge on real excerpts, and calibrate on held-out real texts vs imitations (a dazzled judge gets weight 0). | [§4.6.1](../design/current.md#461-style-match-core-the-judge-needs-a-client) |
| [`agreement.py`](agreement.py) | `agreement` | Check each scorer against human picks: agreement rate, correlation, unscored pairs, suggested weights. | [§4.9](../design/current.md#49-agreement-check) |

## Taste models

| File | Concept | What it shows | Design |
|---|---|---|---|
| [`taste_models.py`](taste_models.py) | taste-model wrappers | Lazy loading inside a `GpuLease`, `close()`, the license gate (`accept_license=True`), with `FakeTasteModel`. | [§4.3](../design/current.md#43-taste-models-extras-lazy) |
| [`character_consistency.py`](character_consistency.py) | `character_consistency` | Keep the candidate that shows the same character as a reference sheet (split into panels), with `FakeEmbedder`; as a hone-select file scorer. | [§4.3](../design/current.md#character-consistency) |
| [`song_checks.py`](song_checks.py) | `audio_checks`, `audio_report`, `songeval(max_seconds=...)` | Objective audio checks on two synthesised songs (clean vs overdriven); SongEval retrying on shorter excerpts after an out-of-memory error. Needs the `songs` extra. | [§4.3](../design/current.md#song-quality-checks) |
| [`model_registry.py`](model_registry.py) | `models()`, `model_info()` | Whose ratings each model learned, its license and gate, the extra to install and the VRAM to lease. | [§4.3](../design/current.md#registry-and-license-gate) |

## Integration

| File | Concept | What it shows | Design |
|---|---|---|---|
| [`select_bridge.py`](select_bridge.py) | `for_select` | Use any scorer in hone-select through its candidate shape (`data`, `files`, prompt pairs) without importing it. | [§5](../design/current.md#5-the-candidate-scorer-shape) |
| [`records_and_tracing.py`](records_and_tracing.py) | spans, sinks, trace context | `hone.taste.score` spans, joining a caller's trace, child spans, content capture, secret redaction, file sinks. | [§7](../design/current.md#7-records) |
| [`command_line.py`](command_line.py) | the `hone-taste` CLI | `score`, `profile ask` (and why it needs a terminal), `agreement`, `models`, exit codes. | [§8](../design/current.md#8-command-line-extra-cli) |
| [`lyrics_taste.py`](lyrics_taste.py) | everything together | A song pipeline's lyric selection: slop + a listener panel + a few picks, recorded to SQLite. | [§1](../design/current.md#1-purpose) |

To run a panel against a real model, replace the fake client with
`hone_taste.adapters.openai.OpenAITextClient(...)` (see `bring_your_own_client.py`); real taste models need
their extra and run on a GPU (see the README's model table). Real-model tests live in `tests/gpu/`.
