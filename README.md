# hone-taste

[![CI](https://github.com/honeworks/hone-taste/actions/workflows/ci.yml/badge.svg)](https://github.com/honeworks/hone-taste/actions/workflows/ci.yml)

**Stand-in scorers for human judgment: taste models, human-likeness checks, audience panels, personal taste.**

Automatic metrics tell you whether an output is *correct*; `hone-taste` estimates whether people would
*like* it and whether it *feels human-made*, for text, songs and images, without asking a person to rate
things again and again. Every scorer returns the same `Score` (0-1 `value`, `confidence`, `reason`,
per-aspect `details`, `error`), so scorers combine, plug into best-of-N selection, and can be checked
against a few of your own picks.

Part of **[honeworks](https://github.com/honeworks)**: small, standalone tools for reliable generative-AI
workflows. Works on its own; works better with its siblings.

> [!WARNING]
> **Some wrapped models have non-commercial or unclear licences.** hone-taste itself is Apache-2.0 and
> **ships no model weights**; each taste model downloads its weights from the upstream source on first use.
> Three of them need extra care, and two refuse to load unless you pass `accept_license=True`:
>
> - **SongEval** (`tt.songeval`): licence **unclear**. The upstream README says CC-BY-NC-SA-4.0
>   (non-commercial, share-alike); the repository metadata says Apache-2.0. Requires `accept_license=True`.
> - **MuQ** (the audio encoder SongEval uses): weights are **CC-BY-NC-4.0, non-commercial only**.
>   Covered by SongEval's `accept_license=True`.
> - **PickScore** (`tt.image_preference("pickscore")`): **no licence stated** on the model card.
>   Requires `accept_license=True`.
>
> Passing `accept_license=True` only records that *you* have read and accepted the upstream terms for
> your use; it grants no rights. **You are responsible for checking each model's licence** before using
> it, especially commercially. Details: [Models and licenses](#models-and-licenses) and
> [THIRD_PARTY_NOTICES.md](https://github.com/honeworks/hone-taste/blob/main/THIRD_PARTY_NOTICES.md).

**Status:** alpha, version 0.1.0. The API can still change between minor versions.

## Install

```bash
pip install hone-taste                 # core: slop score, patterns, panels, profiles (no GPU, no torch)
uv add hone-taste                      # the same with uv
pip install "hone-taste[openai,cli]"   # with extras (table below)
```

Until the package is on PyPI, install it from GitHub:

```bash
pip install "git+https://github.com/honeworks/hone-taste"
pip install "hone-taste[songs] @ git+https://github.com/honeworks/hone-taste"   # with an extra
```

Python 3.11 or newer. The core depends only on pydantic.

| Extra | Adds | Pulls in |
|---|---|---|
| (none) | `slop_score`, `patterns`, `audience`, `profile`, `stylometry`, `style_match`, `combine`, `agreement` | pydantic |
| `openai` | `OpenAITextClient` for audience panels on any OpenAI-compatible server (Ollama `/v1`, vLLM ...) | openai |
| `cli` | the `hone-taste` command | typer, rich |
| `text` | `reward_model` (Skywork-Reward-V2) | torch, transformers |
| `detect` | `binoculars` AI-text detector | torch, transformers |
| `images` | `image_preference` (PickScore), `character_consistency` (DINOv2) | torch, transformers, pillow |
| `songs` | `songeval`, `audiobox`, `audio_checks` | torch, transformers, librosa, muq, audiobox-aesthetics, soundfile, scipy |

The torch-based extras download model weights on first use; a GPU helps but small models run on a CPU.

## Quickstart
```python
import hone_taste as tt

slop = tt.slop_score(domain="lyrics")
cliched = slop("It's not just a song, but a journey through the tapestry of time.")
fresh = slop("Dad's work boots by the door still smell like diesel and rain.")
print(round(cliched.value, 2), round(fresh.value, 2))  # the cliched line scores lower
print(cliched.details["hits"][0])  # what was found, with character spans

banned = tt.patterns(["neon", "echoes of"])
mine = tt.combine({"slop": 2, "banned": 1}, {"slop": slop, "banned": banned})
print(round(mine("Neon lights and echoes of a dream").value, 2))
```

Every call is recorded as a `hone.taste.score` span in `.hone/taste/spans.db` (set `HONE_HOME` to move it,
or wrap calls in `with tt.recording(tt.NullSink()):` to turn recording off).

## The four families

| Family | Scorers | Needs |
|---|---|---|
| A. Taste models: small models trained on many human ratings | `tt.songeval`, `tt.audiobox`, `tt.reward_model`, `tt.image_preference`; `tt.character_consistency` (same character as a reference?); `tt.audio_checks` (objective checks, no model) | an extra; a GPU helps |
| B. Human-likeness | `tt.slop_score`, `tt.patterns`, `tt.binoculars` | nothing (binoculars: extra `detect`) |
| C. Audience panel: an LLM plays target personas | `tt.audience(personas, question, client)` | any `DecisionClient` or `TextClient` |
| D. Personal taste: learn one person's taste from a few picks, or an author's style from their texts | `tt.profile(name)`; `tt.stylometry`, `tt.style_match`, `tt.calibrate_style` | 5-10 picks; about ten texts |

## Documentation

| Where | What |
|---|---|
| [docs/guide.md](https://github.com/honeworks/hone-taste/blob/main/docs/guide.md) | each scorer family, with runnable snippets |
| [docs/records.md](https://github.com/honeworks/hone-taste/blob/main/docs/records.md) | what every call records (`hone.taste.score` spans) |
| [docs/cli.md](https://github.com/honeworks/hone-taste/blob/main/docs/cli.md) | the `hone-taste` command line |
| [docs/adapters.md](https://github.com/honeworks/hone-taste/blob/main/docs/adapters.md) | bringing your own model client or GPU lease |
| [examples/](https://github.com/honeworks/hone-taste/blob/main/examples/README.md) | one runnable, explained example per concept; start with `examples/quickstart.py` |

The examples and doc snippets run offline with the package's fakes (no network, GPU or downloads), and
the test suite executes all of them. Real use of an audience panel needs an LLM, for example a local
[Ollama](https://ollama.com) server with a pulled model such as `gemma3:12b` and the `openai` extra.

## Honesty rules
- **Whose taste?** Each taste model learned from a specific group of raters (table below). Their taste is
  not yours; check with `tt.agreement(...)` against a few of your own picks before trusting a scorer.
- **Detectors are signals, never gates.** `binoculars` and `slop_score` have known false positives on
  formal, templated and non-native writing. Use them to rank or flag, not to reject.
- **Personas are one LLM pretending.** A panel is a cheap guess at an audience, not an audience. Weight it
  lower until it agrees with real picks. The panel warns when its model family matches the generator's,
  flags a ceiling (every persona gave the top rating) and can compare two candidates instead of rating
  one (`panel.pairwise(a, b)`), which tells candidates apart when ratings saturate.
- **Style judges read real text, not descriptions.** A judge given a style description rewards
  imitations of the description (in one calibration it preferred ghostwritten imitations over the real
  author every time). `tt.style_judge` sees only real excerpts; check it with `tt.calibrate_style` on
  held-out real texts before trusting it.
- **Goodhart.** Optimizing hard against one taste model finds outputs that fool it. Combine families
  (`tt.combine`) and re-check with real people now and then.

## Models and licenses

Checked 2026-09-27; `hone-taste models` prints the same table from `hone_taste/data/models.toml`. Licences
change: check the upstream model card before you rely on this table. No weights are included in the
package.

| Scorer | Model | Learned from | License | Gate |
|---|---|---|---|---|
| `songeval` | ASLP-lab/SongEval + MuQ encoder | 2,399 full-length songs rated on five aesthetic dimensions by 16 annotators with music backgrounds | unclear: README says CC-BY-NC-SA-4.0, repo metadata Apache-2.0; MuQ weights CC-BY-NC-4.0 (non-commercial) | `accept_license=True` |
| `audiobox` | facebook/audiobox-aesthetics | speech, music and sound clips rated on four axes by trained raters | CC-BY-4.0 | - |
| `reward_model` | Skywork/Skywork-Reward-V2-Qwen3-0.6B | Skywork-SynPref-40M preference pairs (human-verified, AI-assisted curation) | Apache-2.0 | - |
| `image_preference("pickscore")` | yuvalkirstain/PickScore_v1 | Pick-a-Pic: web-app users' picks between generated images for their own prompts | not stated on the model card | `accept_license=True` |
| `image_preference("hpsv3")` | MizzenAI/HPSv3 | HPDv3: about 1.17M human pairwise comparisons | Apache-2.0 weights, MIT code | - (needs `pip install hpsv3`, ~18 GB VRAM) |
| `character_consistency` | facebook/dinov2-small | no human ratings: self-supervised image features | Apache-2.0 | - |
| `binoculars` | Qwen/Qwen2.5-0.5B + -Instruct | no human ratings: perplexity ratio of two related models | Apache-2.0 models, BSD-3-Clause method | - |

`image_preference` has no packaged reference set yet: pass `reference=` (a JSON file with the raw scores of
a few good and bad images, see `hone_taste.normalize`) so raw scores can be mapped to 0-1.

## Use it with the rest of honeworks
- **hone-select**: `tt.for_select(scorer, field="lyrics")` or `tt.for_select(scorer, file="audio")` turns
  any scorer into a hone-select scorer (it reads `candidate.data[field]` / `candidate.files[file]`); no
  import of hone-select is needed. `tt.for_select_pairwise(panel, field="idea")` makes an audience panel
  a pairwise judge.
- **hone-models**: its decision / text clients and GPU lease fit the `DecisionClient`, `TextClient` and
  `GpuLease` ports directly: `tt.audience(..., client=hone_models.decision(...))`,
  `tt.songeval(gpu=<hone-models GPU lease>)`.
- **hone-lens** reads the `hone.taste.score` spans; trace ids flow in through `trace=` / the context.

## Design
[design/](https://github.com/honeworks/hone-taste/blob/main/design/README.md) explains why hone-taste exists, how it is designed today
([design/current.md](https://github.com/honeworks/hone-taste/blob/main/design/current.md)) and why it changed ([design/changes/](https://github.com/honeworks/hone-taste/tree/main/design/changes/)).
Why SongEval and PickScore are gated rather than left out is recorded in
[design/decisions.md](https://github.com/honeworks/hone-taste/blob/main/design/decisions.md#d-016-taste-model-wrappers-registry-and-license-gate).
Contributions: [CONTRIBUTING.md](https://github.com/honeworks/hone-taste/blob/main/CONTRIBUTING.md).

## How this was built
hone-taste was specified by a human and built by AI coding agents (Claude) working against written
specifications and acceptance tests; a human reviewed the decisions they made, and commits written with
AI carry a `Co-Authored-By` line. Every design change, with what was found, what was decided and why, is
in [design/changes/](https://github.com/honeworks/hone-taste/tree/main/design/changes/); the smaller implementation choices,
including those still awaiting the owner's review, are in [design/decisions.md](https://github.com/honeworks/hone-taste/blob/main/design/decisions.md).

## License
Apache-2.0. Copyright 2026 Bahman Shadmehr. Third-party data and code notices:
[THIRD_PARTY_NOTICES.md](https://github.com/honeworks/hone-taste/blob/main/THIRD_PARTY_NOTICES.md).
