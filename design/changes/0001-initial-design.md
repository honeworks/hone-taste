# 0001: Initial design

## Status

`implemented in 0.1.0`

Three decisions made while building it are **awaiting owner review** and block publishing: the licensing
of the SongEval, MuQ and PickScore wrappers and the re-implemented SongEval head (D-016), the
redistributed slop lists (D-006), and the extra `combined` scorer family (D-003). See
[`../decisions.md`](../decisions.md#awaiting-owner-review).

## Context

The need came from a local song-making pipeline (ideas, lyrics, music, videos) that already chose the best
of several generated drafts at each step. Its automatic scores measured correctness: the sung words are
audible, the audio is clean, banned words are absent. None of them said whether a song was catchy or a
lyric sounded like a person wrote it, and asking the author to rate every draft did not scale.

A research pass ([`../history/0000-research.md`](../history/0000-research.md)) found good but scattered
building blocks: taste models trained on human ratings (SongEval, Audiobox Aesthetics, Skywork-Reward-V2,
HPSv3, PickScore), the Slop Score word lists, zero-shot AI-text detectors (Binoculars), persona simulation
(TinyTroupe) and few-shot preference learning. Each had its own code, inputs and score range, and few
stated whose taste they encode or under which license.

## Problem

Provide stand-ins for human judgment that:

- answer "would people like this?" and "does this feel human-made?" for text, songs and images;
- return one comparable score shape, so they combine and plug into any selection step;
- are honest about whose taste they encode, their license and their failure modes;
- adapt to one person with a handful of picks instead of a hundred;
- stay light: usable without a GPU or torch, with heavy models optional;
- never confuse "could not score" with a bad score.

## Options

| Question | Options considered | Chosen |
|---|---|---|
| Relation to selection engines | a plug-in that imports a selection package; a standalone scorer pack that matches a common candidate-scorer shape | standalone; `for_select` produces the shape without importing anything |
| Taste models | train our own; wrap published models behind one backend seam | wrap published models (training is a non-goal) |
| Detectors | hard gates; signals | signals only (known false positives on formal and non-native writing) |
| Personal taste | few-shot personal reward models (FSPO-style); examples in the judge prompt plus re-weighting the other scorers from pairwise picks | examples in prompts plus re-weighting; reward models later |
| Choosing which picks to ask for | random pairs; closest predicted margin / largest scorer disagreement | closest margin, then disagreement (active questioning) |
| Persona judges | batched in one prompt; one call per persona | one call per persona, so personas do not bleed into each other |
| Heavy models | import at package load; lazy load inside an injected GPU lease with `close()` | lazy, leased, closable, injectable |
| Models with unclear license terms | leave them out; ship them silently; ship them behind an explicit `accept_license=True` gate | the gate, with the license and whose taste in the registry and README |
| Slop word lists | derive our own; ship upstream if its license allows redistribution | upstream (MIT), unchanged, with the notice |
| Profile fitting | NumPy / scikit-learn; pure Python | pure Python (a handful of scorers and picks; no new core dependency) |
| Media in v0.1 | text and songs only; text, songs and images | text, songs and images |
| Where profiles live | inside a workflow runner's metadata; a local JSON file per person | a local JSON file under `${HONE_HOME:-.hone}` |

## Decision

Build `hone-taste` as described in [`../current.md`](../current.md):

- four families (taste models, human-likeness, audience panels, personal taste) behind one `Score` and one
  `Scorer` shape, with typed input kinds and `None` for "could not score";
- taste models as thin scorers over a `load` / `predict` / `unload` backend, lazy and GPU-leased, with a
  model registry that records license and source data and gates unclear licenses;
- the Slop Score port with domain presets, a pattern scorer and the Binoculars detector;
- an audience panel over a `DecisionClient` or `TextClient`, with anchors, aggregates, disagreement and
  a same-family warning;
- personal profiles with picks, a pure-Python Bradley-Terry fit, active questions and a terminal prompt;
- combining, reference-set normalization and a small agreement report against human picks;
- `for_select` for the candidate-scorer shape; `hone.taste.score` spans in the shared span format;
- ports with public fakes and contract checkers; a CLI; docs and examples executed by the tests.

The smaller choices made while building it are in [`../decisions.md`](../decisions.md) (D-001 to D-022).

## Consequences

- **Better:** one interface for very different judges; scores combine and explain themselves; the core
  installs without torch and runs without a GPU; licensing and "whose taste" are visible at the point of
  use; every behaviour is pinned by acceptance cases and runnable examples.
- **Costs:** taste models depend on third-party weights and packages with their own constraints (MuQ needs
  `transformers<5`; HPSv3 needs about 18 GB of VRAM and its own package); the re-implemented SongEval head
  must track upstream (it loads strictly, so a change fails loudly).
- **Harder:** publishing needs an owner decision on the gated licenses; reference sets are small and not
  calibrated (none yet for images); the Binoculars threshold is only approximate for the small default
  pair; personas remain one model pretending.

## Migration and compatibility

First release: nothing to migrate. The public API, the `Score` shape, the ports (`PORTS_VERSION = "1"`),
the span name and attributes (schema version 1), the profile file format and the CLI exit codes are the
compatibility surface from here on; later changes to them need a change record.
