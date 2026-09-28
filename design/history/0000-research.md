# 0000: Research: stand-ins for human judgment

*Written September 2026, before any code existed. Rewritten for readers; the findings are unchanged.
What happened to each idea is noted at the end.*

## Names used at the time

The research used working names that were replaced before the first release:

| Then | Now |
|---|---|
| `tastekit` (`import tastekit as tk`) | `hone-taste` (`import hone_taste as tt`) |
| `bestofn` (generate, score, select) | `hone-select` |
| `modelkit` (model access, GPU leases) | `hone-models` |
| `flowkit` (workflow runner) | `hone-flow` |

## Starting point

The question came from a local pipeline that generates song ideas, lyrics, music and videos, and picks
the best of several drafts at each step. Its automatic scores measured correctness: the song score said
the words were audible and the audio clean, not that the song was catchy. Asking a person to rate every
draft does not scale, and asking for a hundred pairwise picks to learn someone's taste is too much.

The pitch that came out of it: *"Human judgment without the humans: predicted taste, human-likeness and
audience reactions for text, songs and images, tuned to you with a handful of picks."*

## What exists

### A. Taste models: predicted human preference

| Medium | Model | Trained on | Output |
|---|---|---|---|
| Songs | SongEval toolkit (ASLP-lab) | 2,399 full songs (140+ h) rated by 16 professional annotators; English and Chinese, 9 genres | coherence, memorability, naturalness of vocal breathing and phrasing, clarity of structure, musicality |
| Songs (newer) | MuseCritic (2026 paper) | natural-language aesthetic critiques | multi-aspect song rewards |
| Audio | Audiobox Aesthetics (Meta) | audio ratings | content enjoyment, usefulness, production complexity and quality |
| Text | Skywork-Reward-V2 (0.6B-8B) | large curated human-preference pairs; top of seven reward-model benchmarks | a preference score for a response to a prompt |
| Text (alt) | ArmoRM and other RewardBench models | preference pairs | score (ArmoRM per aspect) |
| Images | HPSv3 | 1.08M text-image pairs, 1.17M pairwise comparisons | preference score |
| Images (alt) | PickScore, ImageReward, HPSv2 | Pick-a-Pic user choices; expert comparisons; annotations | preference score |

Text reward models score *a response to a prompt* (for lyrics, the prompt is the song brief). Raw outputs
live on different scales, so each needs mapping to 0-1 against a reference set.

### B. Human-likeness checks

| Check | How it works | Strengths and limits |
|---|---|---|
| Slop Score | frequency of words, "not X but Y" contrasts and trigrams overused by LLMs compared with human text, weighted 60 / 25 / 15 | fast, local, explainable (lists the offending phrases); tuned for creative writing and essays |
| Binoculars | zero-shot detector contrasting two related LLMs; over 90% detection at 0.01% false positives in its paper | no training; needs a GPU; weaker on very short text |
| Fast-DetectGPT | zero-shot detector via conditional probability curvature | fast; precise on news-style text |
| Custom pattern lists | the user's own banned phrases and regexes | domain-specific, trivially explainable |

Detectors misjudge some human writing (very formal text, non-native writers), so they should be signals,
never hard gates. A related generation-time idea, the Antislop sampler, suppresses slop by backtracking;
it is out of scope for a scoring package.

### C. Simulated audience

A language model plays several people from the target audience; each gives a rating and a short reaction.
Aggregated as a panel (mean, minimum, disagreement), it is inspired by Microsoft's TinyTroupe. The caveat:
personas are one model pretending. They share its taste and blind spots and tend to be too positive, so a
panel should count less than trained taste models and be checked against real picks. For audio or images,
personas read a description (lyrics, a transcript, taste-model scores) or use a multimodal judge.

### D. Personal taste from a few picks

Three levels, simplest first:

1. examples in the judge prompt: a few liked and disliked items;
2. re-weighting: a few pairwise picks fit weights over the other scorers (a small Bradley-Terry /
   logistic model: "for this person, memorability matters more than production quality");
3. few-shot personal reward models such as FSPO (a 68% win rate over the unpersonalized model in its user
   study).

Active questioning keeps the number of questions low: ask only about the pairs the scorers are least sure
about (closest scores, biggest disagreement). A typical budget is 5-10 questions per profile.

### Checking the stand-ins

Stand-in judges are only useful if they agree with people. The cheap check reuses choices already made
(drafts kept or discarded, emails sent as-is) or a tiny labelled set of 10-20 pairs, and reports per
scorer an agreement rate, a correlation and a recommended weight. A full calibration platform was
explicitly out of scope.

## The practices it proposed

1. Stand-ins, not truth: document what data each scorer learned from and whose taste that is.
2. Combine families; disagreement is a signal to ask a person.
3. Explain, don't just score.
4. Detectors are signals, not gates.
5. Personas weigh less than trained taste models until checked.
6. Ask people rarely and smartly.
7. Beware optimizing too hard against one model (Goodhart); use several scorers and a moderate N.
8. Check licenses per model before making it a default.
9. Local by default where possible.
10. Load heavy models lazily, through a GPU lease.

## The first sketch

```python
import tastekit as tk                 # now: import hone_taste as tt

song = tk.songeval()                  # lazy; GPU lease through the model-access package
text = tk.reward_model("skywork-v2-1.7b")
img  = tk.hpsv3()
slop = tk.slop_score(domain="lyrics")
det  = tk.binoculars()
panel = tk.audience("audiences/blues_listeners.toml")
me = tk.profile("alice")
me.ask_if_needed(candidates, budget=5)

s = song(candidate)                   # Score(value, confidence, reason, details)
```

Scorers were to match the selection package's scorer shape without importing it, and declare the input
kinds they accept (`text`, `text_with_prompt`, `audio`, `image`, `image_with_prompt`; `video` later).

## Open questions at the time, and how they were settled

| Question | Recommended then | Outcome in 0.1.0 |
|---|---|---|
| Depend on the selection package? | no, match its shape | no import; `for_select` produces the shape |
| Licenses of default models | check before shipping | checked; SongEval and PickScore are gated behind `accept_license=True`, awaiting owner review ([decisions](../decisions.md#awaiting-owner-review)) |
| Where profiles live | a local file per person | `${HONE_HOME:-.hone}/taste/profiles/<name>.json` |
| Persona judge default | a local model | any `DecisionClient` or `TextClient`; an OpenAI-compatible adapter covers local servers |
| Reference sets | bundle small defaults | bundled for the default reward model; images need the user's own for now |
| Media in v1 | text, songs and images | all three |
| Name | placeholder | `hone-taste` |
| License, Python | MIT or Apache-2.0; 3.11+ | Apache-2.0; Python 3.11+ |

## What changed on the way to 0.1.0

- **Images:** PickScore became the default image scorer and HPSv3 optional, because HPSv3 needs about
  18 GB of VRAM and its own package.
- **`ask_if_needed`** split into `questions(...)` (which pairs to ask) and `ask(...)` (asking in the
  terminal), so the ranking is usable without a terminal.
- **Panels** take a list of personas (or a TOML file) and a client directly, instead of a named audience
  in a config file.
- **Audio in panels** is not passed directly; the caller passes a text description.
- **Deferred:** MuseCritic, ArmoRM per-aspect scores, Fast-DetectGPT, multilingual detectors, FSPO-style
  personal reward models, video, a web page for questions, and a "novelty against past outputs" scorer
  that needs a memory / search package that does not exist yet.
- **Not recorded here:** the proposed validation spike (running SongEval and Slop Score on existing songs
  and comparing with the author's own selections) belonged to the pipeline, not to this package.

## References

**Taste models**
- [SongEval toolkit](https://github.com/ASLP-lab/SongEval) · [paper (arXiv 2505.10793)](https://arxiv.org/abs/2505.10793) · [dataset](https://huggingface.co/datasets/ASLP-lab/SongEval)
- [MuseCritic: multi-aspect song rewards from critiques](https://arxiv.org/pdf/2608.11755)
- [Skywork-Reward-V2](https://github.com/SkyworkAI/Skywork-Reward-V2) · [paper](https://arxiv.org/html/2507.01352v3)
- [Best open-source reward models 2026 (Modal)](https://modal.com/resources/best-open-source-reward-models-rlhf)
- [HPSv3 paper](https://arxiv.org/pdf/2508.03789) · [HPSv2](https://github.com/tgxs002/HPSv2) · [ImageReward / PickScore](https://github.com/p1atdev/ImageReward-PickScore)

**Human-likeness**
- [Slop Score](https://github.com/sam-paech/slop-score) · [leaderboard](https://eqbench.com/slop-score.html)
- [Antislop framework paper](https://arxiv.org/pdf/2510.15061)
- [Binoculars](https://github.com/ahans30/Binoculars) · [paper](https://arxiv.org/abs/2401.12070)
- [Fast-DetectGPT](https://openreview.net/forum?id=Bpcgcr8E8Z)
- [Zero-shot LLM text detectors in practice](https://medium.com/@vzzz/zero-shot-llm-text-detectors-in-practice-2b737ffcadd2)

**Simulated audience**
- [Microsoft TinyTroupe](https://github.com/microsoft/TinyTroupe) · [paper (arXiv 2507.09788)](https://arxiv.org/html/2507.09788v2)

**Personal taste from few picks**
- [FSPO: few-shot preference optimization](https://fewshot-preference-optimization.github.io/) · [paper](https://arxiv.org/html/2502.19312)
- [Active preference learning for LLMs](https://arxiv.org/html/2402.08114v2)
- [Activation reward models for few-shot alignment](https://arxiv.org/html/2507.01368v1)

**Caveats**
- [Scaling laws for reward model overoptimization](https://proceedings.mlr.press/v202/gao23h/gao23h.pdf)
