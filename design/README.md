# Why hone-taste exists

## The problem

Automatic checks in generative-AI pipelines usually measure **correctness**: the lyrics are audible, the
JSON parses, the audio does not clip, the answer contains the facts. They say nothing about the question
that decides whether the output is any good: **would people like it, and does it feel human-made?**

The obvious answer, asking a person, does not scale. Rating every output by hand is slow, and even the
"learn my taste" approaches usually want a hundred pairwise picks before they help.

Good building blocks exist, but they are scattered: models trained on thousands of human ratings of songs,
text and images; lists of phrases that large language models overuse; zero-shot AI-text detectors;
persona simulations. Each has its own code, input format and score range, and most come with no honest
statement of whose taste they learned or what they may be used for.

## Who it is for

Anyone who generates text, lyrics, songs or images and needs to pick, rank or flag outputs by *taste*,
not just by correctness: a best-of-N selection step, a pipeline that drops generic drafts, or a person who
wants a shortlist before deciding.

## Core ideas

1. **Stand-ins for a human judge, cheapest first.** Four families answer the same question in different
   ways:

   | Family | Idea | Needs a human? |
   |---|---|---|
   | A. Taste models | small models trained on many human ratings predict what people would say | no |
   | B. Human-likeness | overused AI phrasing (slop), AI-text detectors, your own banned patterns | no |
   | C. Audience panel | a language model plays several target personas; each rates the output with a reason | no |
   | D. Personal taste | learn one person's taste from 5-10 picks; ask only about the closest calls | a few picks |

2. **One score shape.** Every scorer returns the same `Score`: a 0-1 `value`, a `confidence`, a `reason`,
   per-aspect `details` and an `error`. Scorers therefore combine, normalize and plug into any
   selection step that accepts a callable.
3. **"Could not score" is `None`, never 0.** A failed model call, a wrong input kind or an empty text gives
   `Score(None, error=...)`; combining drops missing values instead of counting them as bad.
4. **Honesty is part of the result.** Each scorer states whose ratings it learned from and under which
   license; detectors are signals, never gates; personas are one model pretending and should count less;
   optimizing hard against one taste model finds outputs that fool it (Goodhart), so combine families.
5. **Explain, don't just score.** Slop lists the phrases it found with character spans, panels give each
   persona's reason, taste models give per-aspect breakdowns.
6. **Ask people rarely and smartly.** A few pairwise picks re-weight the other scorers for one person, and
   the next questions are the pairs the scorers are least sure about.
7. **Useful alone, light by default.** The core (slop, patterns, panels, profiles, combining) needs no GPU
   and no torch. Heavy models are optional extras, load lazily, run inside an injected GPU lease and are
   freed by `close()`. Models with unclear license terms need an explicit `accept_license=True`.
8. **Small ports, no framework.** The package talks to the outside through a few `typing.Protocol` ports
   (a decision client, a text client, a GPU lease, a record sink). Anything with the right methods fits,
   and every port has a public fake and a contract checker.

## What it deliberately does not do

- It is not a selection engine and not a model client: it scores, and leaves choosing and calling models
  to the caller.
- It does not train taste models from scratch.
- It is not a tool to make AI text undetectable. Human-likeness is used to improve quality, not to evade
  disclosure.
- No video taste, no web page for answering questions (the command line asks in v0.1), and no few-shot
  personal reward models yet.
- It is not an experiment or calibration platform; the agreement check is deliberately small.

## What is in this folder

| File | What it holds |
|---|---|
| [`current.md`](current.md) | the design as it stands today: concepts, rules and the guaranteed behaviour (acceptance cases) |
| [`changes/`](changes/) | one record per design change: what was found, what was decided, why, and how to migrate |
| [`history/`](history/) | the early research the design started from, kept readable |
| [`decisions.md`](decisions.md) | smaller implementation choices, and the ones awaiting owner review |

A new design change starts as a record in `changes/` with status `proposed`; see
[CONTRIBUTING.md](../CONTRIBUTING.md#changing-the-design). How the package was built is described in the
[README](../README.md#how-this-was-built).
