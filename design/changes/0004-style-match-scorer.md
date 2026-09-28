# 0004: A style-match scorer ("does this read like this author?")

## Status

`implemented in 0.1.0` (approved by the owner 2026-09-28; renumbered from a second `0003`). Built:
options 2, 3 and 4 as `tt.stylometry` / `tt.fingerprint` (core, no model), `tt.style_judge` (anchored on
real excerpts only), `tt.style_match` and `tt.calibrate_style`. The package has no style-sheet judge on
purpose. The excerpt-anchored judge has not been calibrated on real posts yet: persona-writer's second
calibration, with this judge, was still to run when this was built.

## Context

Found while building persona-writer (the persona-writer demo app in the honeworks workspace), which writes blog posts in a chosen author's
voice and picks among drafts with hone-select. It needed a scorer for "does this draft read like the
persona?". hone-taste has no such scorer, so the app built three on `hone_taste.types.FunctionScorer`
(`persona-writer/src/persona_writer/style.py`, `stylometry.py`):

- `stylometry`: no model. A fingerprint of the author's posts (mean and spread of 15 surface features:
  words per sentence and its spread, sentences per paragraph, I / we / you rates, contractions,
  questions, dashes, parentheses, word length, headings, list items, code share, bold; plus Burrows'
  Delta over 80 function words). Closeness = half the mean of `max(0, 1 - |z|/3)` over the features,
  half `exp(-(Delta - typical)/typical)`.
- `sheet_judge`: an LLM judge given a written style sheet (tone, rhythm, habits, "never does") and
  short excerpts; a 1-5 "how much does this read like <author>" question plus six yes/no checks.
- `style_judge`: an LLM judge given only the author's real excerpts as reference; a 1-5 closeness
  question, "states things as plainly?", "adds drama the reference does not have?" (inverted).

It calibrated them with `tt.agreement` on posts held out from the profile (public-domain posts by Chris
Wellons, nullprogram.com): for each held-out post, the real opening (~600 words) must score above (a) a
generic post by `gemma4-12b` with the same title and (b) the app's own ghostwritten imitation with the
same title. Judge: `qwen2.5vl-7b`. First calibration (3 posts, 6 pairs):

| Scorer | agrees with the real post | correlation | mean real / generic / ghost |
|---|---|---|---|
| stylometry | 1.00 | 1.00 | 0.56 / 0.37 / 0.44 |
| sheet_judge | 0.33 | -0.33 | 0.68 / 0.66 / 0.91 |
| slop | 0.67 | 0.33 | 0.97 / 0.95 / 0.96 |

The style-sheet judge preferred the imitation (written *from* the same sheet) over the real author in
all three cases: it measures conformance to a description, and an imitation built from the description
conforms better than the original. Stylometry, which knows nothing about meaning, ranked every pair
right. (The excerpt-anchored `style_judge` was added after this finding; its numbers are in
persona-writer's `outputs/personas/wellons-calibration.md` and `REPORT.md`.)

## Problem

- A common need for text generation ("in my voice", "like our docs", "like this author") has no scorer
  in hone-taste, and the obvious one (ask an LLM with a style description) is misleading in exactly the
  case it is used for: ranking imitations.
- The cheap, reliable half (stylometry) is generic and needs no model; every app would rewrite it.
- Calibration against held-out real texts is the step that exposed the problem, and it is a pattern
  (`tt.agreement` over real-vs-imitation pairs) the package could offer directly.

## Options

1. **Leave it to applications** (today).
2. **`tt.stylometry(texts)`** in core: builds a fingerprint from reference texts (no model, no extra) and
   returns a `human_likeness`-style scorer with per-feature z-scores in `details`; the fingerprint is
   JSON-serialisable so it can be stored with a profile.
3. **`tt.style_match(reference_texts, client, *, fingerprint=None, weights=None)`**: stylometry plus an
   excerpt-anchored judge (not a style-sheet judge), combined with weights; `details` carry both halves.
4. **`tt.calibrate_style(scorers, real, imitations)`**: builds the real-vs-imitation pairs and returns an
   `AgreementReport` plus derived weights (each half weighted by its non-negative correlation), the way
   persona-writer does in `calibrate.py`.

## Decision

Proposed: 2 first (small, model-free, clearly useful), then 3 and 4 together, documented with the finding
above as the honesty rule for this scorer: *a judge given a style description rewards imitations of
the description; anchor it on real text and check it against held-out real text before trusting it.*

## Consequences

- A new scorer family member: `personal` (like `profile`), `accepts = {"text"}`.
- Stylometry needs no extra; the judge half uses the existing `DecisionClient` port.
- Fingerprints from few texts are noisy; the spread is floored (25 % of the mean) so one feature cannot
  dominate. The docs should say how many reference texts are enough (persona-writer used 10 posts).

## Migration and compatibility

Additive; nothing existing changes. persona-writer would switch from its own module to the package's
scorer and keep its calibration command.
