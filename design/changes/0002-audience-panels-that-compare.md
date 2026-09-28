# 0002: Audience panels that compare, and say when they cannot tell candidates apart

## Status

`implemented in 0.1.0` (approved by the owner 2026-09-28). Built: option 3 (`details["ceiling"]`,
`tt.spread`) and option 2 as `panel.pairwise(a, b)` with `tt.for_select_pairwise` for hone-select. Each
persona sees the pair in alternating order, so a first-position bias ends in a tie. The batch mode
(`panel.rank(items)`) is not built: pairwise covers the tie-breaks that needed it.

## Context

Found while building OneShotStudio (the OneShotStudio demo app in the honeworks workspace). Its idea and lyrics selections use
`tt.for_select(tt.audience(brief.personas, question, judge, anchors=...), field=...)` with the brief's
two or three personas and `qwen2.5vl-7b` (through `mk.decision`) as the client. Over the three demo runs
the panel's values per selection were:

| Run | idea candidates (4) | lyrics finalists (2) |
|---|---|---|
| sea shanty | 1.0, 1.0, 1.0, 1.0 | 0.75, 0.75 |
| hotel English | 1.0, 0.75, 0.75, 0.75 | 0.5, 0.5 |
| TikTok challenge | 1.0, 1.0, 1.0, 1.0 | 1.0, 0.75 |

Two of three idea panels gave every candidate the top of the scale, even after anchors were added
("1: I would skip it, 3: it is fine, 5: I would share it with friends"). The README already warns that
a panel "tends to be too positive"; in a selection, a panel that rates every candidate the same adds
calls and weight but no information, and nothing in the score says so.

## Problem

- A panel rates each candidate alone, so a one-LLM audience anchors on "this is good" and saturates.
  The value is a ceiling, not a preference.
- Callers cannot tell a real consensus from a saturated scale: `details["disagreement"]` is about
  personas disagreeing on one candidate, not about candidates being told apart.

## Options

1. **Document it** (today's caveat).
2. **Comparative mode**: `tt.audience(..., compare=True)` returns a scorer over a *batch* of candidates
   (`panel.rank(items)`); each persona sees all candidates at once (shuffled, labelled) and distributes
   ratings or ranks them; values are the normalized mean rank. With hone-select, `for_select` would need
   a batch hook, or the panel is used as a pairwise judge (`panel.pairwise(a, b)`), which hone-select
   already supports through the `pairwise` component shape.
3. **Say it**: a `ceiling` flag in `details` when every persona gives the top rating, and a
   `tt.agreement`-style helper `tt.spread(scores)` that reports when a scorer gave (nearly) the same value
   to every candidate of a batch, so callers can drop or down-weight it.
4. **Stricter prompting**: ask each persona for the *weakest* thing first, then a rating; cheap, may
   reduce saturation, cannot remove it.

## Decision

Proposed: 3 now (small, honest, no API break), 2 as the real fix, starting with `panel.pairwise(a, b)`
because hone-select can use it directly for tie-breaks (asked in both orders by hone-select, which also
removes the position bias the plain judge showed in the same runs; see hone-select change 0003).

## Consequences

- Selections can see that a panel did not discriminate and treat it accordingly.
- Comparative panels cost one call per persona per pair (pairwise) or per batch (rank) instead of one per
  candidate; for small N they are no more expensive.
- Records: comparative calls get their own span attribute (`hone.taste.panel.mode = "compare"`).

## Migration and compatibility

Additive: a new `details["ceiling"]` key, a new helper, and a new keyword / method on panels. Existing
calls behave as today.
