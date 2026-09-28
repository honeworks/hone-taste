# 0003: A character-consistency scorer

## Status

`implemented in 0.1.0` for option 1 (approved by the owner 2026-09-28): `tt.character_consistency`
with DINOv2-small, reference sheets split into panels, and `testing.FakeEmbedder`. Option 2, the judge
variant, stays open as decided: it waits for several images per decision in hone-select (its 0005). The
real DINOv2 backend has a `-m gpu` test that has not been run yet.

## Context

Found while building teacher-persona (the teacher-persona demo app in the honeworks workspace): a recurring teacher character must look
like the same person in every picture of every lesson. The image model (z-image-turbo) reads only text,
so the app renders several candidates per scene and keeps the one most consistent with a reference
sheet. hone-taste has image *preference* (`image_preference`: would people like this image for this
prompt?) but nothing for "is this the same character as in the reference?", which is a different
question: a beautiful picture of the wrong person should score low.

The app built two signals itself:

- **Embeddings**: DINOv2-small (Apache-2.0, ~90 MB, CPU) cosine similarity between the candidate and
  the closest panel of the reference sheet, mapped to 0-1 with a floor. DINOv2 is used rather than CLIP
  because CLIP-style embeddings reward the same caption/scene more than the same person.
- **A vision judge** with a 4-item checklist (same face and age, same hair, same accessories, same
  outfit and colours) over reference and candidate.

## Options

1. **`tt.character_consistency(reference, *, model="dinov2-small", device=None, gpu=None, floor=...)`**:
   a family-A style scorer returning `Score(value, details={"cosine": ..., "panel": ...})` for
   `{"image": path}`, with the reference embedded once (optionally split into panels). Recorded as a
   `hone.taste.score` span like the other taste models; `for_select(..., field=...)` works unchanged.
2. Also a judge-based variant, `tt.consistency_panel(reference, judge, aspects=[...])`, that asks a
   vision `DecisionClient` the checklist. Needs several images per decision (see hone-select 0005).
3. Leave it to apps.

## Decision

Proposed: option 1 first (cheap, deterministic, no judge), option 2 once hone-select can send several
images. The honesty rules apply: embeddings measure visual similarity, not identity; the reference set
and floor should be checked against a few picks with `tt.agreement`.

## Consequences

- A new optional extra (`images` already pulls torch and transformers; DINOv2 needs nothing more).
- teacher-persona's numbers can seed the reference set: its report lists embedding and judge scores for
  every candidate it rendered, with the pick.
