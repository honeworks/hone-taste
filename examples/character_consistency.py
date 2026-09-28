"""Character consistency: keep the candidate that shows the same character as the reference sheet.

What: `tt.character_consistency(reference, panels=3)` scores an image path by how close its image
      embedding is to the closest panel of a reference character sheet (DINOv2-small by default). A
      beautiful picture of the wrong person scores low, which `image_preference` cannot tell you. Here
      the embedder is `hone_taste.testing.FakeEmbedder`, a table of vectors, so the example runs offline
      without torch; drop `model=` to use the real model (extra `images`).
How:  1. pass the reference image (or several) and, for a sheet that shows the character n times side by
         side, `panels=n`: the candidate is compared with the closest strip,
      2. call the scorer on each candidate's path; read `value`, `details["cosine"]` and
         `details["panel"]` (which strip matched),
      3. in hone-select, `tt.for_select(scorer, file="image")` reads `candidate.files["image"]`,
      4. `close()` frees the model when you are done.
Why:  an image model that reads only text cannot keep a character consistent across scenes; rendering
      several candidates and keeping the most consistent one can. `value` maps the cosine from `floor`
      (0.25: unrelated pictures of people) to 1. It measures visual similarity, not identity, so check the
      reference and the floor against a few of your own picks with `tt.agreement` (design/changes/0003).

Run: uv run python examples/character_consistency.py
"""

from dataclasses import dataclass, field
from typing import Any

import hone_taste as tt
from hone_taste.testing import FakeEmbedder

# What a real embedder might return: the sheet's three panels (front, three-quarter, side) and three
# candidates for a new scene. "#k" names the k-th strip of a split image.
VECTORS = {
    "teacher-sheet.png#0": [1.0, 0.1, 0.0, 0.0],
    "teacher-sheet.png#1": [0.7, 0.7, 0.1, 0.0],
    "teacher-sheet.png#2": [0.1, 1.0, 0.0, 0.0],
    "lesson3-a.png": [0.2, 0.9, 0.1, 0.2],  # the teacher, seen from the side
    "lesson3-b.png": [0.6, 0.4, 0.5, 0.3],  # similar clothes, a different face
    "lesson3-c.png": [0.0, 0.1, 1.0, 0.5],  # a lovely picture of someone else
}

embedder = FakeEmbedder(VECTORS)
consistent = tt.character_consistency("teacher-sheet.png", panels=3, model=embedder)

scores = {path: consistent(path) for path in ("lesson3-a.png", "lesson3-b.png", "lesson3-c.png")}
for path, s in scores.items():
    assert s.value is not None
    print(f"{path}: value={s.value:.2f} cosine={s.details['cosine']:.2f} panel={s.details['panel']}")
best = max(scores, key=lambda p: scores[p].value or 0.0)
print("keep:", best, "|", scores[best].details["caveat"])
assert best == "lesson3-a.png" and scores[best].details["panel"] == 2
assert scores["lesson3-c.png"].value == 0.0  # below the floor

# The model loaded once, on the first call; the sheet was embedded once (three strips).
assert embedder.events == ["load"]
assert [c for c in embedder.calls if c[0] == "teacher-sheet.png"] == [
    ("teacher-sheet.png", (k, 3)) for k in range(3)
]


# In hone-select: read the candidate's image file.
@dataclass
class Candidate:  # hone-select passes objects with id, data, files, meta
    id: str
    files: dict[str, str]
    data: dict[str, Any] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)


pick = tt.for_select(consistent, file="image")
print("as a hone-select scorer:", round(pick(Candidate("c1", {"image": "lesson3-a.png"})).value or 0.0, 2))

# An image the embedder cannot read is an error score, never 0.
missing = consistent("no-such-file.png")
print("missing:", missing.value, missing.error)
assert missing.value is None and missing.error

consistent.close()
assert embedder.events == ["load", "unload"]
