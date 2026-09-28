"""Character consistency: does this picture show the same character as the reference? Extra `images`.

A different question from `image_preference` ("would people like this image?"): a beautiful picture of
the wrong person should score low. The default embedder is DINOv2-small (Apache-2.0, about 90 MB, runs on
a CPU); its features describe shapes and parts rather than captions, so two pictures of the same
stylised character land closer than two characters in the same scene (CLIP-style embeddings reward the
same caption). Embeddings measure visual similarity, not identity: check the reference and the `floor`
against a few of your own picks with `tt.agreement` (design/changes/0003).
"""

from __future__ import annotations

import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from importlib import import_module
from typing import Any, Protocol

from hone_taste.errors import ConfigError
from hone_taste.ports import GpuLease
from hone_taste.registry import model_info, require_modules
from hone_taste.scorers.model import free_gpu, model_scorer, resolve_device
from hone_taste.types import FunctionScorer, Score

DEFAULT_FLOOR = 0.25  # cosine at or below this maps to 0 (unrelated pictures of people land about here)
CAVEAT = "visual similarity to the reference, not identity; check the floor against a few real picks"
Reference = str | os.PathLike[str]


class ImageEmbedder(Protocol):
    """An image model: `load()` once, `embed()` many times, `unload()` to free its memory.

    `part=(k, n)` embeds the k-th of n equal vertical strips of the image, left to right (a character
    sheet showing the character n times); `(0, 1)` is the whole image.
    """

    def load(self) -> None: ...
    def embed(self, image: str, part: tuple[int, int] = (0, 1)) -> Sequence[float]: ...
    def unload(self) -> None: ...


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity of two vectors (0 when either is all zeros).

    >>> round(cosine([1.0, 0.0], [1.0, 1.0]), 4)
    0.7071
    """
    if len(a) != len(b):
        raise ValueError(f"embeddings differ in length ({len(a)} vs {len(b)})")
    norm = math.hypot(*a) * math.hypot(*b)
    return sum(x * y for x, y in zip(a, b, strict=True)) / norm if norm else 0.0


@dataclass(slots=True)
class _ClosestPanel:
    """The `TasteModel` behind the scorer: the candidate's cosine to the closest reference panel."""

    embedder: ImageEmbedder
    panels: list[tuple[str, tuple[int, int]]]
    vectors: list[Sequence[float]] = field(default_factory=list[Sequence[float]])

    def load(self) -> None:
        self.embedder.load()

    def predict(self, input: Any) -> dict[str, float]:
        if not self.vectors:  # the reference is embedded once, on the first call
            self.vectors = [self.embedder.embed(image, part) for image, part in self.panels]
        target = self.embedder.embed(str(input))
        cosines = [cosine(vector, target) for vector in self.vectors]
        best = max(range(len(cosines)), key=cosines.__getitem__)
        return {"cosine": cosines[best], "panel": float(best)}

    def unload(self) -> None:
        self.embedder.unload()


def character_consistency(
    reference: Reference | Sequence[Reference],
    *,
    panels: int = 1,
    floor: float = DEFAULT_FLOOR,
    device: str | None = None,
    gpu: GpuLease | None = None,
    model: ImageEmbedder | None = None,
) -> FunctionScorer:
    """Score an image path by how closely it matches the reference character.

    `reference` is one image or several (e.g. a front and a side view); `panels=n` splits each into n
    equal vertical strips (a character sheet), and the candidate is compared with the closest one.
    `value = clamp((cosine - floor) / (1 - floor))`; `details` carry `cosine`, the matching `panel`
    (index over every strip of every reference, in order) and the `floor`. `model=` injects an embedder
    (`hone_taste.testing.FakeEmbedder`); by default DINOv2-small, loaded on the first call.

    >>> from hone_taste.testing import FakeEmbedder
    >>> fake = FakeEmbedder({"sheet.png": [1.0, 0.0], "same.png": [0.9, 0.1], "other.png": [0.0, 1.0]})
    >>> same = character_consistency("sheet.png", model=fake)
    >>> round(same("same.png").value, 2), same("other.png").value
    (0.99, 0.0)
    """
    references = [str(r) for r in ([reference] if isinstance(reference, str | os.PathLike) else reference)]
    if not references or panels < 1 or not 0 <= floor < 1:
        raise ConfigError(
            f"character_consistency needs at least one reference image, panels >= 1 and 0 <= floor < 1; "
            f"got {len(references)} references, panels={panels}, floor={floor}"
        )
    info = model_info("dinov2_small")
    if model is None:
        require_modules("images", "torch", "transformers", "PIL")
        model = DinoEmbedder(info.model_id, device)
    strips = [(image, (k, panels)) for image in references for k in range(panels)]

    def to_score(raw: Mapping[str, float]) -> Score:
        value = min(1.0, max(0.0, (raw["cosine"] - floor) / (1 - floor)))
        panel = int(raw["panel"])
        details = {"cosine": raw["cosine"], "raw": raw["cosine"], "panel": panel, "floor": floor}
        reason = f"cosine {raw['cosine']:.3f} to reference panel {panel}"
        return Score(value, reason=reason, details={**details, "caveat": CAVEAT})

    scorer = model_scorer(
        info, "taste_model", frozenset({"image"}), _ClosestPanel(model, strips), to_score, gpu=gpu
    )
    scorer.name = "character_consistency"
    return scorer


class DinoEmbedder:  # pragma: no cover - needs torch and weights; exercised by tests/gpu
    """DINOv2 through `transformers`: the pooled output, L2-normalized."""

    def __init__(self, model_id: str, device: str | None) -> None:
        self.model_id, self.device = model_id, device
        self._parts: Any = None  # torch, processor, model

    def load(self) -> None:
        torch, transformers = import_module("torch"), import_module("transformers")
        self.device = resolve_device(torch, self.device)
        processor = transformers.AutoImageProcessor.from_pretrained(self.model_id)
        net = transformers.AutoModel.from_pretrained(self.model_id).eval().to(self.device)
        self._parts = (torch, processor, net)

    def embed(self, image: str, part: tuple[int, int] = (0, 1)) -> list[float]:
        torch, processor, net = self._parts
        picture = import_module("PIL.Image").open(image).convert("RGB")
        index, count = part
        width = picture.width // count
        picture = picture.crop((index * width, 0, (index + 1) * width, picture.height))
        with torch.no_grad():
            inputs = processor(images=picture, return_tensors="pt").to(self.device)
            vector = net(**inputs).pooler_output[0]
        return torch.nn.functional.normalize(vector, dim=0).tolist()

    def unload(self) -> None:
        self._parts = None
        free_gpu()
