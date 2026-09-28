"""Image preference models: would people pick this image for this prompt? Extra `images`.

`kind="pickscore"` (default; CLIP-H fine-tuned on Pick-a-Pic, loaded with `transformers`) or
`kind="hpsv3"` (needs the separate `hpsv3` package and a large GPU). Raw scores are mapped to 0-1 with a
reference set (`references/images/<kind>.json`, or pass `reference=`).
"""

from __future__ import annotations

import importlib.util
from importlib import import_module
from pathlib import Path
from typing import Any

from hone_taste.errors import ConfigError, MissingExtra
from hone_taste.normalize import ReferenceSet, as_reference, check_mode, normalized
from hone_taste.ports import GpuLease
from hone_taste.registry import check_license, model_info, require_modules
from hone_taste.scorers.model import TasteModel, free_gpu, model_scorer, resolve_device
from hone_taste.types import FunctionScorer

CLIP_PROCESSOR = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"


def image_preference(
    kind: str = "pickscore",
    device: str | None = None,
    gpu: GpuLease | None = None,
    *,
    reference: str | Path | ReferenceSet | None = None,
    normalization: str = "linear",
    accept_license: bool = False,
    model: TasteModel | None = None,
) -> FunctionScorer:
    """Score `{"prompt": ..., "image": path}`.

    PickScore's license is unclear, so it needs `accept_license=True`.
    """
    if kind not in ("pickscore", "hpsv3"):
        raise ConfigError(f"unknown image preference model {kind!r}; choose 'pickscore' or 'hpsv3'")
    info = model_info(kind)
    check_license(info, accept_license)
    if model is None:
        model = _backend(kind, info.model_id, device)
    check_mode(normalization)
    ref = as_reference(reference, "images", kind)
    return model_scorer(
        info,
        "taste_model",
        frozenset({"image_with_prompt"}),
        model,
        lambda raw: normalized(raw["score"], ref, mode=normalization),
        gpu=gpu,
    )


def _backend(kind: str, model_id: str, device: str | None) -> TasteModel:
    if kind == "pickscore":
        require_modules("images", "torch", "transformers", "PIL")
        return PickScoreModel(model_id, device)
    if importlib.util.find_spec("hpsv3") is None:
        raise MissingExtra("HPSv3 needs its own package: pip install hpsv3 (and about 18 GB of VRAM)")
    return HPSv3Model(device)


class PickScoreModel:  # pragma: no cover - needs torch and weights; exercised by tests/gpu
    """PickScore: `logit_scale.exp() * cosine(text, image)` of the fine-tuned CLIP-H."""

    def __init__(self, model_id: str, device: str | None) -> None:
        self.model_id, self.device = model_id, device
        self._parts: Any = None  # torch, processor, model

    def load(self) -> None:
        torch, transformers = import_module("torch"), import_module("transformers")
        self.device = resolve_device(torch, self.device)
        processor = transformers.AutoProcessor.from_pretrained(CLIP_PROCESSOR)
        net = transformers.AutoModel.from_pretrained(self.model_id).eval().to(self.device)
        self._parts = (torch, processor, net)

    def predict(self, input: Any) -> dict[str, float]:
        torch, processor, net = self._parts
        image = import_module("PIL.Image").open(input["image"]).convert("RGB")
        with torch.no_grad():
            pixels = processor(images=[image], return_tensors="pt").to(self.device)
            text = processor(
                text=[input["prompt"]], padding=True, truncation=True, max_length=77, return_tensors="pt"
            ).to(self.device)
            image_embeds = net.get_image_features(**pixels)
            text_embeds = net.get_text_features(**text)
            image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
            text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
            score = net.logit_scale.exp() * (text_embeds @ image_embeds.T)[0][0]
        return {"score": float(score.item())}

    def unload(self) -> None:
        self._parts = None
        free_gpu()


class HPSv3Model:  # pragma: no cover - needs the hpsv3 package and a large GPU
    """HPSv3 through its own `HPSv3RewardInferencer`."""

    def __init__(self, device: str | None) -> None:
        self.device = device
        self._inferencer: Any = None

    def load(self) -> None:
        self.device = resolve_device(import_module("torch"), self.device)
        self._inferencer = import_module("hpsv3").HPSv3RewardInferencer(device=self.device)

    def predict(self, input: Any) -> dict[str, float]:
        rewards = self._inferencer.reward([input["prompt"]], image_paths=[str(input["image"])])
        return {"score": float(rewards[0][0].item())}

    def unload(self) -> None:
        self._inferencer = None
        free_gpu()
