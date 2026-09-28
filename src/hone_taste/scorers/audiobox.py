"""Meta Audiobox Aesthetics: four 0-10 ratings of an audio file, extra `songs`.

Axes: CE (content enjoyment), CU (content usefulness), PC (production complexity), PQ (production
quality). Complexity is not "better when higher", so by default `value` is the mean of CE, CU and PQ.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from importlib import import_module
from typing import Any

from hone_taste.ports import GpuLease
from hone_taste.registry import check_license, model_info, require_modules
from hone_taste.scorers.model import TasteModel, check_weights, dimensions_score, free_gpu, model_scorer
from hone_taste.types import FunctionScorer

AXES = ("CE", "CU", "PC", "PQ")
DEFAULT_WEIGHTS = {"CE": 1.0, "CU": 1.0, "PQ": 1.0}


def audiobox(
    gpu: GpuLease | None = None,
    *,
    weights: Mapping[str, float] | None = None,
    accept_license: bool = False,
    model: TasteModel | None = None,
) -> FunctionScorer:
    """Score an audio path: `details["dimensions"]` holds CE, CU, PC, PQ mapped from 0-10 to 0-1."""
    info = model_info("audiobox")
    check_license(info, accept_license)
    chosen = check_weights(weights or DEFAULT_WEIGHTS, AXES)
    if model is None:
        require_modules(info.extra, "torch", "audiobox_aesthetics", "soundfile")
        model = AudioboxModel()
    return model_scorer(
        info,
        "taste_model",
        frozenset({"audio"}),
        model,
        lambda raw: dimensions_score(raw, 0, 10, chosen),
        gpu=gpu,
    )


class AudioboxModel:  # pragma: no cover - needs torch and weights; exercised by tests/gpu
    """`audiobox_aesthetics.infer.initialize_predictor()` (picks the GPU when there is one)."""

    def __init__(self) -> None:
        self._predictor: Any = None

    def load(self) -> None:
        self._predictor = import_module("audiobox_aesthetics.infer").initialize_predictor()

    def predict(self, input: Any) -> dict[str, float]:
        # read the file ourselves: torchaudio >= 2.9 needs torchcodec to load files
        samples, rate = import_module("soundfile").read(str(input), dtype="float32", always_2d=True)
        wav = import_module("torch").from_numpy(samples.T.copy())  # (channels, samples)
        result = self._predictor.forward([{"path": wav, "sample_rate": rate}])[0]
        result = json.loads(result) if isinstance(result, str) else result
        return {axis: float(result[axis]) for axis in AXES}

    def unload(self) -> None:
        self._predictor = None
        free_gpu()
