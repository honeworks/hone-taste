"""Lazy loading of heavy models behind a `Scorer`: load on the first call, inside a `GpuLease`, and free
on `close()`.

A backend is any object with `load()`, `predict(input) -> {name: raw value}` and `unload()`
(`TasteModel`). The real backends sit next to their scorers and import torch & co. only in `load()`;
tests inject `hone_taste.testing.FakeTasteModel`.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import replace
from importlib import import_module
from typing import Any, Protocol

from hone_taste._tracing import current_trace
from hone_taste.errors import ConfigError
from hone_taste.ports import GpuLease, NullGpuLease
from hone_taste.registry import ModelInfo
from hone_taste.types import FunctionScorer, Score


class TasteModel(Protocol):
    """A heavy local model: `load()` once, `predict()` many times, `unload()` to free its memory."""

    def load(self) -> None: ...
    def predict(self, input: Any) -> Mapping[str, float]: ...
    def unload(self) -> None: ...


def model_scorer(
    info: ModelInfo,
    family: str,
    accepts: frozenset[str],
    backend: TasteModel,
    to_score: Callable[[Mapping[str, float]], Score],
    *,
    gpu: GpuLease | None = None,
) -> FunctionScorer:
    """A scorer that runs `backend` under `gpu.lease(...)` and turns its raw outputs into a `Score`."""
    lease = gpu if gpu is not None else NullGpuLease()
    loaded = False

    def score(input: Any) -> Score:
        nonlocal loaded
        with lease.lease(info.name, info.vram_gb, trace=current_trace()):
            if not loaded:
                backend.load()
                loaded = True
            raw = backend.predict(input)
        result = to_score(raw)
        honesty = {"model_id": info.model_id, "license": info.license, "source_data": info.source_data}
        return replace(result, details={**result.details, **honesty})

    def close() -> None:
        nonlocal loaded
        if loaded:
            backend.unload()
            loaded = False

    return FunctionScorer(
        info.name,
        family,
        accepts,
        score,
        license=info.license,
        source_data=info.source_data,
        on_close=close,
        model_id=info.model_id,
    )


def check_weights(weights: Mapping[str, float], dimensions: tuple[str, ...]) -> dict[str, float]:
    """Validated dimension weights: known names, >= 0, positive total."""
    unknown = sorted(set(weights) - set(dimensions))
    if unknown or any(w < 0 for w in weights.values()) or sum(weights.values()) <= 0:
        raise ConfigError(
            f"weights must name dimensions from {list(dimensions)} with values >= 0 and a positive total; "
            f"got {dict(weights)}"
        )
    return dict(weights)


def dimensions_score(
    raw: Mapping[str, float], low: float, high: float, weights: Mapping[str, float]
) -> Score:
    """Each dimension mapped from its fixed scale [low, high] to 0-1; value = their weighted mean.

    >>> s = dimensions_score({"a": 5.0, "b": 1.0}, 1, 5, {"a": 3, "b": 1})
    >>> s.value, s.details["dimensions"]
    (0.75, {'a': 1.0, 'b': 0.0})
    """
    dimensions = {k: min(1.0, max(0.0, (v - low) / (high - low))) for k, v in raw.items()}
    total = sum(weights.values())
    value = sum(dimensions[k] * w for k, w in weights.items()) / total
    reason = ", ".join(f"{k}={v:.2f}" for k, v in dimensions.items())
    return Score(
        value, reason=reason, details={"dimensions": dimensions, "raw": dict(raw), "weights": dict(weights)}
    )


def resolve_device(torch: Any, device: str | None) -> str:
    """`device`, or "cuda" when available, else "cpu"."""
    return device or ("cuda" if torch.cuda.is_available() else "cpu")


def free_gpu() -> None:  # pragma: no cover - needs torch
    """Return cached GPU memory after a model was dropped."""
    torch = import_module("torch")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
