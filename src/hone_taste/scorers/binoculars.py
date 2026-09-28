"""Binoculars AI-text detector (Hans et al., 2024), extra `detect`. A signal, never a gate by default.

Score = log-perplexity of the text under the performer / cross-perplexity between observer and performer.
Low scores look machine-written. `p(machine) = sigmoid((threshold - score) / SCALE)`, `value = 1 - p`.
The default threshold is the upstream low-false-positive threshold, calibrated for the Falcon-7B pair;
it is only approximate for the small default pair, so recalibrate on your own texts when it matters.
Known false positives: formal, templated and non-native writing.
"""

from __future__ import annotations

import math
from dataclasses import replace
from importlib import import_module
from typing import Any

from hone_taste.ports import GpuLease
from hone_taste.registry import check_license, model_info, require_modules
from hone_taste.scorers.model import TasteModel, free_gpu, model_scorer, resolve_device
from hone_taste.types import FunctionScorer, Score

DEFAULT_OBSERVER = "Qwen/Qwen2.5-0.5B"
DEFAULT_PERFORMER = "Qwen/Qwen2.5-0.5B-Instruct"
DEFAULT_THRESHOLD = 0.9015310749276843  # upstream "low-fpr" threshold (Falcon-7B / Falcon-7B-instruct)
SCALE = 0.05  # about the distance between upstream's low-fpr and accuracy thresholds


def binoculars(
    observer: str = DEFAULT_OBSERVER,
    performer: str = DEFAULT_PERFORMER,
    device: str | None = None,
    gpu: GpuLease | None = None,
    *,
    threshold: float = DEFAULT_THRESHOLD,
    accept_license: bool = False,
    model: TasteModel | None = None,
) -> FunctionScorer:
    """Score text by how human-written it looks: `value = 1 - p(machine)`; `details["raw_score"]`."""
    info = model_info("binoculars")
    check_license(info, accept_license)
    if (observer, performer) != (DEFAULT_OBSERVER, DEFAULT_PERFORMER):
        models = f"{observer} + {performer}"
        info = replace(info, model_id=models, license=f"see the model cards of {models}")
    if model is None:
        require_modules(info.extra, "torch", "transformers")
        model = BinocularsModel(observer, performer, device)

    def to_score(raw: Any) -> Score:
        return binoculars_score(raw["score"], raw.get("tokens"), threshold)

    return model_scorer(info, "human_likeness", frozenset({"text"}), model, to_score, gpu=gpu)


def binoculars_score(raw: float, tokens: float | None, threshold: float) -> Score:
    """Map a Binoculars score to `1 - p(machine)`; confidence grows with the text length (128 tokens = 1).

    >>> binoculars_score(0.9015310749276843, 64, 0.9015310749276843).value
    0.5
    """
    p_machine = 1.0 / (1.0 + math.exp((raw - threshold) / SCALE))
    label = "looks machine-written" if raw < threshold else "looks human-written"
    return Score(
        1.0 - p_machine,
        confidence=min(1.0, tokens / 128) if tokens else None,
        reason=f"binoculars score {raw:.3f} vs threshold {threshold:.3f}: {label} (a signal, not proof)",
        details={"raw_score": raw, "threshold": threshold, "p_machine": p_machine, "tokens": tokens},
    )


class BinocularsModel:  # pragma: no cover - needs torch and weights; exercised by tests/gpu
    """Observer and performer causal LMs sharing one tokenizer."""

    def __init__(self, observer: str, performer: str, device: str | None) -> None:
        self.observer_id, self.performer_id, self.device = observer, performer, device
        self._parts: Any = None  # torch, tokenizer, observer, performer

    def load(self) -> None:
        torch, transformers = import_module("torch"), import_module("transformers")
        self.device = resolve_device(torch, self.device)
        dtype = torch.bfloat16 if self.device.startswith("cuda") else torch.float32
        lm = transformers.AutoModelForCausalLM
        observer = lm.from_pretrained(self.observer_id, dtype=dtype).to(self.device).eval()
        performer = lm.from_pretrained(self.performer_id, dtype=dtype).to(self.device).eval()
        tokenizer = transformers.AutoTokenizer.from_pretrained(self.observer_id)
        self._parts = (torch, tokenizer, observer, performer)

    def predict(self, input: Any) -> dict[str, float]:
        torch, tokenizer, observer, performer = self._parts
        ids = tokenizer(input, return_tensors="pt", truncation=True, max_length=512).input_ids.to(self.device)
        if ids.shape[1] < 2:
            raise ValueError("text too short for Binoculars (needs at least two tokens)")
        with torch.no_grad():
            observer_logits = observer(ids).logits[0, :-1].float()
            performer_logits = performer(ids).logits[0, :-1].float()
        targets = ids[0, 1:]
        log_ppl = torch.nn.functional.cross_entropy(performer_logits, targets).item()
        observer_probs = torch.softmax(observer_logits, dim=-1)
        x_ppl = torch.nn.functional.cross_entropy(performer_logits, observer_probs).item()
        return {"score": log_ppl / x_ppl, "tokens": float(ids.shape[1])}

    def unload(self) -> None:
        self._parts = None
        free_gpu()
