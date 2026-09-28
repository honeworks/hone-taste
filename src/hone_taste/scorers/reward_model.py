"""A Hugging Face reward model: how good is this response to this prompt? Extra `text`.

Default: the smallest Skywork-Reward-V2 model. The raw reward (a logit) is mapped to 0-1 with a reference
set (`references/text/reward_model.json`, or pass `reference=`).
"""

from __future__ import annotations

from dataclasses import replace
from importlib import import_module
from pathlib import Path
from typing import Any

from hone_taste.errors import ConfigError
from hone_taste.normalize import ReferenceSet, as_reference, check_mode, normalized
from hone_taste.ports import GpuLease
from hone_taste.registry import check_license, model_info, require_modules
from hone_taste.scorers.model import TasteModel, free_gpu, model_scorer, resolve_device
from hone_taste.types import FunctionScorer


def reward_model(
    model_id: str | None = None,
    device: str | None = None,
    gpu: GpuLease | None = None,
    *,
    reference: str | Path | ReferenceSet | None = None,
    normalization: str = "linear",
    accept_license: bool = False,
    model: TasteModel | None = None,
) -> FunctionScorer:
    """Score `{"prompt": ..., "response": ...}` with a sequence-classification reward model.

    `details["raw"]` keeps the reward. A `model_id` other than the default needs `reference=` (a JSON file
    of its raw rewards on good and bad responses, see `hone_taste.normalize`).
    """
    info = model_info("reward_model")
    check_license(info, accept_license)
    if model_id is not None and model_id != info.model_id:
        if reference is None:
            raise ConfigError(f"{model_id} has no packaged reference set; pass reference= with it")
        info = replace(info, model_id=model_id, license=f"see the model card of {model_id}")
    if model is None:
        require_modules(info.extra, "torch", "transformers")
        model = RewardModel(info.model_id, device)
    check_mode(normalization)
    ref = as_reference(reference, "text", "reward_model")
    return model_scorer(
        info,
        "taste_model",
        frozenset({"text_with_prompt"}),
        model,
        lambda raw: normalized(raw["reward"], ref, mode=normalization),
        gpu=gpu,
    )


class RewardModel:  # pragma: no cover - needs torch and weights; exercised by tests/gpu
    """`AutoModelForSequenceClassification` over the chat template of (prompt, response)."""

    def __init__(self, model_id: str, device: str | None) -> None:
        self.model_id, self.device = model_id, device
        self._parts: Any = None  # torch, tokenizer, model

    def load(self) -> None:
        torch, transformers = import_module("torch"), import_module("transformers")
        device = resolve_device(torch, self.device)
        dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
        tokenizer = transformers.AutoTokenizer.from_pretrained(self.model_id)
        net = transformers.AutoModelForSequenceClassification.from_pretrained(
            self.model_id, dtype=dtype, num_labels=1
        )
        self._parts = (torch, tokenizer, net.to(device).eval())
        self.device = device

    def predict(self, input: Any) -> dict[str, float]:
        torch, tokenizer, net = self._parts
        conversation = [
            {"role": "user", "content": input["prompt"]},
            {"role": "assistant", "content": input["response"]},
        ]
        text = tokenizer.apply_chat_template(conversation, tokenize=False)
        tokens = tokenizer(text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            return {"reward": float(net(**tokens).logits[0][0].item())}

    def unload(self) -> None:
        self._parts = None
        free_gpu()
