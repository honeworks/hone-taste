"""SongEval: five aesthetic dimensions of a full song (audio file in), extra `songs`.

SongEval (github.com/ASLP-lab/SongEval) is not pip-installable. Its small scoring head is re-implemented
below from upstream `model.py` so the checkpoint can be loaded (see THIRD_PARTY_NOTICES.md: the
upstream license is unclear). The checkpoint is downloaded once to
`${HONE_HOME:-.hone}/taste/models/songeval/` (or pass `checkpoint=`). Features come from the MuQ encoder
(`muq` package). Scores are on SongEval's 1-5 scale.
"""

from __future__ import annotations

import os
import urllib.request
from collections.abc import Mapping
from dataclasses import dataclass, replace
from importlib import import_module
from pathlib import Path
from typing import Any

from hone_taste.errors import ConfigError, ModelUnavailable
from hone_taste.ports import GpuLease
from hone_taste.registry import check_license, model_info, require_modules
from hone_taste.scorers.model import (
    TasteModel,
    check_weights,
    dimensions_score,
    free_gpu,
    model_scorer,
    resolve_device,
)
from hone_taste.types import FunctionScorer, Score

MIN_EXCERPT_S = 30.0  # below this an excerpt says little about a song
DIMENSIONS = ("coherence", "musicality", "memorability", "clarity", "naturalness")  # SongEval head order
MUQ_MODEL = "OpenMuQ/MuQ-large-msd-iter"


def songeval(
    device: str | None = None,
    gpu: GpuLease | None = None,
    *,
    weights: Mapping[str, float] | None = None,
    checkpoint: str | Path | None = None,
    accept_license: bool = False,
    model: TasteModel | None = None,
    max_seconds: float | None = None,
) -> FunctionScorer:
    """Score a song (audio path): `value` = weighted mean of the five dimensions mapped from 1-5 to 0-1.

    `details["dimensions"]` holds coherence, musicality, memorability, clarity (of song structure) and
    naturalness (of vocal breathing and phrasing). The license is unclear (non-commercial at least for the
    MuQ encoder), so `accept_license=True` is required.
    `model=` injects a backend (e.g. `hone_taste.testing.FakeTasteModel`).

    Long songs: `max_seconds=90` scores the middle 90 s only and, after an out-of-memory error, retries
    on half the length (down to 30 s); `details["excerpt_s"]` says what was scored (`None` = the whole
    song). A backend gets the limit as `predict(path, max_seconds=...)`. Without `max_seconds` the whole
    song is scored once, as before.

    >>> from hone_taste.testing import FakeTasteModel
    >>> fake = FakeTasteModel(dict.fromkeys(DIMENSIONS, 4.0))
    >>> songeval(model=fake, accept_license=True, max_seconds=90)("song.wav").details["excerpt_s"]
    90.0
    """
    info = model_info("songeval")
    check_license(info, accept_license)
    chosen = check_weights(weights or dict.fromkeys(DIMENSIONS, 1.0), DIMENSIONS)
    if model is None:
        require_modules(info.extra, "torch", "librosa", "muq", "safetensors")
        model = SongEvalModel(device, Path(checkpoint) if checkpoint else None, info.checkpoint_url)
    if max_seconds is not None and max_seconds < MIN_EXCERPT_S:
        raise ConfigError(f"max_seconds must be at least {MIN_EXCERPT_S:g} (or None for the whole song)")

    def to_score(raw: Mapping[str, float]) -> Score:
        result = dimensions_score({k: raw[k] for k in DIMENSIONS}, 1, 5, chosen)
        excerpt = float(raw.get("excerpt_s", 0.0)) or None
        return replace(result, details={**result.details, "excerpt_s": excerpt})

    return model_scorer(
        info, "taste_model", frozenset({"audio"}), Excerpts(model, max_seconds), to_score, gpu=gpu
    )


def _out_of_memory(exc: Exception) -> bool:
    return "out of memory" in str(exc).lower()


@dataclass(slots=True)
class Excerpts:
    """Runs the backend on the whole song, or on a middle excerpt that halves after each out-of-memory
    error (only when `max_seconds` is set: without it, behaviour is exactly as before 0005).

    Its raw output adds `excerpt_s` (0 = the whole song) to the backend's dimensions."""

    backend: TasteModel
    max_seconds: float | None

    def load(self) -> None:
        self.backend.load()

    def unload(self) -> None:
        self.backend.unload()

    def predict(self, input: Any) -> dict[str, float]:
        seconds = self.max_seconds
        if seconds is None:
            return {**self.backend.predict(input), "excerpt_s": 0.0}
        while True:
            try:
                raw = self.backend.predict(input, max_seconds=seconds)  # type: ignore[call-arg]
                return {**raw, "excerpt_s": seconds}
            except Exception as exc:
                if not _out_of_memory(exc) or seconds / 2 < MIN_EXCERPT_S:
                    raise
                seconds /= 2


def default_checkpoint() -> Path:
    return Path(os.environ.get("HONE_HOME", ".hone")) / "taste" / "models" / "songeval" / "model.safetensors"


def download(url: str, path: Path) -> None:
    """Fetch `url` to `path` atomically (temp file + rename)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(".partial")
    try:
        with urllib.request.urlopen(url, timeout=60) as response, partial.open("wb") as out:  # noqa: S310 - fixed https URL from the registry
            while chunk := response.read(1 << 20):
                out.write(chunk)
    except OSError as exc:
        partial.unlink(missing_ok=True)
        raise ModelUnavailable(f"could not download the SongEval checkpoint from {url}: {exc}") from exc
    partial.replace(path)


class SongEvalModel:  # pragma: no cover - needs torch and weights; exercised by tests/gpu
    """MuQ features (layer 6) -> SongEval's attention head -> five scores in [1, 5]."""

    def __init__(self, device: str | None, checkpoint: Path | None, checkpoint_url: str) -> None:
        self.device, self.checkpoint, self.checkpoint_url = device, checkpoint, checkpoint_url
        self._parts: Any = None  # torch, muq encoder, head

    def load(self) -> None:
        torch = import_module("torch")
        device = resolve_device(torch, self.device)
        path = self.checkpoint or default_checkpoint()
        if not path.exists():
            download(self.checkpoint_url, path)
        head = _songeval_head(torch)
        head.load_state_dict(import_module("safetensors.torch").load_file(str(path)))  # strict: all keys
        encoder = import_module("muq").MuQ.from_pretrained(MUQ_MODEL)
        self._parts = (torch, encoder.to(device).eval(), head.to(device).eval())
        self.device = device

    def predict(self, input: Any, max_seconds: float | None = None) -> dict[str, float]:
        torch, encoder, head = self._parts
        librosa = import_module("librosa")
        offset = 0.0
        if max_seconds is not None:
            offset = max(0.0, (librosa.get_duration(path=str(input)) - max_seconds) / 2)
        wav, _ = librosa.load(str(input), sr=24000, offset=offset, duration=max_seconds)
        try:
            with torch.no_grad():
                audio = torch.tensor(wav).unsqueeze(0).to(self.device)
                features = encoder(audio, output_hidden_states=True)["hidden_states"][6]
                scores = head(features).squeeze(0).tolist()
        except Exception:
            free_gpu()  # release what the failed pass held before a shorter retry
            raise
        return dict(zip(DIMENSIONS, scores, strict=True))

    def unload(self) -> None:
        self._parts = None
        free_gpu()


def _songeval_head(torch: Any) -> Any:  # pragma: no cover - needs torch
    """SongEval's `Generator` (github.com/ASLP-lab/SongEval model.py; config.yaml sizes).

    See THIRD_PARTY_NOTICES.md for the upstream license statements.
    """
    nn = torch.nn

    class Generator(nn.Module):
        def __init__(self, in_features: int = 1024, hidden: int = 4096, classes: int = 5, layers: int = 4):
            super().__init__()  # pyright: ignore[reportUnknownMemberType]
            self.attn = nn.ModuleList(
                [
                    nn.MultiheadAttention(embed_dim=in_features, num_heads=8, dropout=0.2, batch_first=True)
                    for _ in range(layers)
                ]
            )
            self.ffd = nn.Sequential(
                nn.Linear(in_features, hidden), nn.ReLU(), nn.Linear(hidden, in_features)
            )
            self.dropout = nn.Dropout(0.2)
            self.fc = nn.Linear(in_features * 2, classes)
            self.proj = nn.Tanh()

        def forward(self, ssl_feature: Any) -> Any:
            ssl_feature = self.ffd(ssl_feature)
            attended = ssl_feature
            for attn in self.attn:
                attended, _ = attn(attended, attended, attended)
            pooled = torch.concat([torch.mean(attended, dim=1), torch.max(ssl_feature, dim=1)[0]], dim=1)
            return self.proj(self.fc(self.dropout(pooled))) * 2.0 + 3

    return Generator()
