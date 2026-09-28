"""Metadata of the heavy models hone-taste wraps (`data/models.toml`): whose taste, license, VRAM, extra.

>>> info = model_info("reward_model")
>>> info.license, info.extra
('Apache-2.0', 'text')
>>> sorted(models())[:3]
['audiobox', 'binoculars', 'dinov2_small']
"""

from __future__ import annotations

import importlib.util
import tomllib
from dataclasses import dataclass
from functools import cache
from importlib import resources
from typing import Any

from hone_taste.errors import ConfigError, LicenseNotAccepted, MissingExtra


@dataclass(frozen=True, slots=True)
class ModelInfo:
    """One registry entry. `license_clear=False` means the scorer needs `accept_license=True`."""

    name: str
    model_id: str
    url: str
    license: str
    license_clear: bool
    source_data: str
    vram_gb: float
    extra: str
    checkpoint_url: str = ""


@cache
def models() -> dict[str, ModelInfo]:
    """Every registry entry by name."""
    text = resources.files("hone_taste").joinpath("data", "models.toml").read_text(encoding="utf-8")
    entries: dict[str, dict[str, Any]] = tomllib.loads(text)
    return {name: ModelInfo(name=name, **entry) for name, entry in entries.items()}


def model_info(name: str) -> ModelInfo:
    try:
        return models()[name]
    except KeyError:
        raise ConfigError(f"unknown model {name!r}; known: {sorted(models())}") from None


def check_license(info: ModelInfo, accept_license: bool) -> None:
    """Raise `LicenseNotAccepted` for a model with unclear terms unless the user accepted them."""
    if not info.license_clear and not accept_license:
        raise LicenseNotAccepted(
            f"{info.name} ({info.model_id}) needs accept_license=True because its terms are not clear. "
            f"License: {info.license}. Read the terms at {info.url} first."
        )


def require_modules(extra: str, *modules: str) -> None:
    """Raise `MissingExtra` naming the extra to install when any of `modules` is not importable."""
    missing = [m for m in modules if importlib.util.find_spec(m) is None]
    if missing:
        raise MissingExtra(
            f"{', '.join(missing)} not installed; install the extra: pip install 'hone-taste[{extra}]'"
        )
