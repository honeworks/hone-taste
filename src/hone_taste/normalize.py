"""Map raw model outputs (logits, 1-5 ratings ...) to 0-1 using a reference set.

A reference set is a JSON file of raw values the scorer produced on known-good and known-bad examples:
`{"domain": "text", "scorer": "reward_model", "source": "...", "good": [...], "bad": [...]}`.
Package data lives in `hone_taste/references/<domain>/<scorer>.json`; pass `path=` for your own.

Two modes: `linear` (default; straight line between the 5th and 95th percentile of the reference values,
clamped) and `percentile` (share of reference values below the raw value).
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from dataclasses import dataclass
from importlib import resources
from pathlib import Path

from hone_taste.errors import ConfigError
from hone_taste.types import Score


@dataclass(frozen=True, slots=True)
class ReferenceSet:
    """Raw scorer values on known-good and known-bad examples, used to normalize new raw values."""

    domain: str
    name: str
    values: tuple[float, ...]
    source: str = ""

    def __post_init__(self) -> None:
        if len(set(self.values)) < 2:
            raise ConfigError(f"reference set {self.label!r} needs at least two different values")

    @property
    def label(self) -> str:
        return f"{self.domain}/{self.name}"


def reference_set(domain: str, name: str, *, path: str | Path | None = None) -> ReferenceSet:
    """Load a reference set from package data (`references/<domain>/<name>.json`) or from `path`."""
    if path is not None:
        text = Path(path).read_text(encoding="utf-8")
    else:
        resource = resources.files("hone_taste").joinpath("references", domain, f"{name}.json")
        if not resource.is_file():
            raise ConfigError(
                f"no packaged reference set {domain}/{name}; pass reference= (or path=) with a JSON file of "
                "raw values {'good': [...], 'bad': [...]}"
            )
        text = resource.read_text(encoding="utf-8")
    try:
        data = json.loads(text)
        values = tuple(float(v) for v in [*data.get("good", []), *data.get("bad", [])])
    except (ValueError, TypeError, AttributeError) as exc:
        raise ConfigError(f"reference set {domain}/{name} is not valid: {exc}") from exc
    return ReferenceSet(domain, name, values, source=str(data.get("source", "")))


def as_reference(reference: str | Path | ReferenceSet | None, domain: str, name: str) -> ReferenceSet:
    """`reference` itself, the set at that path, or the packaged set `domain/name`."""
    if isinstance(reference, ReferenceSet):
        return reference
    return reference_set(domain, name, path=reference)


def _quantile(sorted_values: list[float], q: float) -> float:
    position = q * (len(sorted_values) - 1)
    low = int(position)
    high = min(low + 1, len(sorted_values) - 1)
    return sorted_values[low] + (sorted_values[high] - sorted_values[low]) * (position - low)


def linear(raw: float, reference: ReferenceSet) -> float:
    """Linear map from [5th, 95th percentile] of the reference values to [0, 1], clamped.

    >>> ref = ReferenceSet("demo", "x", tuple(float(v) for v in range(101)))
    >>> linear(50.0, ref), linear(-10.0, ref), linear(500.0, ref)
    (0.5, 0.0, 1.0)
    """
    values = sorted(reference.values)
    low, high = _quantile(values, 0.05), _quantile(values, 0.95)
    if high == low:  # all but the extreme values are equal
        return 1.0 if raw > low else 0.0 if raw < low else 0.5
    return min(1.0, max(0.0, (raw - low) / (high - low)))


def percentile(raw: float, reference: ReferenceSet) -> float:
    """Share of reference values below `raw` (ties count half).

    >>> percentile(2.0, ReferenceSet("demo", "x", (1.0, 2.0, 3.0, 4.0)))
    0.375
    """
    below = sum(v < raw for v in reference.values)
    equal = sum(v == raw for v in reference.values)
    return (below + 0.5 * equal) / len(reference.values)


MODES: dict[str, Callable[[float, ReferenceSet], float]] = {"linear": linear, "percentile": percentile}


def check_mode(mode: str) -> None:
    if mode not in MODES:
        raise ConfigError(f"unknown normalization mode {mode!r}; choose one of {sorted(MODES)}")


def normalized(raw: float, reference: ReferenceSet, *, mode: str = "linear") -> Score:
    """A `Score` whose value is `raw` mapped to 0-1; the raw value is kept in `details["raw"]`.

    >>> s = normalized(3.0, ReferenceSet("demo", "x", (1.0, 5.0)), mode="percentile")
    >>> s.value, s.details["raw"]
    (0.5, 3.0)
    """
    check_mode(mode)
    details = {"raw": raw, "reference_set": reference.label, "normalization": mode}
    if not math.isfinite(raw):
        return Score(None, details=details, error=f"raw value is not finite: {raw}")
    return Score(MODES[mode](raw, reference), details=details)
