"""Objective audio checks: does this song annoy the ear? No model, CPU only (extra `songs`).

`audio_checks(...)` measures a song (loudness, dynamics, clipping, harshness, sibilance, mud, glitches;
optionally tempo, key and meter against a request) and scores the share of checks passed. The limits are
first guesses from about ten generated songs (OneShotStudio): defaults to calibrate, not truths
(design/changes/0005). The measurements live in `audio_measure.py`, imported only when a song is measured.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from importlib import import_module
from typing import Any

from hone_taste.errors import ConfigError
from hone_taste.registry import require_modules
from hone_taste.types import FunctionScorer, Score

NOTES = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]
SAME = {"Db": "C#", "D#": "Eb", "Gb": "F#", "G#": "Ab", "A#": "Bb", "Cb": "B", "Fb": "E"}

LIMITS: dict[str, float] = {
    "clipped_share_max": 0.0005,  # share of samples in runs of 3+ at full scale
    "harsh_ratio_max": 0.55,  # 2-5 kHz energy / 250 Hz-2 kHz energy, 90th percentile of 1 s windows
    "sibilance_ratio_max": 0.35,  # 5-10 kHz / 250 Hz-2 kHz, 90th percentile
    "mud_share_max": 0.45,  # 200-500 Hz share of 100 Hz-10 kHz energy, median
    "glitches_per_min_max": 3.0,
    "lra_min": 3.0,  # LU: below this the song is squashed flat
    "lra_max": 18.0,
    "tempo_tolerance": 0.04,  # 4 %, after allowing half / double / dotted tempo readings
    "master_lufs": -14.0,
    "master_lufs_tolerance": 1.5,
    "master_true_peak_max": -1.0,  # dBTP
}


def parse_key(text: str) -> tuple[int, str] | None:
    """(pitch class, "major" | "minor") from "D minor", "F#m", "C-sharp minor", "Ab major".

    >>> parse_key("C-sharp minor"), parse_key("Ab major"), parse_key("?")
    ((1, 'minor'), (8, 'major'), None)
    """
    match = re.match(r"\s*([A-Ga-g])\s*-?\s*(sharp|flat|[#b♯♭])?\s*(major|minor|maj|min|m)?", text, re.I)
    if not match:
        return None
    accidental = (match.group(2) or "").lower()
    accidental = {"sharp": "#", "flat": "b", "♯": "#", "♭": "b"}.get(accidental, accidental)
    note = SAME.get(match.group(1).upper() + accidental, match.group(1).upper() + accidental)
    if note not in NOTES:
        return None
    minor = (match.group(3) or "").lower() in ("minor", "min", "m") or "minor" in text.lower()
    return NOTES.index(note), "minor" if minor else "major"


def key_match(asked: str, heard: str) -> str:
    """ "exact", "relative" (same notes: C major / A minor, a common estimator confusion), "other" or
    "unknown" (either key unreadable)."""
    a, h = parse_key(asked), parse_key(heard)
    if a is None or h is None:
        return "unknown"
    if a == h:
        return "exact"
    shift = 9 if a[1] == "major" else 3  # the relative minor is 9 semitones above the major tonic
    return "relative" if a[1] != h[1] and (a[0] + shift) % 12 == h[0] else "other"


def tempo_match(asked: float, heard: float, tolerance: float = LIMITS["tempo_tolerance"]) -> bool:
    """True when the measured tempo is the asked one or its half, double or dotted reading."""
    return heard > 0 and any(abs(heard / (asked * f) - 1) <= tolerance for f in (1, 0.5, 2, 1.5, 2 / 3, 3))


def meter_match(asked: str, family: str) -> bool:
    """Lenient: only a clear waltz (3/4) versus a clear 4/4 request can fail."""
    top = asked.split("/", maxsplit=1)[0]
    if not family or top not in ("2", "3", "4"):
        return True  # unknown, or a compound meter (6/8, 12/8) that reads either way
    return family == ("3" if top == "3" else "4")


def checks(
    report: Mapping[str, Any], limits: Mapping[str, float], request: Mapping[str, Any]
) -> dict[str, bool]:
    """Named pass / fail checks of one measured song. `request` may hold `bpm`, `key`, `meter` and
    `mastered` (adds the delivery checks: loudness near -14 LUFS, true peak below -1 dBTP)."""
    result = {
        "no_clipping": report["clipped_share"] <= limits["clipped_share_max"],
        "not_harsh": report["harsh_ratio"] <= limits["harsh_ratio_max"],
        "not_sibilant": report["sibilance_ratio"] <= limits["sibilance_ratio_max"],
        "not_muddy": report["mud_share"] <= limits["mud_share_max"],
        "no_glitches": report["glitches_per_min"] <= limits["glitches_per_min_max"],
        "dynamics": limits["lra_min"] <= report["loudness_range_lu"] <= limits["lra_max"],
    }
    if request.get("bpm") and report["tempo_bpm"]:
        result["tempo"] = tempo_match(request["bpm"], report["tempo_bpm"], limits["tempo_tolerance"])
    if request.get("key") and report["key"]:
        result["key"] = key_match(request["key"], report["key"]) != "other"
    if request.get("meter") and report["meter_family"]:
        result["meter"] = meter_match(request["meter"], report["meter_family"])
    if request.get("mastered"):
        result["loudness"] = (
            abs(report["integrated_lufs"] - limits["master_lufs"]) <= limits["master_lufs_tolerance"]
        )
        result["true_peak"] = report["true_peak_dbtp"] <= limits["master_true_peak_max"] + 0.05
    return result


def audio_report(path: str, *, with_music: bool = True) -> dict[str, Any]:
    """Every measurement of one audio file as a dict (`with_music=False` skips tempo, key and meter)."""
    require_modules("songs", "numpy", "scipy", "librosa")
    measure = import_module("hone_taste.scorers.audio_measure")  # numpy / scipy only when measuring
    report: dict[str, Any] = measure.measure(path, with_music=with_music).as_dict()
    return report


def audio_checks(
    limits: Mapping[str, float] | None = None,
    *,
    bpm: float | None = None,
    key: str | None = None,
    meter: str | None = None,
    mastered: bool = False,
) -> FunctionScorer:
    """Score an audio path by the share of objective checks it passes (extra `songs`, CPU, no model).

    `limits` override entries of `LIMITS`; `bpm`, `key` ("D minor") and `meter` ("3/4") add checks
    against a request; `mastered=True` adds the delivery checks. `details`: every `measurement`, each
    check's verdict, the `failed` names and the limits used. About 3 s per song.
    """
    unknown = sorted(set(limits or {}) - set(LIMITS))
    if unknown:
        raise ConfigError(f"unknown audio limits {unknown}; known: {sorted(LIMITS)}")
    used = {**LIMITS, **(limits or {})}
    request = {"bpm": bpm, "key": key, "meter": meter, "mastered": mastered}
    require_modules("songs", "numpy", "scipy", "librosa")
    with_music = bool(bpm or key or meter)

    def score(path: Any) -> Score:
        report = audio_report(str(path), with_music=with_music)
        verdicts = checks(report, used, request)
        failed = [name for name, ok in verdicts.items() if not ok]
        reason = f"failed: {', '.join(failed)}" if failed else f"all {len(verdicts)} checks passed"
        details = {"measurements": report, "checks": verdicts, "failed": failed, "limits": used}
        return Score(sum(verdicts.values()) / len(verdicts), reason=reason, details=details)

    return FunctionScorer(
        "audio_checks",
        "taste_model",
        frozenset({"audio"}),
        score,
        license="Apache-2.0",
        source_data="no model, no ratings: signal measurements (BS.1770 loudness, spectral band ratios)",
    )
