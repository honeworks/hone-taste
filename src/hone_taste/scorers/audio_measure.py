"""Objective measurements of a song on the CPU (numpy, scipy, librosa; extra `songs`).

`measure(path)` returns an `AudioReport`: loudness (ITU-R BS.1770 integrated LUFS, EBU loudness range),
true peak (4x oversampled), clipping, harshness (2-5 kHz) and sibilance (5-10 kHz) against the body of the
mix (250 Hz-2 kHz), muddiness (200-500 Hz), spectral centroid, glitches (noise bursts and clicks from
spectral-flux spikes), tempo, key and a rough meter. Ported from OneShotStudio's audio checks
(design/changes/0005); `audio_checks.py` turns a report into pass / fail checks.

Imported only by `audio_checks` / `audio_report` when they run, so the core never imports numpy.
"""

# scipy and librosa ship no type information; the numeric code below is checked by its tests instead.
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false
# pyright: reportMissingTypeStubs=false, reportUnknownParameterType=false, reportMissingParameterType=false

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from importlib import import_module
from pathlib import Path
from typing import Any

import numpy as np
from scipy import signal as sps

NOTES = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]
MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])


@dataclass
class AudioReport:
    duration_s: float
    rate: int
    integrated_lufs: float
    loudness_range_lu: float
    true_peak_dbtp: float
    clipped_share: float
    harsh_ratio: float
    sibilance_ratio: float
    mud_share: float
    centroid_hz: float
    glitches_per_min: float
    glitch_times: list[float] = field(default_factory=list)
    tempo_bpm: float = 0.0
    key: str = ""
    meter_family: str = ""  # "3" (waltz-like) or "4" (duple / quadruple); rough

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def load(path: str | Path, rate: int = 44100) -> tuple[np.ndarray, int]:
    """(channels, samples) float32 at `rate`."""
    librosa = import_module("librosa")  # heavy: only when audio is analysed

    audio, sr = librosa.load(str(path), sr=rate, mono=False)
    audio = np.atleast_2d(np.asarray(audio, dtype=np.float32))
    return audio, int(sr)


def measure(path: str | Path, *, with_music: bool = True) -> AudioReport:
    """All measurements of one audio file. `with_music=False` skips tempo, key and meter (faster)."""
    audio, rate = load(path)
    return measure_array(audio, rate, with_music=with_music)


def measure_array(audio: np.ndarray, rate: int, *, with_music: bool = True) -> AudioReport:
    audio = np.atleast_2d(audio)
    mono = audio.mean(axis=0)
    bands = band_ratios(mono, rate)
    glitch_times = glitches(mono, rate)
    duration = audio.shape[1] / rate
    report = AudioReport(
        duration_s=round(duration, 3),
        rate=rate,
        integrated_lufs=round(integrated_loudness(audio, rate), 2),
        loudness_range_lu=round(loudness_range(audio, rate), 2),
        true_peak_dbtp=round(true_peak(audio), 2),
        clipped_share=float(clipped_share(audio)),
        glitches_per_min=round(len(glitch_times) / max(duration / 60, 1e-6), 2),
        glitch_times=glitch_times,
        harsh_ratio=bands["harsh_ratio"],
        sibilance_ratio=bands["sibilance_ratio"],
        mud_share=bands["mud_share"],
        centroid_hz=bands["centroid_hz"],
    )
    if with_music and duration >= 4:
        report.tempo_bpm, report.key, report.meter_family = music_features(mono, rate)
    return report


# -- loudness (ITU-R BS.1770-4, EBU Tech 3342) -------------------------------------------------------------


def k_weighted(audio: np.ndarray, rate: int) -> np.ndarray:
    """The K-weighting pre-filter (high shelf + high pass), designed for any sample rate."""
    # Stage 1: high shelf (pyloudnorm's parameterisation of the BS.1770 filter)
    gain, q, fc = 3.99984385397, 0.7071752369554193, 1681.9744509555319
    a_gain = 10 ** (gain / 40)
    w0 = 2 * math.pi * fc / rate
    alpha = math.sin(w0) / (2 * q)
    cos = math.cos(w0)
    b1 = [
        a_gain * ((a_gain + 1) + (a_gain - 1) * cos + 2 * math.sqrt(a_gain) * alpha),
        -2 * a_gain * ((a_gain - 1) + (a_gain + 1) * cos),
        a_gain * ((a_gain + 1) + (a_gain - 1) * cos - 2 * math.sqrt(a_gain) * alpha),
    ]
    a1 = [
        (a_gain + 1) - (a_gain - 1) * cos + 2 * math.sqrt(a_gain) * alpha,
        2 * ((a_gain - 1) - (a_gain + 1) * cos),
        (a_gain + 1) - (a_gain - 1) * cos - 2 * math.sqrt(a_gain) * alpha,
    ]
    # Stage 2: high pass at ~38 Hz
    q2, fc2 = 0.5003270373253953, 38.13547087613982
    w2 = 2 * math.pi * fc2 / rate
    alpha2 = math.sin(w2) / (2 * q2)
    cos2 = math.cos(w2)
    b2 = [(1 + cos2) / 2, -(1 + cos2), (1 + cos2) / 2]
    a2 = [1 + alpha2, -2 * cos2, 1 - alpha2]
    out = sps.lfilter(b1, a1, audio, axis=-1)
    return np.asarray(sps.lfilter(b2, a2, out, axis=-1))


def _block_loudness(audio: np.ndarray, rate: int, block_s: float, step_s: float) -> np.ndarray:
    weighted = k_weighted(np.atleast_2d(audio), rate)
    size, step = int(block_s * rate), max(1, int(step_s * rate))
    if weighted.shape[1] < size:
        return np.array([])
    starts = range(0, weighted.shape[1] - size + 1, step)
    power = np.array([np.mean(weighted[:, s : s + size] ** 2, axis=1).sum() for s in starts])
    return -0.691 + 10 * np.log10(power + 1e-12)


def integrated_loudness(audio: np.ndarray, rate: int) -> float:
    """Gated integrated loudness in LUFS (-70 LUFS absolute gate, -10 LU relative gate)."""
    blocks = _block_loudness(audio, rate, 0.4, 0.1)
    blocks = blocks[blocks > -70]
    if blocks.size == 0:
        return -70.0
    relative = 10 * np.log10(np.mean(10 ** (blocks / 10))) - 10
    kept = blocks[blocks > relative]
    return float(10 * np.log10(np.mean(10 ** (kept / 10))))


def loudness_range(audio: np.ndarray, rate: int) -> float:
    """EBU loudness range in LU: 95th minus 10th percentile of gated 3 s short-term loudness."""
    blocks = _block_loudness(audio, rate, 3.0, 0.5)
    blocks = blocks[blocks > -70]
    if blocks.size < 2:
        return 0.0
    relative = 10 * np.log10(np.mean(10 ** (blocks / 10))) - 20
    kept = blocks[blocks > relative]
    return float(np.percentile(kept, 95) - np.percentile(kept, 10))


def true_peak(audio: np.ndarray) -> float:
    """dBTP: the sample peak after 4x oversampling."""
    up = sps.resample_poly(np.atleast_2d(audio), 4, 1, axis=-1)
    return float(20 * np.log10(np.max(np.abs(up)) + 1e-12))


def clipped_share(audio: np.ndarray, level: float = 0.999, run: int = 3) -> float:
    """Share of samples that sit in runs of `run` or more consecutive samples at full scale."""
    hits = 0
    for channel in np.atleast_2d(audio):
        at_top = np.abs(channel) >= level
        if not at_top.any():
            continue
        edges = np.flatnonzero(np.diff(np.concatenate(([0], at_top.view(np.int8), [0]))))
        lengths = edges[1::2] - edges[::2]
        hits += int(lengths[lengths >= run].sum())
    return hits / max(1, np.atleast_2d(audio).size)


# -- spectrum ------------------------------------------------------------------------------------------


def _spectrogram(mono: np.ndarray, rate: int, size: int = 2048) -> tuple[np.ndarray, np.ndarray]:
    freqs, _, spec = sps.stft(mono, fs=rate, nperseg=size, noverlap=size * 3 // 4, boundary=None)  # pyright: ignore[reportArgumentType]
    return freqs, np.abs(spec) ** 2


def band_ratios(mono: np.ndarray, rate: int) -> dict[str, float]:
    """Harshness, sibilance, mud and centroid over 1 s windows (loud windows only)."""
    freqs, power = _spectrogram(mono, rate)
    frames_per_s = max(1, int(rate / 512))

    def band(lo: float, hi: float) -> np.ndarray:
        return power[(freqs >= lo) & (freqs < hi)].sum(axis=0)

    body, harsh, sib = band(250, 2000), band(2000, 5000), band(5000, 10000)
    mud, full = band(200, 500), band(100, 10000)
    n = power.shape[1] // frames_per_s
    if n == 0:
        return {"harsh_ratio": 0.0, "sibilance_ratio": 0.0, "mud_share": 0.0, "centroid_hz": 0.0}

    def windows(x: np.ndarray) -> np.ndarray:
        return x[: n * frames_per_s].reshape(n, frames_per_s).sum(axis=1)

    wb, wh, ws, wm, wf = (windows(x) for x in (body, harsh, sib, mud, full))
    loud = wf > np.max(wf) * 1e-3  # ignore silence
    if not loud.any():
        return {"harsh_ratio": 0.0, "sibilance_ratio": 0.0, "mud_share": 0.0, "centroid_hz": 0.0}
    total = power.sum(axis=1)
    centroid = float((freqs * total).sum() / max(total.sum(), 1e-12))
    return {
        "harsh_ratio": round(float(np.percentile(wh[loud] / (wb[loud] + 1e-12), 90)), 4),
        "sibilance_ratio": round(float(np.percentile(ws[loud] / (wb[loud] + 1e-12), 90)), 4),
        "mud_share": round(float(np.median(wm[loud] / (wf[loud] + 1e-12))), 4),
        "centroid_hz": round(centroid, 1),
    }


def glitches(mono: np.ndarray, rate: int, *, spike: float = 12.0) -> list[float]:
    """Times (s) of noise bursts and clicks: broadband spectral-flux spikes far above the song's usual
    onsets (robust z-score over median / MAD) whose frame is noise-like (high spectral flatness)."""
    _, power = _spectrogram(mono, rate, size=1024)
    if power.shape[1] < 8:
        return []
    log = np.log1p(power / (power.mean() + 1e-12))
    flux = np.maximum(0, np.diff(log, axis=1)).sum(axis=0)
    median = np.median(flux)
    mad = np.median(np.abs(flux - median)) + 1e-9
    z = (flux - median) / (1.4826 * mad)
    flat = np.exp(np.mean(np.log(power + 1e-12), axis=0)) / (np.mean(power, axis=0) + 1e-12)
    hop_s = 256 / rate
    times: list[float] = []
    for k in np.flatnonzero(z > spike):
        t = round((k + 1) * hop_s, 2)
        if flat[k + 1] > 0.3 and (not times or t - times[-1] > 0.25):
            times.append(t)
    return times


# -- tempo, key, meter ---------------------------------------------------------------------------------


def music_features(mono: np.ndarray, rate: int) -> tuple[float, str, str]:
    librosa = import_module("librosa")

    y = librosa.resample(mono, orig_sr=rate, target_sr=22050) if rate != 22050 else mono
    tempo, beats = librosa.beat.beat_track(y=y, sr=22050)
    bpm = float(np.atleast_1d(tempo)[0])
    return round(bpm, 1), key_of(y, 22050), meter_family(y, 22050, beats)


def key_of(y: np.ndarray, rate: int) -> str:
    """Krumhansl-Schmuckler key estimate from the mean chroma, e.g. 'D minor'."""
    librosa = import_module("librosa")

    chroma = librosa.feature.chroma_cqt(y=y, sr=rate).mean(axis=1)
    best = max(
        (float(np.corrcoef(np.roll(profile, k), chroma)[0, 1]), f"{NOTES[k]} {mode}")
        for mode, profile in (("major", MAJOR), ("minor", MINOR))
        for k in range(12)
    )
    return best[1]


def meter_family(y: np.ndarray, rate: int, beats: np.ndarray) -> str:
    """'3' when beat-level accents repeat every 3 beats more than every 4 (waltz-like), else '4'. Rough."""
    librosa = import_module("librosa")

    if len(beats) < 16:
        return ""
    onset = librosa.onset.onset_strength(y=y, sr=rate)
    accents = onset[np.clip(beats, 0, len(onset) - 1)]
    accents = accents - accents.mean()

    def corr(lag: int) -> float:
        return float(np.dot(accents[:-lag], accents[lag:]) / max(1, len(accents) - lag))

    return "3" if corr(3) > corr(4) and corr(3) > corr(2) else "4"
