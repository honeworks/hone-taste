"""AC-24: song quality. SongEval scores a shorter middle excerpt after an out-of-memory error and says
so; audio_checks scores a clean song above a clipped, harsh, hissy one, listing what failed."""

from pathlib import Path
from typing import Any

import pytest

import hone_taste as tt
from hone_taste.testing import FakeTasteModel

np = pytest.importorskip("numpy")
soundfile = pytest.importorskip("soundfile")
RATE = 44100
DIMENSIONS = ("coherence", "musicality", "memorability", "clarity", "naturalness")


def song(path: Path, *, broken: bool) -> str:
    rng = np.random.default_rng(3)
    t = np.arange(20 * RATE) / RATE
    swell = 0.3 + 0.7 * (t / t[-1])  # a crescendo gives the clean song some dynamics
    mono = sum(np.sin(2 * np.pi * f * t) for f in (440, 554, 660)) * 0.1 * swell
    if broken:
        mono = mono * 12 + np.diff(rng.standard_normal(len(t)), prepend=0) * 3  # clipped and hissy
    soundfile.write(str(path), np.clip(mono, -1, 1).astype(np.float32), RATE)
    return str(path)


def test_ac24_song_quality(tmp_path: Path) -> None:
    def small_gpu(input: Any, **options: Any) -> None:
        if (options.get("max_seconds") or 1e9) > 60:
            raise RuntimeError("CUDA out of memory")

    fake = FakeTasteModel(dict.fromkeys(DIMENSIONS, 4.0))
    predict = fake.predict
    fake.predict = lambda input, **options: small_gpu(input, **options) or predict(input, **options)  # type: ignore[method-assign]
    result = tt.songeval(model=fake, accept_license=True, max_seconds=90)("long-song.wav")
    assert result.value == 0.75
    assert result.details["excerpt_s"] == 45.0

    checks = tt.audio_checks()
    clean, broken = (
        checks(song(tmp_path / "clean.wav", broken=False)),
        checks(song(tmp_path / "bad.wav", broken=True)),
    )
    assert clean.value is not None
    assert broken.value is not None
    assert clean.value > broken.value
    assert {"no_clipping", "not_harsh"} <= set(broken.details["failed"])
    assert clean.details["failed"] == []
