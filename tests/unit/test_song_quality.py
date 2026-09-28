"""SongEval excerpts and out-of-memory retries; the objective audio checks (design/changes/0005)."""

from pathlib import Path
from typing import Any

import pytest

import hone_taste as tt
from hone_taste.scorers.audio_checks import LIMITS, checks, key_match, meter_match, parse_key, tempo_match
from hone_taste.scorers.songeval import DIMENSIONS
from hone_taste.testing import FakeTasteModel

np = pytest.importorskip("numpy")
measure = pytest.importorskip("hone_taste.scorers.audio_measure")
RATE = 44100


class OutOfMemory(RuntimeError):
    pass


def running_out(limit: float):
    """A backend that runs out of memory on anything longer than `limit` seconds (None = whole song)."""
    fake = FakeTasteModel(dict.fromkeys(DIMENSIONS, 4.0))
    original = fake.predict

    def predict(input: Any, **options: Any):
        seconds = options.get("max_seconds")
        if seconds is None or seconds > limit:
            raise OutOfMemory("CUDA out of memory. Tried to allocate 2.00 GiB")
        return original(input, **options)

    fake.predict = predict  # type: ignore[method-assign]
    return fake


def test_songeval_scores_the_whole_song_by_default() -> None:
    fake = FakeTasteModel(dict.fromkeys(DIMENSIONS, 5.0))
    result = tt.songeval(model=fake, accept_license=True)("song.wav")
    assert result.value == 1.0
    assert result.details["excerpt_s"] is None
    assert fake.options == [{}]
    assert "excerpt_s" not in result.details["raw"]


def test_songeval_halves_after_out_of_memory() -> None:
    fake = running_out(50)
    result = tt.songeval(model=fake, accept_license=True, max_seconds=90)("song.wav")
    assert result.details["excerpt_s"] == 45.0  # 90 s -> 45 s
    assert fake.options == [{"max_seconds": 45.0}]
    whole = tt.songeval(model=running_out(50), accept_license=True)("song.wav")  # no limit: as before
    assert whole.value is None
    assert "out of memory" in whole.error


def test_songeval_gives_up_below_the_minimum_and_keeps_other_errors() -> None:
    result = tt.songeval(model=running_out(10), accept_license=True, max_seconds=60)("song.wav")
    assert result.value is None
    assert "out of memory" in result.error

    broken = FakeTasteModel(lambda input: {"coherence": 1.0})  # missing dimensions: not an OOM
    assert "KeyError" in tt.songeval(model=broken, accept_license=True)("song.wav").error
    with pytest.raises(tt.errors.ConfigError, match="at least 30"):
        tt.songeval(model=FakeTasteModel(), accept_license=True, max_seconds=10)


def tone(freqs: list[float], seconds: float = 8.0, amp: float = 0.1):
    t = np.arange(int(seconds * RATE)) / RATE
    mono = sum(np.sin(2 * np.pi * f * t) for f in freqs) * amp
    return np.stack([mono, mono]).astype(np.float32)


def test_loudness_and_true_peak_of_a_known_sine() -> None:
    audio = tone([1000])  # -20 dBFS peak per channel, stereo: about -20 LUFS
    assert measure.integrated_loudness(audio, RATE) == pytest.approx(-20.0, abs=0.6)
    assert measure.true_peak(audio) == pytest.approx(-20.0, abs=0.2)
    assert measure.loudness_range(audio, RATE) < 1.0
    assert measure.integrated_loudness(np.zeros((1, RATE)), RATE) == -70.0  # silence
    assert measure.loudness_range(tone([1000], seconds=2), RATE) == 0.0  # too short for 3 s blocks


def test_clipping_harshness_and_glitches_are_caught() -> None:
    clean = measure.measure_array(tone([700, 1100, 1500]), RATE, with_music=False).as_dict()
    verdicts = checks(clean, LIMITS, {})
    assert [n for n, ok in verdicts.items() if not ok] == ["dynamics"]  # a steady tone has no dynamics

    clipped = np.clip(tone([220], amp=3.0), -1, 1)
    assert measure.clipped_share(clipped) > LIMITS["clipped_share_max"]
    assert measure.clipped_share(tone([220])) == 0.0

    rng = np.random.default_rng(0)
    hiss = np.diff(rng.standard_normal(8 * RATE), prepend=0).astype(np.float32) * 0.05  # tilted up
    assert measure.band_ratios(hiss + tone([200])[0], RATE)["harsh_ratio"] > LIMITS["harsh_ratio_max"]
    assert measure.band_ratios(np.zeros(2 * RATE), RATE)["harsh_ratio"] == 0.0

    clicks = tone([220, 330])[0].copy()
    for at in (2.0, 4.5, 6.0):
        k = int(at * RATE)
        clicks[k : k + 200] += rng.standard_normal(200).astype(np.float32) * 0.9
    assert len(measure.glitches(clicks, RATE)) == 3
    assert measure.glitches(np.zeros(2000), RATE) == []


def test_keys_tempo_and_meter_matching() -> None:
    assert parse_key("F#m") == (6, "minor")
    assert parse_key("H major") is None
    assert key_match("C major", "A minor") == "relative"
    assert key_match("A minor", "C major") == "relative"
    assert key_match("C# minor", "C# major") == "other"
    assert key_match("D minor", "D minor") == "exact"
    assert key_match("?", "D minor") == "unknown"
    assert tempo_match(84, 172.3)
    assert tempo_match(92, 46.2)
    assert not tempo_match(78, 100.0)
    assert not tempo_match(78, 0.0)
    assert meter_match("3/4", "3")
    assert not meter_match("4/4", "3")
    assert meter_match("6/8", "3")
    assert meter_match("3/4", "")


def test_a_minor_triad_is_heard_in_its_key() -> None:
    d, f, a = 146.83, 174.61, 220.0
    heard = measure.key_of(tone([d, f, a], seconds=6)[0], RATE)
    assert key_match("D minor", heard) in ("exact", "relative")


def write_wav(path: Path, audio: Any) -> Path:
    soundfile = pytest.importorskip("soundfile")
    soundfile.write(str(path), audio.T, RATE)
    return path


def test_audio_checks_scorer_on_a_file(tmp_path: Path) -> None:
    beat = tone([220, 330, 440], seconds=8, amp=0.1)
    kick = 0.5 * np.exp(-np.arange(2000) / 300) * np.sin(2 * np.pi * 80 * np.arange(2000) / RATE)
    for k in range(16):  # a kick drum every half second: 120 bpm
        start = int(k * 0.5 * RATE)
        beat[:, start : start + 2000] += kick
    song = write_wav(tmp_path / "song.wav", beat)
    scorer = tt.audio_checks(bpm=120, key="A minor", meter="4/4", mastered=True)
    result = scorer(str(song))
    assert result.value is not None
    names = set(result.details["checks"])
    assert {"no_clipping", "not_harsh", "dynamics", "loudness", "true_peak"} <= names
    assert result.details["checks"]["tempo"] is True
    assert result.details["measurements"]["tempo_bpm"] == pytest.approx(120, rel=0.04)
    assert result.details["failed"] == [n for n, ok in result.details["checks"].items() if not ok]
    assert result.value == pytest.approx(1 - len(result.details["failed"]) / len(names))
    assert result.details["measurements"]["duration_s"] == pytest.approx(8.0, abs=0.01)
    assert "failed:" in result.reason

    report = tt.audio_report(str(song), with_music=False)
    assert report["tempo_bpm"] == 0.0
    assert report["key"] == ""
    assert tt.audio_checks()(str(tmp_path / "missing.wav")).value is None


def test_audio_checks_limits() -> None:
    with pytest.raises(tt.errors.ConfigError, match="unknown audio limits"):
        tt.audio_checks({"loudness_max": 1.0})
    lenient = {**LIMITS, "lra_min": 0.0}
    report = measure.measure_array(tone([700, 1100]), RATE, with_music=False).as_dict()
    assert all(checks(report, lenient, {}).values())
    assert tt.audio_checks({"lra_min": 0.0}).family == "taste_model"
