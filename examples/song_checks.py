"""Song quality: objective audio checks, and SongEval on long songs without running out of memory.

What: `tt.audio_checks()` measures a song on the CPU, with no model (loudness, dynamics, clipping,
      harshness, sibilance, mud, glitches; tempo, key and meter against a request) and scores the share
      of checks passed. `details` hold every measurement and verdict. `tt.songeval(max_seconds=...)`
      scores a middle excerpt of a long song, and after an out-of-memory error it retries on a shorter
      one. Here two short songs (one clean, one overdriven) are synthesised with the standard library,
      and SongEval's model is a small stand-in backend that runs out of memory on long input.
How:  1. `checks = tt.audio_checks(bpm=..., key=..., meter=..., mastered=...)`, all optional,
      2. `checks(path)`: read `value`, `details["failed"]` and `details["measurements"]`,
      3. `tt.audio_report(path)` gives only the measurements,
      4. `tt.songeval(accept_license=True, max_seconds=90)` for long songs on a small GPU;
         `details["excerpt_s"]` says what was scored (without `max_seconds`: the whole song, once).
Why:  "does this song annoy the ear?" has objective parts that no taste model reports: a clipped or
      harsh master can still get a good aesthetic score. The limits are first guesses from about ten
      generated songs (`hone_taste.scorers.audio_checks.LIMITS`): calibrate them on your own songs. SongEval
      on a whole four-minute song ran out of memory on an 8 GB GPU in 3 of 6 takes (design/changes/0005).
      Needs the `songs` extra.

Run: uv run python examples/song_checks.py
"""

import math
import random
import struct
import tempfile
import wave
from pathlib import Path

import hone_taste as tt

RATE = 22050


def write_song(path: Path, *, overdriven: bool) -> str:
    """Ten seconds of a swelling A major chord; the overdriven one clips, with hiss."""
    rng = random.Random(0)  # noqa: S311 - a fixed seed for a sample signal, not security
    frames = bytearray()
    for n in range(10 * RATE):
        t = n / RATE
        swell = 0.3 + 0.7 * t / 10
        x = sum(math.sin(2 * math.pi * f * t) for f in (440, 554, 660)) * 0.1 * swell
        if overdriven:
            x = x * 12 + rng.uniform(-1, 1) * 0.8
        frames += struct.pack("<h", int(max(-1.0, min(1.0, x)) * 32767))
    with wave.open(str(path), "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(RATE)
        out.writeframes(bytes(frames))
    return str(path)


with tempfile.TemporaryDirectory() as folder:
    clean = write_song(Path(folder, "clean.wav"), overdriven=False)
    loud = write_song(Path(folder, "overdriven.wav"), overdriven=True)
    checks = tt.audio_checks()
    good, bad = checks(clean), checks(loud)
    print(f"clean: value={good.value:.2f} {good.reason}")
    print(f"overdriven: value={bad.value:.2f} {bad.reason}")
    report = tt.audio_report(clean, with_music=False)
    print("measurements:", {k: report[k] for k in ("integrated_lufs", "loudness_range_lu", "harsh_ratio")})
    assert (good.value or 0) > (bad.value or 0)
    assert "no_clipping" in bad.details["failed"]


# SongEval on a small GPU: a stand-in backend that runs out of memory on more than 60 seconds of audio.
class SmallGpuSongEval:
    """Any object with load(), predict(path, max_seconds=None) and unload() can back tt.songeval."""

    def __init__(self):
        self.asked = []

    def load(self):
        pass

    def unload(self):
        pass

    def predict(self, input, max_seconds=None):
        self.asked.append(max_seconds)
        if max_seconds is None or max_seconds > 60:
            raise RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")
        return {"coherence": 4.0, "musicality": 3.5, "memorability": 4.5, "clarity": 3.0, "naturalness": 4.0}


backend = SmallGpuSongEval()
result = tt.songeval(model=backend, accept_license=True, max_seconds=90)("four-minute-song.wav")
print(f"songeval: value={result.value:.2f} excerpt_s={result.details['excerpt_s']} tried={backend.asked}")
assert backend.asked == [90, 45.0]  # the middle 90 s, then 45 s (never below 30 s)
assert result.details["excerpt_s"] == 45.0
