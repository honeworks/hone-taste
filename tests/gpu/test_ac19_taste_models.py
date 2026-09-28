"""AC-19 [real, optional]: the smallest Skywork-Reward-V2 and SongEval on a short generated WAV.

Skipped with a reason when the extras are missing or weights cannot be downloaded.
"""

import math
import struct
import wave
from pathlib import Path

import pytest

import hone_taste as tt

pytestmark = pytest.mark.gpu
HUMAN = [
    "My grandmother kept her buttons in a biscuit tin that had long since stopped smelling of biscuits. On "
    "wet Sundays she'd tip them out on the rug and let me sort them by colour, then by size, then by which "
    "ones she said came off my grandfather's army coat, which I now suspect was all of them.",
    "Honestly the bus was twenty minutes late again and by the time I got to work the coffee machine had "
    "broken, so I spent the morning drinking tea that tasted faintly of the soup someone microwaved in the "
    "kettle area yesterday.",
]
MACHINE = [
    "Artificial intelligence is transforming industries across the globe. By leveraging advanced algorithms "
    "and vast amounts of data, businesses can streamline operations, enhance decision-making, and deliver "
    "personalized experiences to customers. However, it is essential to address ethical considerations.",
    "Regular exercise offers numerous benefits for both physical and mental health. It helps maintain a "
    "healthy weight, strengthens the cardiovascular system, and reduces the risk of chronic diseases. "
    "Additionally, physical activity can boost mood, reduce stress, and improve overall well-being.",
]
CHECKPOINT = Path(__file__).parents[2] / ".hone" / "taste" / "models" / "songeval" / "model.safetensors"


def _unavailable(result: tt.Score) -> bool:
    text = result.error.lower()
    return any(word in text for word in ("connection", "resolve", "download", "401", "403", "404", "offline"))


def test_ac19_reward_model(gpu_lock: None) -> None:
    try:
        scorer = tt.reward_model()
    except tt.errors.MissingExtra as exc:
        pytest.skip(str(exc))
    prompt = "What is the capital of France? Answer in one sentence."
    try:
        good = scorer({"prompt": prompt, "response": "The capital of France is Paris."})
        bad = scorer({"prompt": prompt, "response": "Bananas are yellow because of the moon."})
    finally:
        scorer.close()  # free the shared GPU even when an assertion below fails
    if good.value is None and _unavailable(good):
        pytest.skip(f"weights not downloadable: {good.error}")
    assert good.details["model_id"] == "Skywork/Skywork-Reward-V2-Qwen3-0.6B"
    assert good.value is not None, good.error
    assert bad.value is not None, bad.error
    assert 0.0 <= bad.value <= good.value <= 1.0
    assert good.details["raw"] > bad.details["raw"]


def _melody(path: Path, seconds: float = 8.0, rate: int = 24000) -> None:
    notes = [261.63, 293.66, 329.63, 392.0, 440.0, 392.0, 329.63, 293.66]
    frames = bytearray()
    for i in range(int(seconds * rate)):
        freq = notes[int(i / rate * 2) % len(notes)]
        phase = 2 * math.pi * freq * i / rate
        sample = 0.3 * math.sin(phase) + 0.1 * math.sin(2 * phase)
        frames += struct.pack("<h", int(sample * 32767))
    with wave.open(str(path), "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(rate)
        out.writeframes(bytes(frames))


def test_ac19_songeval(gpu_lock: None, tmp_path: Path) -> None:
    try:
        # research test run (see the license note in the README); the checkpoint is kept between runs
        scorer = tt.songeval(accept_license=True, checkpoint=CHECKPOINT)
    except tt.errors.MissingExtra as exc:
        pytest.skip(str(exc))
    song = tmp_path / "melody.wav"
    _melody(song)
    try:
        result = scorer(song)
    finally:
        scorer.close()
    if result.value is None and _unavailable(result):
        pytest.skip(f"weights not downloadable: {result.error}")
    assert result.value is not None, result.error
    assert 0.0 <= result.value <= 1.0
    dimensions = {"coherence", "musicality", "memorability", "clarity", "naturalness"}
    assert set(result.details["dimensions"]) == dimensions
    assert all(0.0 <= v <= 1.0 for v in result.details["dimensions"].values())


def test_audiobox_real(gpu_lock: None, tmp_path: Path) -> None:
    try:
        scorer = tt.audiobox()
    except tt.errors.MissingExtra as exc:
        pytest.skip(str(exc))
    song = tmp_path / "melody.wav"
    _melody(song)
    try:
        result = scorer(song)
    finally:
        scorer.close()
    if result.value is None and _unavailable(result):
        pytest.skip(f"weights not downloadable: {result.error}")
    assert result.value is not None, result.error
    assert set(result.details["dimensions"]) == {"CE", "CU", "PC", "PQ"}
    assert 0.0 <= result.value <= 1.0


def test_binoculars_real(gpu_lock: None) -> None:
    try:
        scorer = tt.binoculars()
    except tt.errors.MissingExtra as exc:
        pytest.skip(str(exc))
    try:
        human = [scorer(text) for text in HUMAN]
        machine = [scorer(text) for text in MACHINE]
    finally:
        scorer.close()
    if human[0].value is None and _unavailable(human[0]):
        pytest.skip(f"weights not downloadable: {human[0].error}")
    assert all(s.value is not None and 0.0 <= s.value <= 1.0 for s in human + machine)
    # tolerant: on average the human texts look more human-written
    assert sum(s.details["raw_score"] for s in human) > sum(s.details["raw_score"] for s in machine)
