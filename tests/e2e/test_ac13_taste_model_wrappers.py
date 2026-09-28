"""AC-13: taste-model wrappers with FakeTasteModel injected (no heavy deps): loaded lazily, GpuLease
entered around load and inference, close() releases."""

import json
from pathlib import Path
from typing import Any

import pytest

import hone_taste as tt
from hone_taste.testing import FakeGpuLease, FakeTasteModel

SONGEVAL_RAW = {"coherence": 5.0, "musicality": 3.0, "memorability": 1.0, "clarity": 4.0, "naturalness": 2.0}


@pytest.fixture
def reference(tmp_path: Path) -> Path:
    path = tmp_path / "ref.json"
    path.write_text(json.dumps({"good": [10.0, 20.0], "bad": [0.0, 5.0]}))
    return path


def build(name: str, model: FakeTasteModel, gpu: FakeGpuLease, reference: Path) -> Any:
    return {
        "songeval": lambda: tt.songeval(gpu=gpu, model=model, accept_license=True),
        "audiobox": lambda: tt.audiobox(gpu=gpu, model=model),
        "reward_model": lambda: tt.reward_model(gpu=gpu, model=model, reference=reference),
        "pickscore": lambda: tt.image_preference(
            gpu=gpu, model=model, reference=reference, accept_license=True
        ),
        "binoculars": lambda: tt.binoculars(gpu=gpu, model=model),
    }[name]()


CASES = [
    ("songeval", SONGEVAL_RAW, "song.wav"),
    ("audiobox", {"CE": 8.0, "CU": 6.0, "PC": 2.0, "PQ": 7.0}, "song.wav"),
    ("reward_model", {"reward": 12.0}, {"prompt": "Write a haiku.", "response": "Autumn wind..."}),
    ("pickscore", {"score": 15.0}, {"prompt": "a red barn", "image": "barn.png"}),
    ("binoculars", {"score": 0.95, "tokens": 64.0}, "Some text to check."),
]


@pytest.mark.parametrize(("name", "raw", "input"), CASES)
def test_ac13_lazy_load_lease_and_close(name: str, raw: dict, input: object, reference: Path) -> None:
    events: list[str] = []
    model, gpu = FakeTasteModel(raw, events=events), FakeGpuLease(events=events)
    scorer = build(name, model, gpu, reference)
    assert events == []  # nothing loaded when the scorer is created

    first, second = scorer(input), scorer(input)
    lease, release = f"lease {name}", f"release {name}"
    assert events == [
        lease,
        "load",
        "predict",
        release,
        lease,
        "predict",
        release,
    ]  # one load, inside the lease
    assert gpu.calls[0]["vram_gb"] == tt.model_info(name).vram_gb
    assert first == second
    assert first.value is not None
    assert 0.0 <= first.value <= 1.0
    assert first.details["model_id"] == scorer.model_id == tt.model_info(name).model_id
    assert first.details["license"] == scorer.license
    assert first.details["source_data"] == scorer.source_data == tt.model_info(name).source_data
    assert scorer.source_data

    scorer.close()
    assert events[-1] == "unload"
    assert not model.loaded
    scorer.close()  # closing twice is harmless
    assert events.count("unload") == 1
    scorer(input)
    assert events.count("load") == 2  # the next call loads again


def test_ac13_values_follow_the_raw_outputs(reference: Path) -> None:
    weights = {"coherence": 1.0, "memorability": 1.0}
    song = tt.songeval(model=FakeTasteModel(SONGEVAL_RAW), weights=weights, accept_license=True)
    s = song("song.wav")
    assert s.value == 0.5  # (1.0 + 0.0) / 2 on the 1-5 scale
    assert s.details["dimensions"]["musicality"] == 0.5
    assert s.details["raw"] == SONGEVAL_RAW

    box = tt.audiobox(model=FakeTasteModel({"CE": 10.0, "CU": 10.0, "PC": 0.0, "PQ": 10.0}))
    assert box("a.wav").value == 1.0  # production complexity is left out by default

    reward = tt.reward_model(
        model=FakeTasteModel(lambda pair: {"reward": len(pair["response"])}), reference=reference
    )
    short, long = reward({"prompt": "p", "response": "ok"}), reward({"prompt": "p", "response": "x" * 15})
    assert short.value is not None
    assert long.value is not None
    assert short.value < long.value
    assert long.details["raw"] == 15

    detector = tt.binoculars(model=FakeTasteModel(lambda text: {"score": 0.7 if "delve" in text else 1.1}))
    machine, human = detector("Let us delve."), detector("Rain on the tin roof.")
    assert machine.value is not None
    assert human.value is not None
    assert machine.value < 0.5 < human.value
    assert machine.details["threshold"] == pytest.approx(0.9015, abs=1e-4)
    assert "raw_score" in machine.details


def test_ac13_model_failures_are_scores_not_exceptions() -> None:
    def broken(_: object) -> dict[str, float]:
        raise RuntimeError("CUDA out of memory")

    events: list[str] = []
    scorer = tt.songeval(model=FakeTasteModel(broken), gpu=FakeGpuLease(events=events), accept_license=True)
    result = scorer("song.wav")
    assert result.value is None
    assert "CUDA out of memory" in result.error
    assert events[-1] == "release songeval"  # the lease is released on failure
    assert scorer(42).error.startswith("accepts audio")  # wrong input kind: no model call


class FailingLoad(FakeTasteModel):
    def load(self) -> None:
        self.events.append("load")
        raise OSError("download failed")


def test_ac13_load_failure_is_a_score_and_is_retried() -> None:
    events: list[str] = []
    scorer = tt.audiobox(model=FailingLoad(events=events), gpu=FakeGpuLease(events=events))
    first = scorer("a.wav")
    assert first.value is None
    assert "download failed" in first.error
    assert events == ["lease audiobox", "load", "release audiobox"]
    scorer("a.wav")
    assert events.count("load") == 2  # not cached as loaded: the next call tries again
    scorer.close()
    assert "unload" not in events  # nothing was loaded, nothing to free
