"""character_consistency with a fake embedder (design/changes/0003)."""

import json

import pytest

import hone_taste as tt
from hone_taste.scorers.character import cosine
from hone_taste.testing import FakeEmbedder, FakeGpuLease

VECTORS = {
    "sheet.png#0": [1.0, 0.0, 0.0],
    "sheet.png#1": [0.0, 1.0, 0.0],
    "sheet.png#2": [0.0, 0.0, 1.0],
    "side.png": [0.6, 0.8, 0.0],
    "pose.png": [0.1, 0.99, 0.0],
    "stranger.png": [-1.0, 0.0, 0.0],
}


def test_closest_panel_of_a_split_sheet() -> None:
    fake = FakeEmbedder(VECTORS)
    scorer = tt.character_consistency("sheet.png", panels=3, model=fake)
    result = scorer("pose.png")
    assert result.details["panel"] == 1
    assert result.details["cosine"] == pytest.approx(cosine([0.0, 1.0, 0.0], [0.1, 0.99, 0.0]))
    assert result.value == pytest.approx((result.details["cosine"] - 0.25) / 0.75)
    assert result.details["floor"] == 0.25
    assert "not identity" in result.details["caveat"]
    assert result.details["model_id"] == "facebook/dinov2-small"
    assert result.details["license"] == "Apache-2.0"
    assert scorer.name == "character_consistency"
    assert scorer.family == "taste_model"
    assert scorer("stranger.png").value == 0.0  # below the floor: clamped, never negative

    scorer("pose.png")
    reference_calls = [c for c in fake.calls if c[0] == "sheet.png"]
    assert reference_calls == [("sheet.png", (0, 3)), ("sheet.png", (1, 3)), ("sheet.png", (2, 3))]  # once


def test_several_references_and_a_custom_floor() -> None:
    scorer = tt.character_consistency(
        ["sheet.png", "side.png"],
        panels=1,
        floor=0.0,
        model=FakeEmbedder({"sheet.png": [1.0, 0.0], "side.png": [0.0, 1.0], "c.png": [0.0, 1.0]}),
    )
    result = scorer("c.png")
    assert result.details["panel"] == 1
    assert result.value == pytest.approx(1.0)


def test_lazy_load_inside_the_lease_and_close() -> None:
    fake = FakeEmbedder(VECTORS)
    lease = FakeGpuLease()
    scorer = tt.character_consistency("sheet.png", panels=3, model=fake, gpu=lease)
    assert fake.events == []
    scorer("pose.png")
    assert fake.events == ["load"]
    assert lease.events == ["lease dinov2_small", "release dinov2_small"]
    scorer.close()
    assert fake.events == ["load", "unload"]


def test_unknown_image_is_an_error_score_and_recorded() -> None:
    sink = tt.MemorySink()
    with tt.recording(sink):
        result = tt.character_consistency("sheet.png", panels=3, model=FakeEmbedder(VECTORS))("missing.png")
    assert result.value is None
    assert "KeyError" in result.error
    ok = tt.character_consistency("sheet.png", panels=3, model=FakeEmbedder(VECTORS))
    with tt.recording(sink):
        ok("pose.png")
    attributes = sink.spans[-1]["attributes"]
    assert attributes["hone.taste.scorer"] == "character_consistency"
    assert attributes["hone.taste.model_id"] == "facebook/dinov2-small"
    assert attributes["hone.taste.raw"] == pytest.approx(
        json.loads(attributes["hone.taste.details"])["cosine"]
    )


@pytest.mark.parametrize(
    "kwargs", [{"reference": []}, {"reference": "a.png", "panels": 0}, {"reference": "a.png", "floor": 1.0}]
)
def test_config_errors(kwargs: dict) -> None:
    with pytest.raises(tt.errors.ConfigError):
        tt.character_consistency(model=FakeEmbedder({}), **kwargs)


def test_cosine() -> None:
    assert cosine([0.0, 0.0], [1.0, 1.0]) == 0.0
    with pytest.raises(ValueError, match="length"):
        cosine([1.0], [1.0, 2.0])


def test_fake_embedder() -> None:
    fake = FakeEmbedder(lambda path, part: [float(part[0]), 1.0])
    with pytest.raises(RuntimeError, match="before load"):
        fake.embed("a.png")
    fake.load()
    assert fake.embed("a.png", (2, 3)) == [2.0, 1.0]
