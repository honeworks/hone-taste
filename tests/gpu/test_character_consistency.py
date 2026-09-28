"""Real DINOv2-small for character_consistency on generated pictures (design/changes/0003).

Written with the change; runs only with `-m gpu` (it downloads about 90 MB and runs on CPU or GPU).
Skipped with a reason when the extra is missing or the weights cannot be downloaded.
"""

from pathlib import Path

import pytest

import hone_taste as tt

pytestmark = pytest.mark.gpu


def _figure(path: Path, shirt: tuple[int, int, int], hair: tuple[int, int, int], shift: int = 0) -> Path:
    """A simple flat-colour character: head, hair, shirt, on a light background."""
    image_module = pytest.importorskip("PIL.Image")
    draw_module = pytest.importorskip("PIL.ImageDraw")
    picture = image_module.new("RGB", (224, 224), (235, 235, 230))
    draw = draw_module.Draw(picture)
    draw.ellipse((80 + shift, 30, 144 + shift, 94), fill=(230, 190, 160))
    draw.rectangle((78 + shift, 26, 146 + shift, 50), fill=hair)
    draw.rectangle((60 + shift, 100, 164 + shift, 210), fill=shirt)
    picture.save(path)
    return path


def test_real_dinov2_prefers_the_same_character(gpu_lock: None, tmp_path: Path) -> None:
    reference = _figure(tmp_path / "ref.png", (200, 30, 30), (40, 25, 10))
    same = _figure(tmp_path / "same.png", (200, 30, 30), (40, 25, 10), shift=12)
    other = _figure(tmp_path / "other.png", (30, 60, 200), (230, 210, 120))
    try:
        scorer = tt.character_consistency(reference, device="cpu")
    except tt.errors.MissingExtra as exc:
        pytest.skip(str(exc))
    try:
        close, far = scorer(str(same)), scorer(str(other))
    finally:
        scorer.close()
    if close.value is None and any(w in close.error.lower() for w in ("connection", "resolve", "download")):
        pytest.skip(f"weights unavailable: {close.error}")
    assert close.value is not None, close.error
    assert far.value is not None, far.error
    assert close.details["cosine"] > far.details["cosine"]
