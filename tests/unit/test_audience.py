import warnings
from pathlib import Path

import pytest

from hone_taste.audience import audience
from hone_taste.errors import ConfigError
from hone_taste.testing import FakeDecisionClient, FakeTextClient


def test_personas_from_toml(tmp_path: Path) -> None:
    path = tmp_path / "panel.toml"
    path.write_text('personas = ["a fan", "  ", "a critic"]\n')
    client = FakeDecisionClient()
    audience(path, "Like it?", client)("x")
    assert len(client.calls) == 2


@pytest.mark.parametrize("content", ["personas = [", "other = 1"])
def test_bad_persona_file(tmp_path: Path, content: str) -> None:
    path = tmp_path / "panel.toml"
    path.write_text(content)
    with pytest.raises(ConfigError):
        audience(path, "Like it?", FakeDecisionClient())


def test_config_errors() -> None:
    with pytest.raises(ConfigError, match="at least one persona"):
        audience([], "q", FakeDecisionClient())
    with pytest.raises(ConfigError, match="scale"):
        audience(["a"], "q", FakeDecisionClient(), scale=(5, 1))
    with pytest.raises(ConfigError, match="unknown aggregate"):
        audience(["a"], "q", FakeDecisionClient(), aggregate="max")
    with pytest.raises(ConfigError, match="decide"):
        audience(["a"], "q", object())  # type: ignore[arg-type]


def test_default_anchors_follow_the_scale() -> None:
    client = FakeDecisionClient()
    audience(["a"], "q", client, scale=(0, 10))("x")
    question = client.calls[0]["questions"]["rating"]
    assert question["anchors"] == {"0": "not at all", "5": "somewhat", "10": "absolutely"}
    assert question["scale"] == [0, 10]


def test_image_input_is_attached(tmp_path: Path) -> None:
    client = FakeDecisionClient()
    audience(["a"], "Would you click?", client)(tmp_path / "thumb.png")
    assert client.calls[0]["images"] == [str(tmp_path / "thumb.png")]


def test_string_image_path_is_attached_but_other_strings_are_text(tmp_path: Path) -> None:
    image = tmp_path / "cover.png"
    image.write_bytes(b"png")
    client = FakeDecisionClient()
    panel = audience(["a"], "Would you click?", client)
    panel(str(image))
    panel("cover.png")  # not an existing file: judged as text
    assert client.calls[0]["images"] == [str(image)]
    assert client.calls[1]["images"] == []
    assert client.calls[1]["state"] == "cover.png"


def test_missing_answer_is_an_error() -> None:
    result = audience(["a"], "q", FakeDecisionClient(answers={}))("x")
    assert result.value is None
    assert result.details["personas"][0]["error"] == "not answered"


def test_text_client_is_wrapped() -> None:
    text = FakeTextClient(['{"rating": {"answer": 5, "rationale": "love it"}}'])
    result = audience(["a"], "q", text)("x")
    assert result.value == 1.0
    assert result.details["personas"][0]["rationale"] == "love it"


def test_same_family_warning() -> None:
    client = FakeDecisionClient()
    client.model = "gemma3:12b"  # type: ignore[attr-defined]
    with pytest.warns(UserWarning, match="same family"):
        panel = audience(["a"], "q", client, generator_model="google/gemma-3-27b-it")
    assert "same family" in panel("x").details["warning"]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        other = audience(["a"], "q", client, generator_model="qwen3:8b")
        assert "warning" not in other("x").details
        audience(["a"], "q", FakeDecisionClient(), generator_model="qwen3:8b")  # judge model unknown
