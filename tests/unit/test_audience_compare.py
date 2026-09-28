"""Ceiling flag, pairwise comparison and the spread helper (design/changes/0002)."""

import json
from pathlib import Path
from typing import Any

import pytest

import hone_taste as tt
from hone_taste.testing import FakeDecisionClient

PERSONAS = ["a fan", "a critic", "a producer"]


def top(state: Any, name: str, question: Any) -> dict[str, Any]:
    return {"type": "score", "value": 1.0, "raw": 5, "rationale": "love it"}


def prefers(word: str):
    """A judge that picks whichever of A / B contains `word`, whatever the order."""

    def rule(state: Any, name: str, question: Any) -> dict[str, Any]:
        a_part = state.split("\n\nB:\n")[0]
        choice = "A" if word in a_part else "B"
        return {"type": "choice", "value": 1.0, "choice": choice, "rationale": f"{word} wins"}

    return rule


def test_ceiling_when_every_persona_gives_the_top() -> None:
    assert tt.audience(PERSONAS, "q", FakeDecisionClient(top))("x").details["ceiling"] is True
    assert tt.audience(PERSONAS, "q", FakeDecisionClient())("x").details["ceiling"] is False


def test_no_ceiling_when_every_persona_failed() -> None:
    def fail(state: Any, name: str, question: Any) -> dict[str, Any]:
        return {"type": "score", "value": None, "error": "bad"}

    result = tt.audience(PERSONAS, "q", FakeDecisionClient(fail))("x")
    assert result.value is None
    assert result.details["ceiling"] is False


def test_pairwise_alternates_the_order_and_maps_answers_back() -> None:
    client = FakeDecisionClient(prefers("rain"))
    panel = tt.audience(PERSONAS, "Which chorus is better?", client)
    result = panel.pairwise("rain on the roof", "neon tapestry")
    assert result.value == 1.0
    assert result.details["choice"] == "a"
    assert result.confidence == 1.0
    assert result.details["mode"] == "compare"
    states = [call["state"] for call in client.calls]
    assert states[0].startswith("A:\nrain")
    assert states[1].startswith("A:\nneon")
    assert [p["shown_first"] for p in result.details["personas"]] == ["a", "b", "a"]
    (question,) = client.calls[0]["questions"].values()
    assert question["type"] == "choice"
    assert question["options"] == ["A", "B", "no preference"]
    assert "a fan" in question["instructions"]
    assert "Which chorus" in question["instructions"]

    reverse = panel.pairwise("neon tapestry", "rain on the roof")
    assert reverse.value == 0.0
    assert reverse.details["choice"] == "b"


def test_position_bias_becomes_a_tie() -> None:
    result = tt.audience(PERSONAS[:2], "q", FakeDecisionClient()).pairwise("x", "y")  # the fake says "A"
    assert result.value == 0.5
    assert result.details["choice"] == "tie"
    assert result.confidence == 0.0


def test_no_preference_counts_half() -> None:
    def neutral(state: Any, name: str, question: Any) -> dict[str, Any]:
        return {"type": "choice", "value": 1.0, "choice": "no preference"}

    assert tt.audience(PERSONAS, "q", FakeDecisionClient(neutral)).pairwise("x", "y").value == 0.5


def test_pairwise_errors_per_persona() -> None:
    calls = {"n": 0}

    def flaky(state: Any, name: str, question: Any) -> dict[str, Any]:
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("down")
        if calls["n"] == 2:
            return {"type": "choice", "value": None, "error": "timeout"}
        return {"type": "choice", "value": 1.0, "choice": "A"}

    result = tt.audience(PERSONAS, "q", FakeDecisionClient(flaky)).pairwise("x", "y")
    errors = [p["error"] for p in result.details["personas"]]
    assert errors == ["RuntimeError: down", "timeout", ""]
    assert result.value == 1.0  # persona 3 saw a first and chose A


def test_pairwise_invalid_or_missing_answers_fail_the_comparison() -> None:
    def odd(state: Any, name: str, question: Any) -> dict[str, Any]:
        return {"type": "choice", "value": 1.0, "choice": "C"}

    result = tt.audience(["a"], "q", FakeDecisionClient(odd)).pairwise("x", "y")
    assert result.value is None
    assert "invalid choice 'C'" in result.error
    assert result.details["choice"] is None
    missing = tt.audience(["a"], "q", FakeDecisionClient(answers={})).pairwise("x", "y")
    assert "not answered" in missing.error


def test_pairwise_attaches_two_images(tmp_path: Path) -> None:
    a, b = tmp_path / "a.png", tmp_path / "b.png"
    a.write_bytes(b"x")
    b.write_bytes(b"y")
    client = FakeDecisionClient()
    tt.audience(PERSONAS[:2], "Nicer cover?", client).pairwise(a, b)
    assert client.calls[0]["images"] == [str(a), str(b)]
    assert client.calls[1]["images"] == [str(b), str(a)]
    assert "attached images" in client.calls[0]["state"]


def test_pairwise_checks_input_kinds_and_records_the_mode() -> None:
    panel = tt.audience(["a"], "q", FakeDecisionClient(), generator_model=None)
    sink = tt.MemorySink()
    with tt.recording(sink):
        assert "accepts" in panel.pairwise("x", 3).error
        panel.pairwise("x", "y")
    assert [s["attributes"]["hone.taste.panel.mode"] for s in sink.spans] == ["compare", "compare"]
    assert sink.spans[0]["status"]["code"] == "error"
    assert json.loads(sink.spans[1]["attributes"]["hone.taste.details"])["choice"] == "a"


def test_pairwise_keeps_the_same_family_warning() -> None:
    class Named(FakeDecisionClient):
        model = "qwen2.5vl-7b"

    with pytest.warns(UserWarning, match="same family"):
        panel = tt.audience(["a"], "q", Named(), generator_model="qwen3:8b")
    assert "same family" in panel.pairwise("x", "y").details["warning"]


def test_panel_without_comparison_is_a_config_error() -> None:
    panel = tt.AudiencePanel("audience", "audience", frozenset({"text"}), lambda x: tt.Score(0.5))
    with pytest.raises(tt.errors.ConfigError):
        panel.pairwise("x", "y")


def test_spread() -> None:
    flat = tt.spread([tt.Score(1.0), tt.Score(1.0), tt.Score(1.0)])
    assert (flat.range, flat.flat, flat.at_ceiling, flat.known, flat.unscored) == (0.0, True, True, 3, 0)
    varied = tt.spread([0.25, 0.75, None], tolerance=0.1)
    assert (varied.range, varied.flat, varied.at_ceiling, varied.unscored) == (0.5, False, False, 1)
    single = tt.spread([tt.Score(0.4)])
    assert single.range is None
    assert not single.flat
    assert not single.at_ceiling
    assert not tt.spread([]).at_ceiling


class Candidate:
    def __init__(self, text: Any) -> None:
        self.id, self.data, self.files, self.meta = "c", {"text": text}, {}, {}


def test_select_pairwise_raises_on_failure_and_links_the_trace() -> None:
    def fail(state: Any, name: str, question: Any) -> dict[str, Any]:
        raise RuntimeError("down")

    judge = tt.for_select_pairwise(tt.audience(["a"], "q", FakeDecisionClient(fail)), field="text")
    with pytest.raises(RuntimeError, match="every persona failed"):
        judge(Candidate("x"), Candidate("y"))
    sink = tt.MemorySink()
    ok = tt.for_select_pairwise(tt.audience(["a"], "q", FakeDecisionClient()), field="text")
    with tt.recording(sink):
        ok(Candidate("x"), Candidate("y"), trace={"hone.run_id": "r1"})
    (span,) = sink.spans
    assert span["attributes"]["hone.run_id"] == "r1"
    assert span["attributes"]["hone.scorer"] == "audience_pairwise"
    assert ok.version
