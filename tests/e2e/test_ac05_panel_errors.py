"""AC-5: a client error for one persona: that persona has an error, the aggregate uses the rest;
value is None only if all personas fail."""

from typing import Any

import pytest

import hone_taste as tt


def flaky(state: Any, name: str, question: Any) -> dict[str, Any]:
    if "critic" in question["instructions"]:
        raise ConnectionError("judge server unreachable")
    if "silent" in question["instructions"]:
        return {"type": "score", "value": None, "error": "refused to answer"}
    return {"type": "score", "value": 0.75, "raw": 4, "rationale": "fine"}


def test_ac5_panel_errors() -> None:
    client = tt.testing.FakeDecisionClient(answers=flaky)
    panel = tt.audience(["a fan", "a critic", "a silent type"], "Like it?", client)
    result = panel("text")

    assert result.value == pytest.approx(0.75)
    by_persona = {p["persona"]: p for p in result.details["personas"]}
    assert by_persona["a critic"]["value"] is None
    assert by_persona["a critic"]["error"] == "ConnectionError: judge server unreachable"
    assert by_persona["a silent type"]["error"] == "refused to answer"
    assert by_persona["a fan"]["error"] == ""
    assert result.details["disagreement"] is None  # only one persona answered
    assert result.reason == "mean of 1/3 personas"

    all_fail = tt.audience(["a critic", "a silent type"], "Like it?", client)("text")
    assert all_fail.value is None
    assert "every persona failed" in all_fail.error
    assert "judge server unreachable" in all_fail.error
