"""AC-4: audience panel with FakeDecisionClient: one question per persona, anchors passed,
aggregate + disagreement + rationales in details."""

from typing import Any

import pytest

import hone_taste as tt

PERSONAS = [
    "A 25-year-old who streams blues-rock and skips songs within 20 seconds",
    "A 55-year-old live-blues fan who hates cliches",
    "A producer judging whether the hook works in a short video",
]
RAW = {PERSONAS[0]: 5, PERSONAS[1]: 2, PERSONAS[2]: 4}


def rate(state: Any, name: str, question: Any) -> dict[str, Any]:
    persona = next(p for p in PERSONAS if p in question["instructions"])
    raw = RAW[persona]
    return {"type": "score", "value": (raw - 1) / 4, "raw": raw, "rationale": f"{persona[:12]} says {raw}"}


def test_ac4_audience_panel() -> None:
    client = tt.testing.FakeDecisionClient(answers=rate)
    panel = tt.audience(
        PERSONAS,
        "Would you keep listening past the first chorus?",
        client,
        anchors={"1": "skip", "5": "replay"},
    )
    lyric = "My baby left on the 5:15, took the dog and the good guitar"
    result = panel(lyric)

    assert len(client.calls) == 3  # one decide() per persona
    for call, persona in zip(client.calls, PERSONAS, strict=True):
        assert call["state"] == lyric
        (question,) = call["questions"].values()
        assert question["type"] == "score"
        assert persona in question["instructions"]
        assert "first chorus" in question["instructions"]
        assert question["scale"] == [1, 5]
        assert question["anchors"] == {"1": "skip", "5": "replay"}

    assert result.value == pytest.approx((1.0 + 0.25 + 0.75) / 3)
    personas = result.details["personas"]
    assert [p["persona"] for p in personas] == PERSONAS
    assert [p["raw"] for p in personas] == [5, 2, 4]
    assert all(p["rationale"].endswith(f"says {p['raw']}") for p in personas)
    assert result.details["disagreement"] == pytest.approx(0.3118, abs=1e-4)  # population stdev
    assert result.details["caveat"]

    lowest = tt.audience(PERSONAS, "Keep listening?", client, aggregate="min")(lyric)
    assert lowest.value == pytest.approx(0.25)
    assert tt.audience(PERSONAS, "Keep listening?", client, aggregate="median")(lyric).value == pytest.approx(
        0.75
    )
