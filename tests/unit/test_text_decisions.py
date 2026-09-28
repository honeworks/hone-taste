import json
from typing import Any

import pytest

from hone_taste.ports import TextResult
from hone_taste.testing import FakeTextClient
from hone_taste.text_decisions import TextDecisionClient

QUESTIONS: dict[str, dict[str, Any]] = {
    "yes": {"type": "yes_no", "instructions": "Is it sad?"},
    "pick": {"type": "choice", "instructions": "Mood?", "options": ["calm", "angry"]},
    "score": {"type": "score", "instructions": "How vivid?", "scale": [0, 10], "anchors": {"0": "flat"}},
}


def test_all_question_types_and_prompt() -> None:
    reply = {
        "yes": {"answer": True, "rationale": "tears"},
        "pick": {"answer": "calm", "rationale": ""},
        "score": {"answer": 7.5, "rationale": "bright"},
    }
    text = FakeTextClient([json.dumps(reply)])
    answers = TextDecisionClient(text, params={"temperature": 0}).decide(
        "It rained.", QUESTIONS, images=["a.png"]
    )

    assert answers["yes"]["value"] == 1.0
    assert answers["yes"]["probabilities"] == {"yes": 1.0, "no": 0.0}
    assert answers["pick"] == {
        "type": "choice",
        "calibrated": False,
        "rationale": None,
        "error": None,
        "value": 1.0,
        "choice": "calm",
    }
    assert answers["score"]["value"] == 0.75
    assert answers["score"]["raw"] == 7.5
    assert answers["score"]["rationale"] == "bright"

    call = text.calls[0]
    assert call["params"] == {"temperature": 0}
    assert call["schema"]["required"] == ["yes", "pick", "score"]
    assert call["schema"]["properties"]["pick"]["properties"]["answer"]["enum"] == ["calm", "angry"]
    content = call["messages"][1]["content"]
    assert content[1] == {"type": "image", "path": "a.png"}
    prompt = content[0]["text"]
    assert "It rained." in prompt
    assert "Options: calm, angry" in prompt
    assert "from 0 to 10" in prompt
    assert "0 = flat" in prompt


@pytest.mark.parametrize(
    ("reply", "error"),
    [
        ({"yes": {"answer": "maybe"}}, "invalid yes_no answer: 'maybe'"),
        ({"yes": {"rationale": "hm"}}, "no answer in the model output"),
        ({}, "no answer in the model output"),
        ({"pick": {"answer": "purple"}}, "invalid choice answer: 'purple'"),
    ],
)
def test_invalid_answers(reply: dict[str, Any], error: str) -> None:
    name = next(iter(reply), "yes")
    answers = TextDecisionClient(FakeTextClient([json.dumps(reply)])).decide("s", {name: QUESTIONS[name]})
    assert answers[name]["value"] is None
    assert answers[name]["error"] == error


class Scripted:
    def __init__(self, result: Any) -> None:
        self.result = result

    def complete(self, messages: Any, **kwargs: Any) -> Any:
        return self.result


@pytest.mark.parametrize(
    ("result", "error"),
    [
        (TextResult(text="", error="context too long"), "context too long"),
        ({"text": "not json"}, "model output is not JSON"),
        ({"text": "[1]"}, "model output is not a JSON object"),
    ],
)
def test_call_level_errors_mark_every_question(result: Any, error: str) -> None:
    answers = TextDecisionClient(Scripted(result)).decide({"lyric": "x"}, QUESTIONS)
    assert {a["error"] for a in answers.values()} == {error}
    assert all(a["value"] is None for a in answers.values())


def test_parsed_is_used_when_given() -> None:
    answers = TextDecisionClient(Scripted({"text": "", "parsed": {"yes": {"answer": False}}})).decide(
        "s", {"yes": QUESTIONS["yes"]}
    )
    assert answers["yes"]["value"] == 0.0
