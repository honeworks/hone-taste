"""The fakes and the TextClient-backed DecisionClient satisfy the port contract checkers."""

import json
from collections.abc import Mapping
from typing import Any

from hone_taste.testing import FakeDecisionClient, FakeTextClient, contracts
from hone_taste.text_decisions import TextDecisionClient


def test_fake_decision_client() -> None:
    contracts.check_decision_client(FakeDecisionClient())


def test_fake_text_client() -> None:
    contracts.check_text_client(FakeTextClient(['{"ok": true}']))
    contracts.check_text_client(FakeTextClient(["not json"]))  # a schema failure is reported as error


def _reply(messages: Any, schema: Mapping[str, Any] | None) -> str:
    props = (schema or {}).get("properties", {})
    answers = {"q1": {"answer": True}, "q2": {"answer": "blue"}, "q3": {"answer": 4}}
    return json.dumps({k: {**v, "rationale": "r"} for k, v in answers.items() if k in props})


def test_text_decision_client() -> None:
    contracts.check_decision_client(TextDecisionClient(FakeTextClient([_reply])))
