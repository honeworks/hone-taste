"""AC-18 (offline half): audience panel through the OpenAI adapter on a fake HTTP transport.

The real-model half runs against Ollama in tests/gpu/test_ac18_audience_ollama.py.
"""

import json
from typing import Any

import httpx2 as httpx  # the HTTP library the openai SDK uses
import openai

import hone_taste as tt
from hone_taste.adapters.openai import OpenAITextClient
from hone_taste.testing import contracts

CLICHED = "Neon lights and whispered dreams, echoes of a broken heart"
FRESH = "Dad's work boots by the door still smell like diesel and rain"


def reply(request: httpx.Request) -> httpx.Response:
    """A scripted model: 'OK' as text; JSON answers when a schema is asked; clichés rate 2, else 5."""
    body = json.loads(request.content)
    prompt = json.dumps(body["messages"])
    fmt = body.get("response_format")
    if fmt is None:
        content = "OK"
    else:
        props = fmt["json_schema"]["schema"].get("properties", {})
        canned = {"ok": True, "q1": True, "q2": "blue", "q3": 4, "rating": 2 if "Neon" in prompt else 5}
        answers = {
            k: canned[k] if k == "ok" else {"answer": canned[k], "rationale": "scripted"} for k in props
        }
        content = json.dumps(answers)
    return httpx.Response(
        200,
        json={
            "id": "c1",
            "object": "chat.completion",
            "created": 0,
            "model": "fake-model",
            "choices": [
                {"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        },
    )


def test_ac18_audience_panel_via_openai_adapter() -> None:
    sdk: Any = openai.OpenAI(
        base_url="http://fake-ollama.test/v1",
        api_key="ollama",
        max_retries=0,
        http_client=httpx.Client(transport=httpx.MockTransport(reply)),
    )
    client = OpenAITextClient("fake-model", client=sdk)
    contracts.check_text_client(client)
    contracts.check_decision_client(tt.TextDecisionClient(client))

    panel = tt.audience(
        ["A 25-year-old who skips songs within 20 seconds", "A 55-year-old blues fan who hates cliches"],
        question="Would you keep listening past the first chorus?",
        client=client,
    )
    cliched, fresh = panel(CLICHED), panel(FRESH)
    assert cliched.value is not None
    assert fresh.value is not None
    assert cliched.value < fresh.value
    assert len(fresh.details["personas"]) == 2
