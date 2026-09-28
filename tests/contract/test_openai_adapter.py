"""The OpenAI adapter on a fake HTTP transport: TextClient contract, DecisionClient via TextDecisionClient."""

import json
import sys
from pathlib import Path
from typing import Any

import httpx2 as httpx  # the HTTP library the openai SDK uses
import openai
import pytest

import hone_taste as tt
from hone_taste.adapters.openai import OpenAITextClient
from hone_taste.errors import MissingExtra, ModelUnavailable
from hone_taste.testing import contracts
from hone_taste.text_decisions import TextDecisionClient

BASE = "http://fake-openai.test/v1"


def completion(content: str, finish_reason: str = "stop") -> dict[str, Any]:
    return {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "created": 0,
        "model": "fake-model",
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": finish_reason}
        ],
        "usage": {"prompt_tokens": 11, "completion_tokens": 3, "total_tokens": 14},
    }


def answer_for(request: httpx.Request) -> httpx.Response:
    body = json.loads(request.content)
    fmt = body.get("response_format")
    if fmt is None:
        return httpx.Response(200, json=completion("OK"))
    props = fmt["json_schema"]["schema"].get("properties", {})
    if "ok" in props:
        return httpx.Response(200, json=completion('```json\n{"ok": true}\n```'))
    canned = {"q1": True, "q2": "blue", "q3": 4, "rating": 2}
    reply = {k: {"answer": canned[k], "rationale": "r"} for k in props}
    return httpx.Response(200, json=completion("<think>hmm</think>" + json.dumps(reply)))


def sdk(handler: Any, sent: list[dict[str, Any]] | None = None) -> openai.OpenAI:
    """An OpenAI SDK client whose HTTP transport is a local function (no network)."""

    def record(request: httpx.Request) -> httpx.Response:
        if sent is not None:
            sent.append(json.loads(request.content))
        return handler(request)

    return openai.OpenAI(
        base_url=BASE,
        api_key="sk-test",
        max_retries=0,
        http_client=httpx.Client(transport=httpx.MockTransport(record)),
    )


@pytest.fixture
def client() -> Any:
    sent: list[dict[str, Any]] = []
    return OpenAITextClient("fake-model", client=sdk(answer_for, sent), temperature=0), sent


def test_text_client_contract(client: Any) -> None:
    text_client, sent = client
    contracts.check_text_client(text_client)
    assert all(body["temperature"] == 0 for body in sent)
    assert "unknown_param_is_ignored" not in sent[2]


def test_decision_client_contract(client: Any) -> None:
    contracts.check_decision_client(TextDecisionClient(client[0]))


def test_result_fields_and_images(client: Any, tmp_path: Path) -> None:
    text_client, sent = client
    image = tmp_path / "cover.png"
    image.write_bytes(b"\x89PNG fake")
    result = text_client.complete(
        [
            {
                "role": "user",
                "content": [{"type": "text", "text": "Describe"}, {"type": "image", "path": str(image)}],
            }
        ],
        seed=3,
    )
    assert (result.text, result.model, result.finish_reason) == ("OK", "fake-model", "stop")
    assert result.usage == {"input_tokens": 11, "output_tokens": 3}
    part = sent[-1]["messages"][0]["content"][1]
    assert part["type"] == "image_url"
    assert part["image_url"]["url"].startswith("data:image/png;base64,")


def test_invalid_json_sets_error() -> None:
    client = sdk(lambda _: httpx.Response(200, json=completion("sure! {oops")))
    result = OpenAITextClient("m", client=client).complete([{"role": "user", "content": "x"}], schema={})
    assert result.parsed is None
    assert result.error == "response is not valid JSON"


def test_transport_errors_raise() -> None:
    client = sdk(lambda _: httpx.Response(404, json={"error": {"message": "no model"}}))
    with pytest.raises(RuntimeError, match="NotFoundError") as info:
        OpenAITextClient("m", client=client).complete([{"role": "user", "content": "x"}])
    assert isinstance(info.value, ModelUnavailable)
    assert isinstance(info.value.__cause__, openai.NotFoundError)


def test_missing_required_keys_set_error() -> None:
    client = sdk(lambda _: httpx.Response(200, json=completion('{"other": 1}')))
    schema = {"type": "object", "required": ["ok"]}
    result = OpenAITextClient("m", client=client).complete([{"role": "user", "content": "x"}], schema=schema)
    assert result.parsed is None
    assert result.error == "response misses required keys ['ok']"


def test_server_error_reaches_panel_as_persona_error() -> None:
    client = sdk(lambda _: httpx.Response(500, json={"error": {"message": "boom"}}))
    panel = tt.audience(["a fan"], "Like it?", OpenAITextClient("m", client=client))
    result = panel("some lyric")
    assert result.value is None
    assert result.details["personas"][0]["error"].startswith("ModelUnavailable: m: InternalServerError")


def test_builds_its_own_sdk_client() -> None:
    adapter = OpenAITextClient("m", base_url=BASE, api_key="k")
    assert str(adapter.client.base_url).startswith(BASE)


def test_inline_base64_images_are_sent_as_data_urls(client: Any) -> None:
    text_client, sent = client
    content = [
        {"type": "text", "text": "Describe"},
        {"type": "image", "data_b64": "AAAA", "mime": "image/jpeg"},
    ]
    text_client.complete([{"role": "user", "content": content}])
    part = sent[-1]["messages"][0]["content"][1]
    assert part == {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,AAAA"}}
    text_client.complete([{"role": "user", "content": [{"type": "image", "data_b64": "BBBB"}]}])
    assert sent[-1]["messages"][0]["content"][0]["image_url"]["url"] == "data:image/png;base64,BBBB"


def test_missing_sdk_names_the_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "openai", None)  # import openai -> ImportError
    with pytest.raises(MissingExtra, match=r"pip install hone-taste\[openai\]"):
        OpenAITextClient("m", base_url=BASE)
    sent: list[dict[str, Any]] = []
    injected = OpenAITextClient("m", client=sdk(answer_for, sent))  # an injected client needs no import
    assert injected.complete([{"role": "user", "content": "hi"}]).text == "OK"
    assert sent[0]["model"] == "m"
