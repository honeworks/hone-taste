"""Contract checkers for the ports hone-taste owns (the shared honeworks port contracts, v1).

Providers (hone-models, your own adapters) run these against their implementations:

>>> from hone_taste.testing import FakeDecisionClient
>>> check_decision_client(FakeDecisionClient())
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from typing import Any

from hone_taste.ports import get_field as get

__all__ = [
    "check_decision_client",
    "check_gpu_lease",
    "check_record_sink",
    "check_text_client",
    "example_span",
    "get",
]


def check_text_client(client: Any) -> None:
    r = client.complete([{"role": "user", "content": "Say OK."}])
    assert isinstance(get(r, "text"), str)
    assert get(r, "error") is None or isinstance(get(r, "error"), str)
    schema = {"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"]}
    r = client.complete(
        [{"role": "user", "content": 'Return {"ok": true}.'}],
        schema=schema,
        trace={"traceparent": "00-" + "a" * 32 + "-" + "b" * 16 + "-01"},
    )
    assert get(r, "parsed") is not None or get(r, "error")
    r = client.complete([{"role": "user", "content": "x"}], unknown_param_is_ignored=1)


def check_decision_client(client: Any) -> None:
    qs: dict[str, dict[str, Any]] = {
        "q1": {"type": "yes_no", "instructions": "Is the sky described as blue?"},
        "q2": {"type": "choice", "instructions": "Colour?", "options": ["blue", "red"]},
        "q3": {"type": "score", "instructions": "How vivid?", "scale": [1, 5]},
    }
    a = client.decide("The sky is blue.", qs)
    assert set(a) <= set(qs)
    for name, ans in a.items():
        v, err = get(ans, "value"), get(ans, "error")
        assert (v is None) == bool(err) or err is None
        if v is not None:
            assert 0.0 <= v <= 1.0
        assert get(ans, "type") == qs[name]["type"]
        assert isinstance(get(ans, "calibrated", False), bool)
    if "q2" in a and get(a["q2"], "value") is not None:
        assert get(a["q2"], "choice") in ("blue", "red")


def check_gpu_lease(g: Any) -> None:
    with g.lease("contract-test", 0.1, timeout_s=5):  # noqa: SIM117 - two nested leases test reentrancy
        with g.lease("contract-test", 0.1, timeout_s=5):  # reentrant
            pass


def check_record_sink(sink: Any, read_back: Callable[[], Iterable[Mapping[str, Any]]]) -> None:
    span = example_span()
    sink.emit(span)
    sink.flush()
    got = [s for s in read_back() if s["span_id"] == span["span_id"]]
    assert got
    assert got[0]["trace_id"] == span["trace_id"]


def example_span() -> dict[str, Any]:
    """A minimal valid span in the shared span format."""
    return {
        "trace_id": "4bf92f3577b34da6a3ce929d0e0e4736",
        "span_id": "00f067aa0ba902b7",
        "parent_span_id": None,
        "name": "hone.test.example",
        "kind": "internal",
        "start_time": "2026-09-27T14:03:11.120Z",
        "end_time": "2026-09-27T14:03:11.220Z",
        "status": {"code": "ok", "message": ""},
        "attributes": {"hone.schema_version": "1"},
        "events": [],
        "resource": {},
        "links": [],
    }
