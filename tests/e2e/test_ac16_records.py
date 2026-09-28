"""AC-16: every score writes a `hone.taste.score` span with the documented attribute names."""

import json
import sqlite3
from pathlib import Path

import pytest

import hone_taste as tt
from hone_taste.testing import FakeDecisionClient, FakeTasteModel

TRACE_ID = "4bf92f3577b34da6a3ce929d0e0e4736"
PARENT = "00f067aa0ba902b7"
FAKE_KEY = "sk-test0123456789abcdefSECRET"
ENV_TOKEN = "plain-token-without-prefix-42"  # noqa: S105 - a planted fake


def read(db: Path) -> list[dict]:
    con = sqlite3.connect(db)
    con.row_factory = sqlite3.Row
    rows = [dict(r) for r in con.execute("SELECT * FROM spans ORDER BY rowid")]
    con.close()
    for row in rows:
        row["attributes"] = json.loads(row["attributes"])
    return rows


def test_ac16_records(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HONE_HOME", str(tmp_path / "home"))
    slop = tt.slop_score()
    panel = tt.audience(["a listener"], "Would you keep listening?", FakeDecisionClient())
    both = tt.combine({"slop": 1.0, "panel": 1.0}, {"slop": slop, "panel": panel})

    both("It's not just a song, but a journey.", trace={"traceparent": f"00-{TRACE_ID}-{PARENT}-01"})
    tt.patterns(["x"])(42)  # accepts mismatch -> error span

    db = tmp_path / "home" / "taste" / "spans.db"
    con = sqlite3.connect(db)
    meta = dict(con.execute("SELECT key, value FROM meta").fetchall())
    tables = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
    changes = con.execute("SELECT COUNT(*) FROM changes").fetchone()[0]
    con.close()
    assert meta["schema"] == "hone-spans"
    assert meta["schema_version"] == "1"
    assert meta["package"] == "hone-taste"
    assert {"meta", "spans", "blobs", "changes"} <= tables

    spans = read(db)
    assert changes == len(spans) == 4
    assert {s["name"] for s in spans} == {"hone.taste.score"}
    by_scorer = {s["attributes"]["hone.taste.scorer"]: s for s in spans}
    root, part = by_scorer["combined"], by_scorer["slop"]
    assert root["trace_id"] == TRACE_ID
    assert root["parent_span_id"] == PARENT
    assert part["trace_id"] == TRACE_ID
    assert part["parent_span_id"] == root["span_id"]  # nested scorer links to its parent
    assert by_scorer["audience"]["parent_span_id"] == root["span_id"]
    assert root["kind"] == "internal"
    assert root["status_code"] == "ok"

    attrs = part["attributes"]
    assert attrs["hone.schema_version"] == "1"
    assert attrs["hone.taste.family"] == "human_likeness"
    assert attrs["hone.taste.license"]
    assert 0.0 <= attrs["hone.taste.value"] <= 1.0
    assert json.loads(attrs["hone.taste.details"])["hits"]
    assert root["attributes"]["hone.taste.family"] == "combined"
    assert by_scorer["audience"]["attributes"]["hone.taste.family"] == "audience"

    error = by_scorer["patterns"]
    assert error["status_code"] == "error"
    assert "accepts" in error["status_message"]
    assert "hone.taste.value" not in error["attributes"]  # could not score: no value, never 0


def test_ac16_panel_passes_the_trace_to_its_client() -> None:
    client = FakeDecisionClient()
    with tt.recording(tt.MemorySink()) as sink:
        tt.audience(["a listener"], "Keep listening?", client)("la la")
    traceparent = client.calls[0]["trace"]["traceparent"]
    assert traceparent == f"00-{sink.spans[0]['trace_id']}-{sink.spans[0]['span_id']}-01"


def test_ac16_content_capture_off_and_secrets(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MY_SERVICE_API_TOKEN", ENV_TOKEN)  # secrets named in the environment
    text = f"my key is {FAKE_KEY} and Bearer abc.def-123 please delve; {ENV_TOKEN}"
    scorer = tt.patterns(["delve", "sk-", "Bearer", "plain-token"])
    with tt.recording(tt.SqliteSpanSink(tmp_path / "on.db")):
        scorer(text)
    monkeypatch.setenv("HONE_CAPTURE_CONTENT", "0")
    with tt.recording(tt.SqliteSpanSink(tmp_path / "off.db")):
        scorer(text)

    on, off = read(tmp_path / "on.db")[0], read(tmp_path / "off.db")[0]
    assert "delve" in on["attributes"]["hone.taste.details"]
    hashed = json.loads(off["attributes"]["hone.taste.details"])
    assert set(hashed) == {"sha256", "len"}
    assert off["attributes"]["hone.taste.value"] == on["attributes"]["hone.taste.value"]
    for name in ("on.db", "off.db"):
        raw = b"".join(p.read_bytes() for p in tmp_path.glob(f"{name}*"))
        assert FAKE_KEY.encode() not in raw
        assert b"abc.def-123" not in raw
        assert ENV_TOKEN.encode() not in raw


def test_ac16_taste_model_and_profile_attributes(tmp_path: Path) -> None:
    ref = tmp_path / "ref.json"
    ref.write_text('{"good": [10, 20], "bad": [0, 5]}')
    me = tt.profile("records", path=tmp_path / "me.json")
    me.add_pick(winner="plain words", loser="Let us delve into the tapestry.")
    slop = {"slop": tt.slop_score()}
    me.fit(slop)
    with tt.recording(tt.MemorySink()) as sink:
        tt.reward_model(model=FakeTasteModel({"reward": 12.0}), reference=ref)(
            {"prompt": "p", "response": "r"}
        )
        me.as_scorer(slop)("plain words")
    rm = sink.spans[0]["attributes"]
    assert rm["hone.taste.model_id"] == "Skywork/Skywork-Reward-V2-Qwen3-0.6B"
    assert rm["hone.taste.raw"] == 12.0
    assert rm["hone.taste.reference_set"] == "text/reward_model"
    assert rm["hone.taste.family"] == "taste_model"
    personal = [s for s in sink.spans if s["attributes"]["hone.taste.scorer"] == "profile:records"]
    assert personal[0]["attributes"]["hone.taste.family"] == "personal"


def test_ac16_secrets_in_errors_are_redacted(tmp_path: Path) -> None:
    def leaky(_: object) -> dict[str, float]:
        raise RuntimeError(f"auth failed for {FAKE_KEY}")

    scorer = tt.audiobox(model=FakeTasteModel(leaky))
    for sink in (tt.SqliteSpanSink(tmp_path / "s.db"), tt.JsonlSpanSink(tmp_path / "s.jsonl")):
        with tt.recording(sink):
            assert FAKE_KEY in scorer("a.wav").error  # the caller still sees its own error
        span = sink.spans()[0]
        assert "***" in span["status"]["message"]
        assert "***" in span["attributes"]["hone.taste.error"]
    raw = b"".join(p.read_bytes() for p in tmp_path.glob("s.*"))
    assert FAKE_KEY.encode() not in raw


def test_ac16_records_are_deterministic() -> None:
    def run() -> list[dict]:
        panel = tt.audience(["a listener"], "Keep listening?", FakeDecisionClient())
        both = tt.combine({"slop": 1.0, "panel": 1.0}, {"slop": tt.slop_score(), "panel": panel})
        with tt.recording(tt.MemorySink()) as sink:
            both("It's not just a song, but a journey.")
        volatile = ("trace_id", "span_id", "parent_span_id", "start_time", "end_time", "resource")
        return [{k: v for k, v in span.items() if k not in volatile} for span in sink.spans]

    assert run() == run()
