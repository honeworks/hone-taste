import json
import sqlite3
from pathlib import Path

import pytest

import hone_taste as tt
from hone_taste._tracing import current_trace, start_span, use_trace
from hone_taste.testing.contracts import example_span


def test_sink_failures_never_raise_and_are_reported_once(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "taken").mkdir()
    sink = tt.SqliteSpanSink(tmp_path / "taken")  # a directory: sqlite cannot open it
    jsonl = tt.JsonlSpanSink(tmp_path / "taken")
    for _ in range(2):
        sink.emit(example_span())
        jsonl.emit(example_span())
    assert sink.failures == 2
    assert jsonl.failures == 2
    assert capsys.readouterr().err.count("could not record spans") == 2  # once per sink


def test_large_attributes_go_to_blobs(tmp_path: Path) -> None:
    sink = tt.SqliteSpanSink(tmp_path / "spans.db")
    span = example_span()
    big = "x" * (65 * 1024)
    span["attributes"]["hone.taste.details"] = big
    sink.emit(span)
    stored = sink.spans()[0]["attributes"]["hone.taste.details"]
    con = sqlite3.connect(tmp_path / "spans.db")
    data = con.execute("SELECT data FROM blobs WHERE sha256 = ?", (stored["$blob"],)).fetchone()[0]
    con.close()
    sink.close()
    assert data.decode() == big


def test_jsonl_sink_round_trip(tmp_path: Path) -> None:
    sink = tt.JsonlSpanSink(tmp_path / "sub" / "spans.jsonl", capture_content=False)
    assert sink.spans() == []
    span = example_span()
    span["attributes"]["hone.taste.reason"] = "fine"
    sink.emit(span)
    sink.flush()
    sink.close()
    got = sink.spans()[0]["attributes"]["hone.taste.reason"]
    assert set(json.loads(got)) == {"sha256", "len"}


def test_span_error_on_exception_and_context_restored() -> None:
    with tt.recording(tt.MemorySink()) as sink:

        def fail() -> None:
            with start_span("hone.taste.score"):
                assert current_trace()["traceparent"].startswith("00-")
                raise RuntimeError("boom")

        with pytest.raises(RuntimeError):
            fail()
        assert current_trace() == {}
    assert sink.spans[0]["status"] == {"code": "error", "message": "RuntimeError: boom"}
    assert sink.spans[0]["parent_span_id"] is None
    assert len(sink.spans[0]["trace_id"]) == 32
    assert sink.spans[0]["end_time"].endswith("Z")


def test_shared_ids_come_from_the_context() -> None:
    context = {"hone.run_id": "r1", "hone.lens.finding_id": "F-1", "other": "x"}
    with tt.recording(tt.MemorySink()) as sink, use_trace(context):
        tt.patterns(["a"])("a b")
    attributes = sink.spans[0]["attributes"]
    assert attributes["hone.run_id"] == "r1"
    assert attributes["hone.lens.finding_id"] == "F-1"
    assert "other" not in attributes
    assert sink.spans[0]["resource"]["hone.package"] == "hone-taste"


def test_default_sink_follows_hone_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HONE_HOME", str(tmp_path))
    tt.patterns(["a"])("a")
    assert (tmp_path / "taste" / "spans.db").exists()


def test_null_sink_records_nothing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HONE_HOME", str(tmp_path))
    with tt.recording(tt.NullSink()) as sink:
        tt.patterns(["a"])("a")
    sink.flush()
    sink.close()
    assert not (tmp_path / "taste").exists()
