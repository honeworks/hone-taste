"""Every sink satisfies the RecordSink contract checker."""

from pathlib import Path

import hone_taste as tt
from hone_taste.testing import contracts


def test_sqlite_sink(tmp_path: Path) -> None:
    sink = tt.SqliteSpanSink(tmp_path / "spans.db")
    contracts.check_record_sink(sink, sink.spans)
    sink.close()


def test_jsonl_sink(tmp_path: Path) -> None:
    sink = tt.JsonlSpanSink(tmp_path / "spans.jsonl")
    contracts.check_record_sink(sink, sink.spans)


def test_memory_sink() -> None:
    sink = tt.MemorySink()
    contracts.check_record_sink(sink, lambda: sink.spans)


def test_null_sink_is_a_record_sink() -> None:
    sink: tt.RecordSink = tt.NullSink()
    sink.emit(contracts.example_span())
    sink.flush()
    sink.close()
