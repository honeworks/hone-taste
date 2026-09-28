"""Span sinks (design/current.md §7) and the sink scores are recorded to.

By default every score goes to `${HONE_HOME:-.hone}/taste/spans.db`; `recording(sink)` sends the scores
of a block elsewhere (`NullSink()` turns recording off). Sinks never raise into the caller: a failure is
reported once on stderr and counted in `sink.failures`.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import sys
import threading
import time
from collections.abc import Generator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TypeVar

from hone_taste.ports import RecordSink

CONTENT_ATTRIBUTES = ("hone.taste.details", "hone.taste.reason")  # hashed when content capture is off
SECRET = re.compile(r"sk-[A-Za-z0-9_-]{8,}|Bearer\s+[A-Za-z0-9._~+/=-]+")
SECRET_ENV = re.compile(r"(API_KEY|TOKEN|SECRET|PASSWORD)$", re.IGNORECASE)  # their values are redacted too
BLOB_LIMIT = 64 * 1024

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta   (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS spans (
  span_id        TEXT PRIMARY KEY,
  trace_id       TEXT NOT NULL,
  parent_span_id TEXT,
  name           TEXT NOT NULL,
  kind           TEXT NOT NULL DEFAULT 'internal',
  start_time     TEXT NOT NULL,
  end_time       TEXT,
  status_code    TEXT NOT NULL DEFAULT 'unset',
  status_message TEXT NOT NULL DEFAULT '',
  attributes     TEXT NOT NULL DEFAULT '{}',
  events         TEXT NOT NULL DEFAULT '[]',
  resource       TEXT NOT NULL DEFAULT '{}',
  links          TEXT NOT NULL DEFAULT '[]'
);
CREATE INDEX IF NOT EXISTS spans_trace ON spans(trace_id);
CREATE INDEX IF NOT EXISTS spans_name_time ON spans(name, start_time);
CREATE TABLE IF NOT EXISTS blobs (sha256 TEXT PRIMARY KEY, mime TEXT, size INTEGER, data BLOB NOT NULL);
CREATE TABLE IF NOT EXISTS changes (seq INTEGER PRIMARY KEY AUTOINCREMENT, span_id TEXT NOT NULL,
                                    op TEXT NOT NULL, at TEXT NOT NULL);
"""


def _capture(flag: bool | None) -> bool:
    """The sink's content-capture switch; `None` follows `HONE_CAPTURE_CONTENT` (on unless "0")."""
    return os.environ.get("HONE_CAPTURE_CONTENT", "1") != "0" if flag is None else flag


def utc_now() -> str:
    """ISO-8601 UTC with milliseconds, e.g. 2026-09-27T14:03:11.120Z (shared span format)."""
    return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _redact(text: str) -> str:
    for name, value in os.environ.items():
        if len(value) >= 8 and SECRET_ENV.search(name):
            text = text.replace(value, "***")
    return SECRET.sub("***", text)


def _clean(span: Mapping[str, Any], capture: bool) -> dict[str, Any]:
    """A copy of `span` with secrets replaced and, when `capture` is off, content replaced by hashes."""
    attributes = {k: _redact(v) if isinstance(v, str) else v for k, v in span["attributes"].items()}
    if not capture:
        for key in CONTENT_ATTRIBUTES:
            if isinstance(attributes.get(key), str):
                data = attributes[key].encode()
                attributes[key] = json.dumps({"sha256": hashlib.sha256(data).hexdigest(), "len": len(data)})
    status = dict(span["status"])
    status["message"] = _redact(status.get("message", ""))
    return {**span, "attributes": attributes, "status": status}


class _Reporting:
    """Counts failures and reports the first one on stderr, so recording never breaks scoring."""

    failures: int = 0

    def _failed(self, exc: Exception) -> None:
        self.failures += 1
        if self.failures == 1:
            print(f"hone-taste: could not record spans ({type(exc).__name__}: {exc})", file=sys.stderr)


def _enable_wal(db: sqlite3.Connection, attempts: int = 50) -> None:
    """Switch to WAL. The switch ignores busy_timeout when another process is creating the same store at
    the same moment, so retry briefly."""
    for attempt in range(attempts):
        try:
            db.execute("PRAGMA journal_mode=WAL")
            return
        except sqlite3.OperationalError:
            if attempt == attempts - 1:
                raise
            time.sleep(0.1)


class SqliteSpanSink(_Reporting):
    """Writes spans to the shared SQLite span schema (WAL, one commit per span)."""

    def __init__(self, path: str | Path, *, capture_content: bool | None = None) -> None:
        self.path = Path(path)
        self.capture_content = _capture(capture_content)
        self._db: sqlite3.Connection | None = None
        self._lock = threading.Lock()

    def _connect(self) -> sqlite3.Connection:
        if self._db is None:  # opened on first use: creating a scorer never touches the disk
            self.path.parent.mkdir(parents=True, exist_ok=True)
            db = sqlite3.connect(self.path, timeout=5, check_same_thread=False)
            db.execute("PRAGMA busy_timeout=5000")
            _enable_wal(db)
            db.executescript(SCHEMA)
            now = utc_now()
            meta = [("schema", "hone-spans"), ("schema_version", "1"), ("package", "hone-taste")]
            db.executemany("INSERT OR IGNORE INTO meta VALUES (?, ?)", [*meta, ("created_at", now)])
            db.commit()
            self._db = db
        return self._db

    def _attributes_json(self, db: sqlite3.Connection, attributes: Mapping[str, Any]) -> str:
        """Attributes as JSON; strings over 64 KiB move to `blobs`, referenced as `{"$blob": sha256}`."""
        stored = dict(attributes)
        for key, value in attributes.items():
            if isinstance(value, str) and len(value.encode()) > BLOB_LIMIT:
                data = value.encode()
                digest = hashlib.sha256(data).hexdigest()
                db.execute(
                    "INSERT OR IGNORE INTO blobs VALUES (?, ?, ?, ?)", (digest, "text/plain", len(data), data)
                )
                stored[key] = {"$blob": digest}
        return json.dumps(stored, ensure_ascii=False)

    def emit(self, span: Mapping[str, Any]) -> None:
        try:
            s = _clean(span, self.capture_content)
            with self._lock:
                db = self._connect()
                row = (
                    s["span_id"],
                    s["trace_id"],
                    s.get("parent_span_id"),
                    s["name"],
                    s.get("kind", "internal"),
                    s["start_time"],
                    s.get("end_time"),
                    s["status"].get("code", "unset"),
                    s["status"].get("message", ""),
                    self._attributes_json(db, s["attributes"]),
                    json.dumps(s.get("events", [])),
                    json.dumps(s.get("resource", {})),
                    json.dumps(s.get("links", [])),
                )
                db.execute("INSERT OR REPLACE INTO spans VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", row)
                at = utc_now()
                db.execute(
                    "INSERT INTO changes (span_id, op, at) VALUES (?, 'insert', ?)", (s["span_id"], at)
                )
                db.commit()
        except (sqlite3.Error, OSError, TypeError, ValueError, KeyError) as exc:
            self._failed(exc)

    def spans(self) -> list[dict[str, Any]]:
        """Read every span back in the shared span shape (blob references are left as they are)."""
        with self._lock:
            rows = self._connect().execute("SELECT * FROM spans ORDER BY start_time, rowid").fetchall()
        return [
            {
                "span_id": r[0],
                "trace_id": r[1],
                "parent_span_id": r[2],
                "name": r[3],
                "kind": r[4],
                "start_time": r[5],
                "end_time": r[6],
                "status": {"code": r[7], "message": r[8]},
                "attributes": json.loads(r[9]),
                "events": json.loads(r[10]),
                "resource": json.loads(r[11]),
                "links": json.loads(r[12]),
            }
            for r in rows
        ]

    def flush(self) -> None:
        """Nothing to do: every span is committed when it is emitted."""

    def close(self) -> None:
        with self._lock:
            if self._db is not None:
                self._db.close()
                self._db = None


class JsonlSpanSink(_Reporting):
    """Appends one JSON span per line."""

    def __init__(self, path: str | Path, *, capture_content: bool | None = None) -> None:
        self.path = Path(path)
        self.capture_content = _capture(capture_content)
        self._lock = threading.Lock()

    def emit(self, span: Mapping[str, Any]) -> None:
        try:
            line = json.dumps(_clean(span, self.capture_content), ensure_ascii=False)
            with self._lock:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with self.path.open("a", encoding="utf-8") as fh:
                    fh.write(line + "\n")
        except (OSError, TypeError, ValueError, KeyError) as exc:
            self._failed(exc)

    def spans(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        return [json.loads(line) for line in self.path.read_text(encoding="utf-8").splitlines() if line]

    def flush(self) -> None:
        """Nothing to do: every line is written when it is emitted."""

    def close(self) -> None:
        """Nothing to do: the file is opened per span."""


class MemorySink:
    """Keeps spans in a list (for tests and notebooks)."""

    def __init__(self, *, capture_content: bool | None = None) -> None:
        self.capture_content = _capture(capture_content)
        self.spans: list[dict[str, Any]] = []

    def emit(self, span: Mapping[str, Any]) -> None:
        self.spans.append(_clean(span, self.capture_content))

    def flush(self) -> None: ...

    def close(self) -> None: ...


class NullSink:
    """Drops every span (recording off)."""

    def emit(self, span: Mapping[str, Any]) -> None: ...

    def flush(self) -> None: ...

    def close(self) -> None: ...


SinkT = TypeVar("SinkT", bound=RecordSink)
_SINK: ContextVar[RecordSink | None] = ContextVar("hone_taste_sink", default=None)
_DEFAULT_SINKS: dict[Path, SqliteSpanSink] = {}


def current_sink() -> RecordSink:
    """The sink set by `recording(...)`, else the SQLite store under `${HONE_HOME:-.hone}/taste/`."""
    sink = _SINK.get()
    if sink is not None:
        return sink
    path = Path(os.environ.get("HONE_HOME", ".hone")) / "taste" / "spans.db"
    if path not in _DEFAULT_SINKS:
        _DEFAULT_SINKS[path] = SqliteSpanSink(path)
    return _DEFAULT_SINKS[path]


@contextmanager
def recording(sink: SinkT) -> Generator[SinkT]:
    """Record the scores made inside the block to `sink`.

    >>> import hone_taste as tt
    >>> with tt.recording(tt.MemorySink()) as sink:
    ...     _ = tt.patterns(["delve"])("Let us delve in.")
    >>> sink.spans[0]["name"], sink.spans[0]["attributes"]["hone.taste.value"]
    ('hone.taste.score', 0.8)
    """
    token = _SINK.set(sink)
    try:
        yield sink
    finally:
        _SINK.reset(token)
