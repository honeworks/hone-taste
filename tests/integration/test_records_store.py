"""The SQLite span store under two writer processes and after a writer is killed."""

import json
import signal
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

WRITER = """
import sys
import hone_taste as tt
db, count = sys.argv[1], int(sys.argv[2])
scorer = tt.patterns(["delve"])
with tt.recording(tt.SqliteSpanSink(db)):
    i = 0
    while count < 0 or i < count:
        scorer(f"line {i}: let us delve")
        i += 1
"""


def counts(db: Path) -> tuple[int, int, str]:
    con = sqlite3.connect(db)
    try:
        spans = con.execute("SELECT COUNT(*) FROM spans").fetchone()[0]
        changes = con.execute("SELECT COUNT(*) FROM changes").fetchone()[0]
        integrity = con.execute("PRAGMA integrity_check").fetchone()[0]
    finally:
        con.close()
    return spans, changes, integrity


def _span_count(db: Path) -> int:
    """Spans written so far; 0 while the writer is still creating the file and its tables."""
    try:
        return counts(db)[0] if db.exists() else 0
    except sqlite3.OperationalError:  # the file exists before the WAL switch and the schema
        return 0


def test_two_processes_write_the_same_store(tmp_path: Path) -> None:
    db = tmp_path / "spans.db"
    writers = [
        subprocess.Popen([sys.executable, "-c", WRITER, str(db), "150"], stderr=subprocess.PIPE, text=True)
        for _ in range(2)
    ]
    errors = [w.communicate(timeout=60)[1] for w in writers]
    assert all(w.returncode == 0 for w in writers), errors
    assert not any("could not record spans" in e for e in errors), errors
    assert counts(db) == (300, 300, "ok")


CREATOR = """
import sys, time
from pathlib import Path
import hone_taste as tt
folder, start = Path(sys.argv[1]), float(sys.argv[2])
for store in range(4):  # every process creates the same fresh store at the same moment, four times
    time.sleep(max(0.0, start + 0.4 * store - time.time()))
    sink = tt.SqliteSpanSink(folder / f"{store}.db")
    with tt.recording(sink):
        tt.patterns(["delve"])("let us delve")
    assert sink.failures == 0, sink.failures
"""


def test_processes_creating_the_same_fresh_store_never_drop_spans(tmp_path: Path) -> None:
    start = time.time() + 1.5  # after every interpreter has started
    creators = [
        subprocess.Popen(
            [sys.executable, "-c", CREATOR, str(tmp_path), str(start)], stderr=subprocess.PIPE, text=True
        )
        for _ in range(8)
    ]
    errors = [c.communicate(timeout=60)[1] for c in creators]
    assert all(c.returncode == 0 for c in creators), errors
    assert not any("could not record spans" in e for e in errors), errors
    assert [counts(tmp_path / f"{store}.db") for store in range(4)] == [(8, 8, "ok")] * 4


def test_store_survives_a_killed_writer(tmp_path: Path) -> None:
    db = tmp_path / "spans.db"
    writer = subprocess.Popen([sys.executable, "-c", WRITER, str(db), "-1"])
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline and _span_count(db) < 20:
        time.sleep(0.05)
    writer.send_signal(signal.SIGKILL)
    writer.wait(timeout=10)

    spans, changes, integrity = counts(db)
    assert integrity == "ok"
    assert spans >= 20
    assert changes == spans  # span and change row are committed together
    con = sqlite3.connect(db)
    rows = con.execute("SELECT attributes, end_time FROM spans").fetchall()
    con.close()
    assert all(json.loads(attributes)["hone.taste.scorer"] == "patterns" for attributes, _ in rows)
    assert all(end_time for _, end_time in rows)
