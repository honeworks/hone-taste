"""Records: every score is a `hone.taste.score` span; where it goes, how it joins a trace, what it keeps.

What: each scorer call writes one span in the shared honeworks format (schema version 1, OpenTelemetry
      field names) with `hone.taste.*` attributes (scorer, family, value, raw, reason, details ...).
      By default spans go to SQLite at `${HONE_HOME:-.hone}/taste/spans.db`; `tt.recording(sink)` sends a
      block's spans elsewhere: `tt.MemorySink()`, `tt.JsonlSpanSink(path)`, `tt.SqliteSpanSink(path)` or
      `tt.NullSink()` (off). Any object with `emit`, `flush` and `close` is a sink (`tt.RecordSink`).
How:  1. wrap calls in `with tt.recording(tt.MemorySink()) as sink:` and read `sink.spans`,
      2. pass `trace={"traceparent": ..., "hone.run_id": ...}` to join a caller's trace; the span becomes
         a child of the caller's span and copies the shared `hone.*` ids,
      3. `combine` and profile scorers record their parts as child spans in the same trace; a panel passes
         the context to its DecisionClient as `trace=` (any code can read it with `tt.current_trace()`),
      4. `capture_content=False` (or `HONE_CAPTURE_CONTENT=0`) stores a hash instead of reason / details;
         API-key-looking strings are always replaced by `***`,
      5. file sinks: JSONL (one span per line) or SQLite; both read back with `.spans()`.
Why:  hone-lens and your own tools can later answer "why did this candidate win?" from the records, across
      packages, because every honeworks package writes the same span shape and propagates one trace id.
      Pitfalls: turn content capture off when the scored texts are private; recording never raises into
      scoring (a broken sink is reported once on stderr and counted in `sink.failures`).

Run: uv run python examples/records_and_tracing.py
"""

import json
import tempfile
from pathlib import Path

import hone_taste as tt
from hone_taste.testing import FakeDecisionClient

CALLER_TRACE_ID = "4bf92f3577b34da6a3ce929d0e0e4736"
CALLER_SPAN_ID = "00f067aa0ba902b7"
caller = {"traceparent": f"00-{CALLER_TRACE_ID}-{CALLER_SPAN_ID}-01", "hone.run_id": "run-42"}

slop = tt.slop_score()
banned = tt.patterns(["delve"])

# 1-2. One scorer call inside a caller's trace.
with tt.recording(tt.MemorySink()) as sink:
    slop("Let us delve into the rich tapestry of it.", trace=caller)
span = sink.spans[0]
print(span["name"], span["trace_id"], "parent:", span["parent_span_id"], "status:", span["status"]["code"])
print("attributes:", {k: v for k, v in span["attributes"].items() if k != "hone.taste.details"})
assert span["trace_id"] == CALLER_TRACE_ID and span["parent_span_id"] == CALLER_SPAN_ID
assert span["attributes"]["hone.run_id"] == "run-42" and span["attributes"]["hone.taste.scorer"] == "slop"
assert json.loads(span["attributes"]["hone.taste.details"])["hits"]  # details are stored as JSON

# 3. Nested scorers: the parts are children of the combined span; the panel's client receives the trace.
client = FakeDecisionClient()
panel = tt.audience(["A careful editor"], "Would you publish this?", client)
both = tt.combine({"slop": 1, "banned": 1, "panel": 1}, {"slop": slop, "banned": banned, "panel": panel})
with tt.recording(tt.MemorySink()) as sink:
    both("Let us delve in.", trace=caller)
combined = next(s for s in sink.spans if s["attributes"]["hone.taste.scorer"] == "combined")
children = [s for s in sink.spans if s["parent_span_id"] == combined["span_id"]]
print(
    "child spans of",
    combined["attributes"]["hone.taste.scorer"],
    "->",
    sorted(s["attributes"]["hone.taste.scorer"] for s in children),
)
panel_span = next(s for s in children if s["attributes"]["hone.taste.scorer"] == "audience")
print("the panel's client got trace:", client.calls[0]["trace"]["traceparent"])
assert len(children) == 3 and {s["trace_id"] for s in sink.spans} == {CALLER_TRACE_ID}
assert panel_span["span_id"] in client.calls[0]["trace"]["traceparent"]  # the client's calls nest under it

# 4. Content capture off, and secrets never stored.
leaky_answer = {"type": "score", "value": 0.5, "rationale": "use key sk-live1234567890abcd"}
leaky = tt.audience(["A reviewer"], "Rate it", FakeDecisionClient(answers={"rating": leaky_answer}))
with tt.recording(tt.MemorySink()) as sink:
    leaky("some text")
with tt.recording(tt.MemorySink(capture_content=False)) as private:
    slop("A private diary entry.")
print("redacted:", "sk-live" not in json.dumps(sink.spans))
print("details without capture:", private.spans[0]["attributes"]["hone.taste.details"])
assert "***" in sink.spans[0]["attributes"]["hone.taste.details"]
assert "sk-live" not in json.dumps(sink.spans)
assert "diary" not in json.dumps(private.spans)
assert set(json.loads(private.spans[0]["attributes"]["hone.taste.details"])) == {"sha256", "len"}

# 5. File sinks, read back. Without tt.recording(...) spans go to ${HONE_HOME:-.hone}/taste/spans.db.
with tempfile.TemporaryDirectory() as folder:
    jsonl, sqlite = tt.JsonlSpanSink(Path(folder, "spans.jsonl")), tt.SqliteSpanSink(Path(folder, "spans.db"))
    with tt.recording(jsonl):
        banned("Let us delve in.")
    with tt.recording(sqlite):
        banned("Plain words.")
    print("jsonl values:", [s["attributes"]["hone.taste.value"] for s in jsonl.spans()])
    print("sqlite values:", [s["attributes"]["hone.taste.value"] for s in sqlite.spans()])
    assert jsonl.spans()[0]["attributes"]["hone.taste.value"] == 0.8
    assert sqlite.spans()[0]["attributes"]["hone.taste.value"] == 1.0
    sqlite.close()  # close SQLite before its folder goes away

with tt.recording(tt.NullSink()):  # recording off for this block
    off = slop("Nothing about this call is stored.")
assert off.value is not None  # scoring works as usual; only the span is dropped
