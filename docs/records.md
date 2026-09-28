# Records

Every scorer call writes one span named `hone.taste.score` (kind `internal`) in the shared honeworks
format (schema version 1, OpenTelemetry field names). Nested scorers (`combine`, panels, profiles) write
child spans in the same trace.

## Where
- Default: SQLite at `${HONE_HOME:-.hone}/taste/spans.db` (WAL; tables `meta`, `spans`, `blobs`, `changes`).
- `with tt.recording(sink):` sends the spans of a block to another sink: `tt.SqliteSpanSink(path)`,
  `tt.JsonlSpanSink(path)`, `tt.MemorySink()` or `tt.NullSink()` (off). Any object with
  `emit(span)`, `flush()`, `close()` works (`tt.RecordSink`).
- Recording never breaks scoring: a sink failure is reported once on stderr and counted in `sink.failures`.

```python
import hone_taste as tt

with tt.recording(tt.MemorySink()) as sink:
    tt.patterns(["delve"])(
        "Let us delve in.", trace={"traceparent": "00-" + "a" * 32 + "-" + "b" * 16 + "-01"}
    )
span = sink.spans[0]
print(span["name"], span["trace_id"], span["parent_span_id"])
print({k: v for k, v in span["attributes"].items() if k.startswith("hone.")})
```

## Attributes

| Attribute | Meaning |
|---|---|
| `hone.schema_version` | `"1"` |
| `hone.taste.scorer` | scorer name, e.g. `slop_lyrics`, `audience`, `combined`, `profile:ana` |
| `hone.taste.family` | `taste_model`, `human_likeness`, `audience`, `personal`, `combined` |
| `hone.taste.model_id` | model id (taste models only) |
| `hone.taste.license` | license of the scorer / model |
| `hone.taste.value` | 0-1; left out when the scorer could not score (status `error`) |
| `hone.taste.raw` | raw model value before normalization, when numeric |
| `hone.taste.reference_set` | `domain/name` of the reference set used |
| `hone.taste.confidence`, `hone.taste.reason`, `hone.taste.error` | from the `Score` |
| `hone.taste.details` | the `Score.details` as JSON |
| `hone.taste.panel.mode` | `"compare"` on a panel's pairwise comparison (`panel.pairwise(a, b)`); absent on ratings |
| `hone.run_id`, `hone.item`, `hone.step`, `hone.candidate_id`, `hone.scorer` | from the trace context, when present |

## Trace context
Pass `trace={"traceparent": ...}` to any scorer call (or to a `for_select` scorer) to join a caller's
trace; `tt.current_trace()` returns the active context so you can pass it on. The panel passes it to its
`DecisionClient`, taste models to their `GpuLease`.

## Content capture and secrets
`HONE_CAPTURE_CONTENT=0` (or `capture_content=False` on a sink) stores `{"sha256", "len"}` instead of
`hone.taste.details` and `hone.taste.reason`. Strings that look like API keys (`sk-...`, `Bearer ...`)
and the values of environment variables whose names end in `API_KEY`, `TOKEN`, `SECRET` or `PASSWORD`
are replaced by `***` before anything is written.

Example: [`records_and_tracing.py`](../examples/records_and_tracing.py) (sinks, trace context, child spans,
content capture, redaction).
