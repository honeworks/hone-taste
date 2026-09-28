"""Trace context and span creation (design/current.md §6 and §7).

The current context lives in a `ContextVar`, so nested calls (a combined scorer calling its parts, a panel
calling its client) inherit the trace id and link to their parent span.
"""

from __future__ import annotations

import os
import re
import secrets
import socket
from collections.abc import Generator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from importlib.metadata import PackageNotFoundError, version
from typing import Any

from hone_taste._records import current_sink, utc_now

# shared span attributes: copied from the trace context onto every span
# (a hone-lens replay carries its finding id)
SHARED_IDS = (
    "hone.run_id",
    "hone.item",
    "hone.step",
    "hone.candidate_id",
    "hone.scorer",
    "hone.lens.finding_id",
)
TRACEPARENT = re.compile(r"00-([0-9a-f]{32})-([0-9a-f]{16})-[0-9a-f]{2}")

_CURRENT: ContextVar[dict[str, str] | None] = ContextVar("hone_taste_trace", default=None)


def current_trace() -> dict[str, str]:
    """The active trace context; pass it as `trace=` when calling another package."""
    return dict(_CURRENT.get() or {})


@contextmanager
def use_trace(trace: Mapping[str, str]) -> Generator[None]:
    """Run the block inside `trace` (e.g. to add `hone.candidate_id` for the scorers it calls)."""
    token = _CURRENT.set(dict(trace))
    try:
        yield
    finally:
        _CURRENT.reset(token)


def _package_version() -> str:
    try:
        return version("hone-taste")
    except PackageNotFoundError:  # running from a source tree that is not installed
        return "0+unknown"


RESOURCE = {
    "service.name": os.environ.get("OTEL_SERVICE_NAME", "hone-taste"),
    "hone.package": "hone-taste",
    "hone.package.version": _package_version(),
    "host.name": socket.gethostname(),
}


@contextmanager
def start_span(name: str, *, trace: Mapping[str, str] | None = None) -> Generator[dict[str, Any]]:
    """Open a span as a child of `trace` (or of the active context) and emit it to the current sink on
    exit. The block fills `span["attributes"]` and may set `span["status"]`; an exception marks it error.
    """
    context = dict(trace) if trace is not None else current_trace()
    match = TRACEPARENT.fullmatch(context.get("traceparent", ""))
    trace_id, parent = (match.group(1), match.group(2)) if match else (secrets.token_hex(16), None)
    span_id = secrets.token_hex(8)
    shared = {key: context[key] for key in SHARED_IDS if key in context}
    span: dict[str, Any] = {
        "trace_id": trace_id,
        "span_id": span_id,
        "parent_span_id": parent,
        "name": name,
        "kind": "internal",
        "start_time": utc_now(),
        "end_time": None,
        "status": {"code": "ok", "message": ""},
        "attributes": {"hone.schema_version": "1", **shared},
        "events": [],
        "resource": {**RESOURCE, "process.pid": os.getpid()},
        "links": [],
    }
    token = _CURRENT.set({**context, "traceparent": f"00-{trace_id}-{span_id}-01"})
    try:
        yield span
    except BaseException as exc:
        span["status"] = {"code": "error", "message": f"{type(exc).__name__}: {exc}"}
        raise
    finally:
        _CURRENT.reset(token)
        span["end_time"] = utc_now()
        current_sink().emit(span)
