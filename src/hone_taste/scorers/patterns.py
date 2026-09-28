"""A scorer from your own banned phrases and regexes."""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

from hone_taste.errors import ConfigError
from hone_taste.types import FunctionScorer, Score

Named = tuple[str, re.Pattern[str]]


def compile_pattern(pattern: str | re.Pattern[str]) -> Named:
    """(name, regex): regexes are used as given; a literal matches as a whole phrase, ignoring case."""
    if isinstance(pattern, re.Pattern):
        return pattern.pattern, pattern
    return pattern, re.compile(rf"(?<!\w){re.escape(pattern)}(?!\w)", re.IGNORECASE)


def find_hits(text: str, patterns: Sequence[Named], label: str = "pattern") -> list[dict[str, Any]]:
    """Every non-empty match, in text order, as `{label: name, "match": ..., "span": [s, e]}`."""
    hits = [
        {label: name, "match": m.group(0), "span": [m.start(), m.end()]}
        for name, pattern in patterns
        for m in pattern.finditer(text)
        if m.end() > m.start()
    ]
    return sorted(hits, key=lambda hit: hit["span"])


def patterns(
    banned: Sequence[str | re.Pattern[str]], name: str = "patterns", *, max_hits: int = 5
) -> FunctionScorer:
    """Score text by how many banned literals / regexes it contains: `1 - min(1, hits / max_hits)`.

    >>> import re
    >>> s = patterns(["delve", re.compile(r"\\bneon \\w+")], max_hits=2)
    >>> result = s("Let us delve into neon dreams.")
    >>> result.value, [hit["match"] for hit in result.details["hits"]]
    (0.0, ['delve', 'neon dreams'])
    """
    if not banned:
        raise ConfigError("patterns() needs at least one banned string or regex")
    if max_hits < 1:
        raise ConfigError(f"max_hits must be >= 1; got {max_hits}")
    compiled = [compile_pattern(p) for p in banned]

    def score(text: str) -> Score:
        hits = find_hits(text, compiled)
        value = 1.0 - min(1.0, len(hits) / max_hits)
        reason = f"{len(hits)} banned pattern hit(s)" if hits else "no banned patterns"
        return Score(value, reason=reason, details={"hits": hits, "max_hits": max_hits})

    return FunctionScorer(
        name,
        "human_likeness",
        frozenset({"text"}),
        score,
        license="user-supplied",
        source_data="user-supplied patterns",
    )
