"""AC-3: the patterns scorer with a regex and a literal: hits and value are correct."""

import re

import pytest

import hone_taste as tt


def test_ac3_patterns() -> None:
    scorer = tt.patterns(["Heart of gold", re.compile(r"\bbaby\b")], name="banned", max_hits=4)
    text = "Baby, you've got a heart of gold, baby, a HEART OF GOLDEN hue."
    result = scorer(text)

    matches = [(h["match"], h["span"]) for h in result.details["hits"]]
    # the literal ignores case and needs whole words ("HEART OF GOLDEN" does not count);
    # the regex keeps its own flags (case-sensitive, so "Baby" does not count)
    assert matches == [("heart of gold", [19, 32]), ("baby", [34, 38])]
    assert all(text[s:e] == m for m, (s, e) in matches)
    assert result.value == pytest.approx(1 - 2 / 4)
    assert scorer.name == "banned"

    assert scorer("clean text").value == 1.0
    assert scorer("baby " * 10).value == 0.0  # capped at max_hits
