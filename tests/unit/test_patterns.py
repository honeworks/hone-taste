import re

import pytest

from hone_taste.errors import ConfigError
from hone_taste.scorers.patterns import patterns


def test_needs_patterns_and_positive_max_hits() -> None:
    with pytest.raises(ConfigError, match="at least one"):
        patterns([])
    with pytest.raises(ConfigError, match="max_hits"):
        patterns(["x"], max_hits=0)


def test_empty_regex_matches_are_ignored() -> None:
    result = patterns([re.compile(r"z*")])("abc")
    assert result.details["hits"] == []
    assert result.value == 1.0
    assert result.reason == "no banned patterns"


def test_literal_special_characters_are_escaped() -> None:
    result = patterns(["(sigh)"], max_hits=1)("well (sigh) okay")
    assert [h["match"] for h in result.details["hits"]] == ["(sigh)"]
    assert result.value == 0.0
    assert result.reason == "1 banned pattern hit(s)"


def test_overlapping_patterns_each_count() -> None:
    result = patterns(["heart", "heart of gold"], max_hits=4)("a heart of gold")
    assert [h["pattern"] for h in result.details["hits"]] == ["heart", "heart of gold"]
    assert result.value == 0.5
