import json
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

from hone_taste.errors import ConfigError
from hone_taste.normalize import ReferenceSet, linear, normalized, percentile, reference_set

finite = st.floats(allow_nan=False, allow_infinity=False, min_value=-1e9, max_value=1e9)


@given(raw=finite, values=st.lists(finite, min_size=2, max_size=50).filter(lambda v: len(set(v)) > 1))
def test_values_always_in_unit_interval(raw: float, values: list[float]) -> None:
    ref = ReferenceSet("d", "n", tuple(values))
    assert 0.0 <= linear(raw, ref) <= 1.0
    assert 0.0 <= percentile(raw, ref) <= 1.0


@given(a=finite, b=finite)
def test_linear_is_monotonic(a: float, b: float) -> None:
    ref = ReferenceSet("d", "n", (-3.0, -1.0, 0.0, 2.0, 7.0))
    low, high = sorted((a, b))
    assert linear(low, ref) <= linear(high, ref)


def test_linear_uses_5th_and_95th_percentile() -> None:
    ref = ReferenceSet("d", "n", tuple(float(v) for v in range(101)))
    assert linear(5.0, ref) == 0.0
    assert linear(95.0, ref) == 1.0
    assert linear(50.0, ref) == pytest.approx(0.5)


def test_linear_degenerate_middle() -> None:
    ref = ReferenceSet("d", "n", (0.0, *([1.0] * 40), 2.0))
    assert (linear(0.5, ref), linear(1.0, ref), linear(1.5, ref)) == (0.0, 0.5, 1.0)


def test_percentile_counts_ties_half() -> None:
    ref = ReferenceSet("d", "n", (1.0, 2.0, 3.0, 4.0))
    assert percentile(0.0, ref) == 0.0
    assert percentile(2.5, ref) == 0.5
    assert percentile(9.0, ref) == 1.0


def test_reference_set_needs_two_distinct_values() -> None:
    with pytest.raises(ConfigError, match="two different values"):
        ReferenceSet("d", "n", (1.0, 1.0))


def test_reference_set_from_path(tmp_path: Path) -> None:
    path = tmp_path / "ref.json"
    path.write_text(json.dumps({"source": "mine", "good": [3, 4], "bad": [-1]}))
    ref = reference_set("mine", "judge", path=path)
    assert ref.values == (3.0, 4.0, -1.0)
    assert ref.label == "mine/judge"
    assert ref.source == "mine"


def test_missing_packaged_reference_set_says_what_to_do() -> None:
    with pytest.raises(ConfigError, match=r"pass reference= \(or path=\)"):
        reference_set("nope", "nothing")


def test_normalized_keeps_raw_and_rejects_unknown_mode() -> None:
    ref = ReferenceSet("d", "n", (0.0, 10.0))
    score = normalized(5.0, ref)
    assert score.value == pytest.approx(0.5)
    assert score.details == {"raw": 5.0, "reference_set": "d/n", "normalization": "linear"}
    with pytest.raises(ConfigError, match="unknown normalization mode 'zscore'"):
        normalized(5.0, ref, mode="zscore")


@pytest.mark.parametrize("mode", ["linear", "percentile"])
@pytest.mark.parametrize("raw", [float("nan"), float("inf")])
def test_non_finite_raw_is_none_not_zero(raw: float, mode: str) -> None:
    score = normalized(raw, ReferenceSet("d", "n", (0.0, 10.0)), mode=mode)
    assert score.value is None
    assert score.error.startswith("raw value is not finite")


@pytest.mark.parametrize("content", ["not json", '{"good": ["x", 1]}', "[1, 2]", '{"good": [1, 1]}'])
def test_bad_reference_file_is_config_error(tmp_path: Path, content: str) -> None:
    path = tmp_path / "bad.json"
    path.write_text(content)
    with pytest.raises(ConfigError):
        reference_set("d", "bad", path=path)
