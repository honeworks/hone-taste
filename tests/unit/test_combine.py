import pytest

from hone_taste.combine import aggregate_function, combine
from hone_taste.errors import ConfigError
from hone_taste.types import FunctionScorer, Score


def fixed(name: str, value: float | None, accepts: str = "text", license: str = "MIT") -> FunctionScorer:
    error = "" if value is not None else f"{name} failed"
    return FunctionScorer(
        name,
        "human_likeness",
        frozenset({accepts}),
        lambda _: Score(value, error=error),
        license=license,
        source_data=f"{name} data",
    )


@pytest.mark.parametrize(
    ("name", "expected"), [("weighted_mean", 0.42), ("mean", 0.42), ("min", 0.2), ("median", 0.5)]
)
def test_aggregates(name: str, expected: float) -> None:
    # weighted mean: (0.2 * 3 + 0.5 + 1.0) / 5 = 0.42; the None value and its weight are dropped
    assert aggregate_function(name)([(0.2, 3.0), (0.5, 1.0), (1.0, 1.0), (None, 9.0)]) == pytest.approx(
        expected
    )


def test_aggregate_all_none_is_none_not_zero() -> None:
    assert aggregate_function("min")([(None, 1.0)]) is None


def test_zero_weight_is_ignored() -> None:
    assert aggregate_function("min")([(0.1, 0.0), (0.5, 1.0)]) == 0.5


def test_unknown_aggregate() -> None:
    with pytest.raises(ConfigError, match="unknown aggregate 'max'"):
        combine({"a": 1}, {"a": fixed("a", 1.0)}, aggregate="max")


def test_combine_details_reason_and_metadata() -> None:
    scorer = combine(
        {"a": 1, "b": 1}, {"a": fixed("a", 1.0), "b": fixed("b", 0.5, "text_with_prompt", "CC-BY-4.0")}
    )
    assert scorer.accepts == frozenset({"text", "text_with_prompt"})
    assert scorer.license == "CC-BY-4.0, MIT"
    assert scorer.source_data == "a: a data; b: b data"
    result = scorer("hello")  # plain text does not fit b (text_with_prompt)
    assert result.details["parts"]["a"] == {"value": 1.0, "weight": 1, "error": ""}
    assert result.details["parts"]["b"]["value"] is None
    assert result.details["parts"]["b"]["error"].startswith("accepts text_with_prompt")
    assert result.value == 1.0
    assert result.reason == "weighted_mean of a=1.00"


def test_combine_all_fail() -> None:
    result = combine({"a": 1}, {"a": fixed("a", None)})("x")
    assert result.value is None
    assert result.error == "no scorer produced a value (a: a failed)"


@pytest.mark.parametrize(
    ("weights", "message"),
    [({"zz": 1}, "not passed: \\['zz'\\]"), ({"a": -1}, ">= 0"), ({"a": 0}, "positive total")],
)
def test_bad_weights(weights: dict[str, float], message: str) -> None:
    with pytest.raises(ConfigError, match=message):
        combine(weights, {"a": fixed("a", 1.0)})
