import pytest

from hone_taste.errors import ConfigError
from hone_taste.fitting import fit_weights, item_key, score_table
from hone_taste.types import FunctionScorer, Score


def table_of(values: dict[str, dict[str, float | None]]) -> dict[str, dict[str, float | None]]:
    return {item_key(k): v for k, v in values.items()}


def test_item_key_is_stable_for_dicts_and_paths() -> None:
    assert item_key({"b": 1, "a": 2}) == item_key({"a": 2, "b": 1})
    assert item_key("x") != item_key("y")


def test_score_table_scores_each_item_once_and_keeps_none() -> None:
    calls: list[str] = []

    def score(text: str) -> Score:
        calls.append(text)
        return Score(None, error="no") if text == "bad" else Score(0.5)

    scorer = FunctionScorer("s", "human_likeness", frozenset({"text"}), score)
    table = score_table({"s": scorer}, ["a", "bad", "a"])
    assert calls == ["a", "bad"]
    assert table[item_key("bad")]["s"] is None


def test_fit_needs_scorers_and_picks() -> None:
    with pytest.raises(ConfigError, match="at least one"):
        fit_weights({}, [], [("a", "b")])
    with pytest.raises(ConfigError, match="at least one"):
        fit_weights({}, ["s"], [])


def test_scorer_that_contradicts_picks_gets_zero_weight() -> None:
    table = table_of({"a": {"good": 0.9, "wrong": 0.1}, "b": {"good": 0.1, "wrong": 0.9}})
    fitted = fit_weights(table, ["good", "wrong"], [("a", "b")])
    assert fitted.weights == {"good": 1.0, "wrong": 0.0}
    assert fitted.coefficients["wrong"] < 0
    assert 0 < fitted.log_loss < 0.7


def test_zero_scores_are_evidence() -> None:
    table = table_of({"a": {"s": 0.3}, "b": {"s": 0.0}})
    fitted = fit_weights(table, ["s"], [("a", "b")])
    assert fitted.weights == {"s": 1.0}
    assert fitted.note == ""


def test_no_evidence_gives_equal_weights_with_note() -> None:
    table = table_of({"a": {"s": None, "t": 0.5}, "b": {"s": 0.3, "t": 0.5}})
    fitted = fit_weights(table, ["s", "t"], [("a", "b")])
    assert fitted.weights == {"s": 0.5, "t": 0.5}
    assert fitted.note == "no scorer predicts the picks"
    assert fitted.accuracy == 0.5  # every margin is 0: ties count half


def test_scale_does_not_change_weights() -> None:
    small = table_of({"a": {"s": 0.02}, "b": {"s": 0.01}, "c": {"s": 0.03}})
    big = table_of({"a": {"s": 20.0}, "b": {"s": 10.0}, "c": {"s": 30.0}})
    pairs = [("a", "b"), ("c", "a")]
    assert fit_weights(small, ["s"], pairs).accuracy == fit_weights(big, ["s"], pairs).accuracy == 1.0
    assert fit_weights(small, ["s"], pairs).log_loss == pytest.approx(fit_weights(big, ["s"], pairs).log_loss)
