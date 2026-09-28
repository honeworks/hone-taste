import pytest

from hone_taste.agreement import AgreementReport, agreement
from hone_taste.errors import ConfigError
from hone_taste.fitting import Weights
from hone_taste.types import FunctionScorer, Score


def test_agreement_needs_scorers_and_pairs() -> None:
    scorer = FunctionScorer("s", "human_likeness", frozenset({"text"}), lambda t: Score(0.5))
    with pytest.raises(ConfigError):
        agreement({}, [("a", "b")])
    with pytest.raises(ConfigError):
        agreement({"s": scorer}, [])


def test_zero_is_a_value_not_missing() -> None:
    zero = FunctionScorer("z", "human_likeness", frozenset({"text"}), lambda t: Score(0.0))
    result = agreement({"z": zero}, [("a", "b")]).scorers["z"]
    assert (result.compared, result.unscored, result.rate, result.correlation) == (1, 0, 0.5, 0.0)


def test_scorer_that_never_scores_has_no_rate() -> None:
    broken = FunctionScorer("b", "human_likeness", frozenset({"text"}), lambda t: Score(None, error="x"))
    report = agreement({"b": broken}, [("a", "b")])
    assert report.scorers["b"].rate is None
    assert report.scorers["b"].correlation is None
    assert report.scorers["b"].unscored == 1
    assert " - " in str(report)
    assert report.to_dict()["scorers"]["b"]["rate"] is None


def test_an_empty_report_still_prints() -> None:
    empty = AgreementReport({}, Weights({}, 0.0, 0.0, 0), 0)
    assert str(empty).splitlines()[1].startswith("scorer")
