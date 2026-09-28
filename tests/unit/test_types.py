from pathlib import Path

import pytest

from hone_taste.errors import ConfigError
from hone_taste.types import FunctionScorer, Score, Scorer, input_kind_error


def _scorer(accepts: set[str], fn=lambda _: Score(1.0)) -> FunctionScorer:
    return FunctionScorer("t", "human_likeness", frozenset(accepts), fn)


@pytest.mark.parametrize(
    ("kind", "good", "bad"),
    [
        ("text", "hello", 3),
        ("text_with_prompt", {"prompt": "p", "response": "r"}, {"prompt": "p"}),
        ("audio", Path("a.wav"), 1.5),
        ("image", "img.png", ["img.png"]),
        ("image_with_prompt", {"prompt": "p", "image": "i.png"}, {"image": "i.png"}),
    ],
)
def test_input_kinds(kind: str, good: object, bad: object) -> None:
    assert input_kind_error(frozenset({kind}), good) == ""
    assert input_kind_error(frozenset({kind}), bad).startswith(f"accepts {kind}; got ")


def test_mismatch_returns_none_score_without_calling_function() -> None:
    calls: list[object] = []
    scorer = _scorer({"text"}, lambda t: calls.append(t) or Score(1.0))
    result = scorer({"prompt": "p", "response": "r"})
    assert result.value is None
    assert result.error == "accepts text; got dict"
    assert calls == []


def test_exception_becomes_error_score() -> None:
    def boom(_: str) -> Score:
        raise RuntimeError("model exploded")

    result = _scorer({"text"}, boom)("x")
    assert result.value is None
    assert result.error == "RuntimeError: model exploded"


def test_zero_is_a_value_not_a_failure() -> None:
    result = _scorer({"text"}, lambda _: Score(0.0))("x")
    assert result.value == 0.0
    assert result.error == ""


def test_close_calls_hook_once_per_close() -> None:
    closed: list[bool] = []
    scorer = FunctionScorer(
        "t", "taste_model", frozenset({"text"}), lambda _: Score(1.0), on_close=lambda: closed.append(True)
    )
    scorer.close()
    _scorer({"text"}).close()  # no hook: no error
    assert closed == [True]


def test_function_scorer_satisfies_protocol() -> None:
    assert isinstance(_scorer({"text"}), Scorer)


@pytest.mark.parametrize("accepts", [set(), {"txt"}])
def test_unknown_or_empty_accepts_fail_at_construction(accepts: set[str]) -> None:
    with pytest.raises(ConfigError, match="unknown input kinds"):
        _scorer(accepts)


def test_fake_text_client_needs_responses() -> None:
    from hone_taste.testing import FakeTextClient

    with pytest.raises(ConfigError, match="at least one response"):
        FakeTextClient([])
