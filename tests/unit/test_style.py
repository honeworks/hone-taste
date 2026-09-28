"""stylometry, style_judge, style_match and calibrate_style (design/changes/0004)."""

import json
from pathlib import Path
from typing import Any

import pytest

import hone_taste as tt
from hone_taste.scorers.stylometry import features
from hone_taste.style import pick_excerpts
from hone_taste.testing import FakeDecisionClient, FakeTextClient
from hone_taste.types import FunctionScorer

STYLE = Path(__file__).parents[1] / "fixtures" / "style"


def texts(kind: str) -> list[str]:
    return [p.read_text(encoding="utf-8") for p in sorted(STYLE.glob(f"{kind}-*.md"))]


REFERENCE = texts("author")


def test_features_count_the_surface() -> None:
    text = "# Title\n\nI don't know (yet). Do you?\n\n- one\n- two\n\n**bold** words here\n\n```c\nx;\n```"
    f = features(text)
    assert f["headings_per_1000"] > 0
    assert f["list_items_per_1000"] > 0
    assert f["bold_per_1000"] > 0
    assert f["contractions"] > 0
    assert f["parentheses"] > 0
    assert f["questions_per_sentence"] == pytest.approx(1 / 3)  # the bold line is a sentence too
    assert 0 < f["code_share"] < 1
    with pytest.raises(ValueError, match="no words"):
        features("```\ncode only\n```")


def test_fingerprint_needs_two_texts_with_words() -> None:
    with pytest.raises(tt.errors.ConfigError, match="at least two"):
        tt.fingerprint(REFERENCE[:1])
    with pytest.raises(tt.errors.ConfigError, match="needs words"):
        tt.fingerprint([REFERENCE[0], "  "])


def test_fingerprint_round_trips_through_json() -> None:
    fp = tt.fingerprint(REFERENCE)
    again = tt.Fingerprint.from_dict(json.loads(json.dumps(fp.to_dict())))
    assert again == fp
    assert tt.stylometry(again)(texts("real")[0]).value == tt.stylometry(fp)(texts("real")[0]).value


def test_stylometry_ranks_the_author_above_imitations() -> None:
    scorer = tt.stylometry(REFERENCE)
    assert scorer.family == "personal"
    real = [scorer(t) for t in texts("real")]
    others = [scorer(t) for t in texts("generic") + texts("ghost")]
    assert min(r.value or 0 for r in real) > max(o.value or 0 for o in others)
    details = others[0].details
    assert set(details) >= {"surface", "function_words", "delta", "typical_delta", "furthest"}
    assert details["reference_texts"] == 8
    assert "z=" in others[0].reason
    assert scorer("").error  # nothing to measure: an error score, never 0


def test_pick_excerpts_round_robin() -> None:
    long_a, long_b = "a " * 40, "b " * 40
    picked = pick_excerpts([f"{long_a}\n\n{long_a}x", f"{long_b}"], count=3)
    assert [p[0] for p in picked] == ["a", "b", "a"]
    assert pick_excerpts(["# only a heading"]) == []
    assert len(pick_excerpts(["word " * 400]).pop().split()) == 150  # no good paragraph: trimmed


def answers(voice: float | None, plain: float | None, exaggerates: float | None):
    table = {"voice": voice, "plain": plain, "exaggerates": exaggerates}

    def rule(state: Any, name: str, question: Any) -> dict[str, Any]:
        value = table[name]
        if value is None:
            return {"type": question["type"], "value": None, "error": "timeout"}
        return {"type": question["type"], "value": value, "raw": 4, "rationale": f"{name} rationale"}

    return rule


def test_style_judge_asks_with_real_excerpts_and_inverts_exaggeration() -> None:
    client = FakeDecisionClient(answers(0.75, 1.0, 1.0))
    judge = tt.style_judge(REFERENCE, client, author="Pat", excerpts=3)
    result = judge("Some draft.")
    assert result.value == pytest.approx((0.75 + 1.0 + 0.0) / 3)
    assert result.details["excerpts"] == 3
    assert result.details["voice_raw"] == 4
    assert result.reason == "voice rationale"
    questions = client.calls[0]["questions"]
    assert set(questions) == {"voice", "plain", "exaggerates"}
    assert "all written by Pat" in questions["voice"]["instructions"]
    assert pick_excerpts(REFERENCE, 3)[2] in questions["voice"]["instructions"]
    assert judge.family == "audience"


def test_style_judge_partial_and_total_failures() -> None:
    partial = tt.style_judge(REFERENCE, FakeDecisionClient(answers(None, 1.0, 0.0)))("x")
    assert partial.value == 1.0
    assert partial.details["errors"] == ["voice: timeout"]
    failed = tt.style_judge(REFERENCE, FakeDecisionClient(answers={}))("x")
    assert failed.value is None
    assert "nothing usable" in failed.error


def test_style_judge_clips_long_texts_and_wraps_text_clients() -> None:
    client = FakeDecisionClient()
    tt.style_judge(REFERENCE, client)("w " * 3000)
    assert len(client.calls[0]["state"].split()) == 2500
    reply = json.dumps(
        {
            k: {"answer": v, "rationale": "r"}
            for k, v in {"voice": 5, "plain": True, "exaggerates": False}.items()
        }
    )
    assert tt.style_judge(REFERENCE, FakeTextClient([reply]))("x").value == 1.0
    with pytest.raises(tt.errors.ConfigError, match="prose paragraph"):
        tt.style_judge(["# heading only"], client)


def test_style_match_combines_both_halves_with_weights() -> None:
    fp = tt.fingerprint(REFERENCE)
    match = tt.style_match(
        REFERENCE,
        FakeDecisionClient(answers(1.0, 1.0, 0.0)),
        fingerprint=fp,
        weights={"stylometry": 3, "style_judge": 1},
    )
    assert match.name == "style_match"
    assert match.family == "personal"
    real = texts("real")[0]
    expected = (3 * (tt.stylometry(fp)(real).value or 0) + 1 * 1.0) / 4
    result = match(real)
    assert result.value == pytest.approx(expected)
    assert set(result.details["parts"]) == {"stylometry", "style_judge"}


def test_style_match_falls_back_to_stylometry_when_the_judge_fails() -> None:
    def down(state: Any, name: str, question: Any) -> dict[str, Any]:
        raise RuntimeError("judge down")

    result = tt.style_match(REFERENCE, FakeDecisionClient(down))(texts("real")[0])
    assert result.value == tt.stylometry(REFERENCE)(texts("real")[0]).value
    assert "judge down" in result.details["parts"]["style_judge"]["error"]
    with pytest.raises(tt.errors.ConfigError):
        tt.style_match(REFERENCE, FakeDecisionClient(), weights={"slop": 1})


def test_calibrate_style_weights_by_correlation_and_scores_once() -> None:
    calls: list[str] = []
    stylo = tt.stylometry(REFERENCE)

    def counted(text: str) -> tt.Score:
        calls.append(text)
        return stylo(text)

    def likes_drama(text: str) -> tt.Score:  # a sheet-style judge: prefers the exaggerated imitation
        return tt.Score(min(1.0, 0.3 + text.count("—") * 0.2 + text.count("?") * 0.1))

    scorers = {
        "stylometry": FunctionScorer("stylometry", "personal", frozenset({"text"}), counted),
        "sheet_judge": FunctionScorer("sheet_judge", "audience", frozenset({"text"}), likes_drama),
    }
    groups = {"generic": texts("generic"), "ghost": texts("ghost")}
    cal = tt.calibrate_style(scorers, texts("real"), groups)
    assert cal.report.pairs == 6
    assert cal.report.scorers["stylometry"].rate == 1.0
    assert cal.weights == {"stylometry": 1.0, "sheet_judge": 0.0}
    assert len(calls) == 9  # every text scored once
    assert list(cal.means) == ["real", "generic", "ghost"]
    assert (cal.means["real"]["stylometry"] or 0) > (cal.means["ghost"]["stylometry"] or 0)
    assert "weights: stylometry=1.00" in str(cal)
    assert json.loads(cal.to_json())["weights"] == cal.weights


def test_calibrate_style_equal_weights_when_nothing_agrees_and_config_errors() -> None:
    flat = FunctionScorer("flat", "personal", frozenset({"text"}), lambda t: tt.Score(0.5))
    cal = tt.calibrate_style({"a": flat, "b": flat}, ["x"], {"ghost": ["y"]})
    assert cal.weights == {"a": 0.5, "b": 0.5}
    assert "no scorer agreed" in cal.note
    with pytest.raises(tt.errors.ConfigError, match="one text per real text"):
        tt.calibrate_style({"a": flat}, ["x", "y"], {"ghost": ["z"]})
    with pytest.raises(tt.errors.ConfigError):
        tt.calibrate_style({}, ["x"], {"ghost": ["y"]})
