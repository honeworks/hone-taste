import json
from pathlib import Path

import pytest

from hone_taste.errors import ConfigError
from hone_taste.scorers.slop import slop_score


def _labels(text: str, domain: str = "general") -> list[str]:
    hits = slop_score(domain)(text).details["hits"]
    return [h.get("word") or h.get("phrase") or h.get("pattern") or h["trigram"] for h in hits]


@pytest.mark.parametrize(
    ("text", "pattern"),
    [
        ("It wasn't anger, but something colder.", "not X, but Y"),
        ("This is not a drill - it's a warning.", "not X - it's Y"),
        ("It isn't a phase. It's who I am.", "it's not X. it's Y"),
        ("It\u2019s not a phase. It\u2019s who I am.", "it's not X. it's Y"),  # curly apostrophes
    ],
)
def test_contrast_patterns(text: str, pattern: str) -> None:
    assert pattern in _labels(text)


@pytest.mark.parametrize(
    "text",
    ["I did not go, but when I did it rained.", "Not only that, but the bus was late.", "It is not late."],
)
def test_contrast_guards(text: str) -> None:
    assert not any("X" in label for label in _labels(text))


def test_trigrams_match_content_words() -> None:
    labels = _labels("She took a deep breath and walked on.")
    assert "took deep breath" in labels


def test_empty_text_cannot_be_scored() -> None:
    result = slop_score()("  ... 123 ")
    assert result.value is None
    assert result.error == "no words to score"


def test_confidence_grows_with_length() -> None:
    scorer = slop_score()
    assert scorer("a plain short line").confidence == pytest.approx(0.04)
    assert scorer("word " * 300).confidence == 1.0


def test_phrase_hit_is_not_double_counted_as_word() -> None:
    labels = _labels("echoes of the past", "lyrics")
    assert labels == ["echoes of"]


def test_rates_and_metadata() -> None:
    scorer = slop_score("lyrics")
    result = scorer("delve " * 10)
    assert result.details["rates_per_1k_words"] == {"words": 1000.0, "contrast": 0.0, "trigrams": 0.0}
    assert result.value == pytest.approx(0.4)  # only the 60% word component is saturated
    assert scorer.name == "slop_lyrics"
    assert scorer.family == "human_likeness"
    assert "MIT" in scorer.license
    assert "slop-score" in scorer.source_data


def test_custom_wordlists_folder(tmp_path: Path) -> None:
    (tmp_path / "slop_list.json").write_text(json.dumps([["zorp"]]))
    (tmp_path / "slop_list_trigrams.json").write_text(json.dumps(["alpha beta gamma"]))
    scorer = slop_score(wordlists=tmp_path)
    assert _labels_of(scorer("Zorp alpha beta gamma delve")) == ["zorp", "alpha beta gamma"]


def _labels_of(result: object) -> list[str]:
    return [h.get("word") or h.get("trigram") for h in result.details["hits"]]  # type: ignore[attr-defined]


def test_missing_wordlists_folder(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match=r"slop_list\.json"):
        slop_score(wordlists=tmp_path)


def test_unknown_domain() -> None:
    with pytest.raises(ConfigError, match="unknown slop domain 'poetry'"):
        slop_score("poetry")


def test_spans_survive_curly_quotes() -> None:
    text = "\u201cIt\u2019s not a phase. It\u2019s a tapestry,\u201d she said."
    hits = slop_score()(text).details["hits"]
    tapestry = next(h for h in hits if h.get("word") == "tapestry")
    start, end = tapestry["span"]
    assert text[start:end] == "tapestry"


def test_contrast_hit_includes_the_y_word() -> None:
    hits = slop_score()("It's not just a song, but a journey.").details["hits"]
    assert hits[0]["match"] == "not just a song, but a journey"


def test_long_text_without_punctuation_is_fast() -> None:
    import time

    text = "it is not the thing that we want but " * 5000
    start = time.perf_counter()
    result = slop_score()(text)
    assert time.perf_counter() - start < 5
    assert result.value is not None
    assert 0.0 <= result.value <= 1.0


def test_same_result_across_processes() -> None:
    import os
    import subprocess
    import sys

    code = (
        "import hone_taste as tt, json;"
        "r = tt.slop_score('lyrics')(\"It's not a phase. It's a neon tapestry, a voice barely whisper.\");"
        "print(json.dumps([r.value, r.details], sort_keys=True))"
    )
    outputs = {
        subprocess.run(
            [sys.executable, "-c", code],
            env={**os.environ, "PYTHONHASHSEED": seed},
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        for seed in ("1", "2")
    }
    assert len(outputs) == 1
