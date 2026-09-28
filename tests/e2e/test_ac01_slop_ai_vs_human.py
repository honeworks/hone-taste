"""AC-1: slop on a planted AI-ish paragraph vs a human paragraph."""

from pathlib import Path

import hone_taste as tt

FIXTURES = Path(__file__).parents[1] / "fixtures" / "text"


def test_ac1_slop_ai_vs_human() -> None:
    ai_text = (FIXTURES / "ai_paragraph.txt").read_text()
    human_text = (FIXTURES / "human_paragraph.txt").read_text()
    slop = tt.slop_score()

    ai, human = slop(ai_text), slop(human_text)
    assert ai.value is not None
    assert human.value is not None
    assert ai.value < human.value
    assert human.value > 0.8
    assert ai.value < 0.3

    words = {h["word"] for h in ai.details["hits"] if "word" in h}
    assert {"delve", "tapestry", "vibrant", "pivotal", "unwavering"} <= words
    assert any(h.get("pattern") == "not X, but Y" for h in ai.details["hits"])
    kinds = {
        next(k for k in ("word", "phrase", "pattern", "trigram") if k in hit) for hit in ai.details["hits"]
    }
    assert kinds == {"word", "pattern", "trigram"}
    spans = [hit["span"] for hit in ai.details["hits"]]
    assert spans == sorted(spans)
    for hit in ai.details["hits"]:
        start, end = hit["span"]
        assert 0 <= start < end <= len(ai_text)
        if "word" in hit:
            assert ai_text[start:end].lower() == hit["word"]
        if "pattern" in hit:
            assert ai_text[start:end] == hit["match"]
        if "trigram" in hit:
            assert ai_text[start:end].lower().startswith(hit["trigram"].split()[0])

    assert slop(ai_text) == ai  # deterministic
    assert tt.slop_score()(ai_text) == ai
