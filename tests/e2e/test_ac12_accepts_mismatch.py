"""AC-12: an input of the wrong kind gives Score(None, error=...), never an exception."""

from pathlib import Path

import hone_taste as tt


def test_ac12_accepts_mismatch() -> None:
    combined = tt.combine({"slop": 1}, {"slop": tt.slop_score()})
    for scorer in (tt.slop_score(), tt.patterns(["x"]), combined):
        for wrong in ({"prompt": "p", "response": "r"}, Path("song.wav"), 42, None):
            result = scorer(wrong)
            assert result.value is None
            assert result.error.startswith("accepts text; got ")
            assert result.details == {}
