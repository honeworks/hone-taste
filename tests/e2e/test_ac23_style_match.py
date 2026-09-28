"""AC-23: style match. Stylometry ranks held-out real posts above a generic post and an exaggerated
imitation; a judge that rewards the imitation gets weight 0 in the calibration; style_match with the
calibrated weights ranks like stylometry and falls back to it when the judge fails."""

from pathlib import Path
from typing import Any

import hone_taste as tt
from hone_taste.testing import FakeDecisionClient

STYLE = Path(__file__).parents[1] / "fixtures" / "style"


def texts(kind: str) -> list[str]:
    return [p.read_text(encoding="utf-8") for p in sorted(STYLE.glob(f"{kind}-*.md"))]


def dazzled_judge(state: Any, name: str, question: Any) -> dict[str, Any]:
    """Like a style-sheet judge: the more drama, the more it "reads like the author"."""
    drama = min(1.0, (state.count("—") + state.count("?") + state.count("**")) / 6)
    value = drama if name != "exaggerates" else 1 - drama
    return {"type": question["type"], "value": value, "raw": 1 + 4 * value, "rationale": "scripted"}


def test_ac23_style_match() -> None:
    reference, real = texts("author"), texts("real")
    imitations = {"generic": texts("generic"), "ghost": texts("ghost")}
    client = FakeDecisionClient(dazzled_judge)
    scorers = {"stylometry": tt.stylometry(reference), "style_judge": tt.style_judge(reference, client)}

    calibration = tt.calibrate_style(scorers, real, imitations)
    assert calibration.report.scorers["stylometry"].rate == 1.0
    assert (calibration.report.scorers["style_judge"].correlation or 0) < 0
    assert calibration.weights == {"stylometry": 1.0, "style_judge": 0.0}

    match = tt.style_match(reference, client, weights=calibration.weights)
    for r, g, h in zip(real, imitations["generic"], imitations["ghost"], strict=True):
        assert (match(r).value or 0) > max(match(g).value or 0, match(h).value or 0)

    fingerprint = tt.Fingerprint.from_dict(tt.fingerprint(reference).to_dict())  # stored with a profile
    broken = tt.style_match(reference, FakeDecisionClient(answers={}), fingerprint=fingerprint)
    result = broken(real[0])
    assert result.value == scorers["stylometry"](real[0]).value
    assert result.details["parts"]["style_judge"]["error"]
