"""AC-22: character_consistency ranks a picture of the reference character above a nicer picture of
someone else, compares with the closest panel of a character sheet, and works through for_select."""

from dataclasses import dataclass, field
from typing import Any

import hone_taste as tt
from hone_taste.testing import FakeEmbedder

# A three-panel sheet (front, three-quarter, side); candidates are close to one panel or to nobody.
VECTORS = {
    "teacher-sheet.png#0": [1.0, 0.1, 0.0, 0.0],
    "teacher-sheet.png#1": [0.7, 0.7, 0.1, 0.0],
    "teacher-sheet.png#2": [0.1, 1.0, 0.0, 0.0],
    "side-view-same-teacher.png": [0.15, 0.95, 0.05, 0.1],
    "beautiful-other-person.png": [0.1, 0.0, 1.0, 0.4],
}


@dataclass
class Candidate:
    id: str
    data: dict[str, Any] = field(default_factory=dict[str, Any])
    files: dict[str, str] = field(default_factory=dict[str, str])
    meta: dict[str, Any] = field(default_factory=dict[str, Any])


def test_ac22_character_consistency() -> None:
    scorer = tt.character_consistency("teacher-sheet.png", panels=3, model=FakeEmbedder(VECTORS))
    same = scorer("side-view-same-teacher.png")
    other = scorer("beautiful-other-person.png")
    assert same.value is not None
    assert other.value is not None
    assert same.value > 0.9
    assert other.value < 0.1
    assert same.details["panel"] == 2  # matched the side view of the sheet

    pick = tt.for_select(scorer, file="image", record=False)
    assert pick(Candidate("c1", files={"image": "side-view-same-teacher.png"})).value == same.value
    assert pick(Candidate("c2", files={"image": "no-such.png"})).value is None
