"""AC-9: normalization with a reference set: values in [0, 1]; percentile mode; raw kept."""

import json
from pathlib import Path

import pytest

import hone_taste as tt


def test_ac9_normalization(tmp_path: Path) -> None:
    path = tmp_path / "judge.json"
    path.write_text(
        json.dumps({"source": "my rated examples", "good": [2.0, 3.5, 4.0, 6.0], "bad": [-4.0, -1.0, 0.0]})
    )
    ref = tt.normalize.reference_set("text", "my_judge", path=path)

    raws = [-100.0, -4.0, -1.0, 0.0, 1.0, 3.5, 6.0, 100.0]
    linear = [tt.normalize.linear(raw, ref) for raw in raws]
    assert all(0.0 <= v <= 1.0 for v in linear)
    assert linear == sorted(linear)  # order preserved
    assert linear[0] == 0.0  # clamped outside the 5th..95th percentile
    assert linear[-1] == 1.0

    pct = tt.normalize.normalized(1.0, ref, mode="percentile")
    assert pct.value == pytest.approx(3 / 7)  # 3 of 7 reference values lie below 1.0
    assert tt.normalize.percentile(1.0, ref) == pct.value

    assert pct.details["raw"] == 1.0
    assert pct.details["reference_set"] == "text/my_judge"
    assert pct.details["normalization"] == "percentile"
