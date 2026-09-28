"""AC-19 (offline half): reward_model and songeval produce values in [0, 1] from their raw outputs.

The real-model half (Skywork-Reward-V2, SongEval weights) runs in tests/gpu/test_ac19_taste_models.py.
"""

import pytest

import hone_taste as tt
from hone_taste.testing import FakeTasteModel


@pytest.mark.parametrize("reward", [-50.0, -1.5, 0.0, 3.2, 50.0])
def test_ac19_reward_model_values_in_range(reward: float) -> None:
    scorer = tt.reward_model(model=FakeTasteModel({"reward": reward}))  # packaged reference set
    s = scorer({"prompt": "What is 2 + 2?", "response": "4"})
    assert s.value is not None
    assert 0.0 <= s.value <= 1.0
    assert s.details["raw"] == reward
    assert s.details["reference_set"] == "text/reward_model"


def test_ac19_reward_model_orders_by_raw_reward() -> None:
    scorer = tt.reward_model(
        model=FakeTasteModel(lambda pair: {"reward": 10.0 if "Paris" in pair["response"] else -10.0})
    )
    good = scorer({"prompt": "Capital of France?", "response": "Paris."})
    bad = scorer({"prompt": "Capital of France?", "response": "Bananas."})
    assert good.value == 1.0
    assert bad.value == 0.0


@pytest.mark.parametrize("level", [1.0, 2.5, 5.0])
def test_ac19_songeval_values_in_range(level: float) -> None:
    raw = dict.fromkeys(["coherence", "musicality", "memorability", "clarity", "naturalness"], level)
    s = tt.songeval(model=FakeTasteModel(raw), accept_license=True)("melody.wav")
    assert s.value == pytest.approx((level - 1) / 4)
    assert all(0.0 <= v <= 1.0 for v in s.details["dimensions"].values())
