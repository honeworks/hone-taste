"""AC-14: a model with an unclear license needs accept_license=True."""

import pytest

import hone_taste as tt
from hone_taste.testing import FakeTasteModel


def test_ac14_license_gate(tmp_path) -> None:
    assert tt.model_info("pickscore").license_clear is False
    with pytest.raises(tt.errors.LicenseNotAccepted, match="accept_license=True") as info:
        tt.image_preference("pickscore", model=FakeTasteModel({"score": 1.0}))
    assert isinstance(info.value, tt.errors.MissingExtra)  # design: a MissingExtra / license error
    assert "huggingface.co/yuvalkirstain/PickScore_v1" in str(info.value)
    with pytest.raises(tt.errors.LicenseNotAccepted, match="non-commercial") as info:
        tt.songeval(model=FakeTasteModel())
    message = str(info.value)
    assert message.count(tt.model_info("songeval").license) == 1
    assert "https://github.com/ASLP-lab/SongEval" in message
    assert "unclear: unclear" not in message  # read as one sentence, not "license is unclear: unclear: ..."

    ref = tmp_path / "ref.json"
    ref.write_text('{"good": [20, 22], "bad": [15, 16]}')
    scorer = tt.image_preference(
        "pickscore", model=FakeTasteModel({"score": 21.0}), reference=ref, accept_license=True
    )
    assert scorer({"prompt": "a barn", "image": "barn.png"}).value is not None

    for name in ("reward_model", "audiobox", "binoculars", "hpsv3"):  # clear terms: no gate
        assert tt.model_info(name).license_clear
    assert (
        tt.audiobox(model=FakeTasteModel({"CE": 5.0, "CU": 5.0, "PC": 5.0, "PQ": 5.0}))("a.wav").value == 0.5
    )
