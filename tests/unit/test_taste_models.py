from pathlib import Path
from types import SimpleNamespace

import pytest

import hone_taste as tt
from hone_taste import normalize
from hone_taste.errors import ConfigError, MissingExtra, ModelUnavailable
from hone_taste.registry import require_modules
from hone_taste.scorers.model import resolve_device
from hone_taste.scorers.songeval import default_checkpoint, download
from hone_taste.testing import FakeGpuLease, FakeTasteModel, contracts


def test_registry_lists_every_wrapped_model() -> None:
    assert set(tt.models()) == {
        "songeval",
        "reward_model",
        "pickscore",
        "hpsv3",
        "audiobox",
        "binoculars",
        "dinov2_small",
    }
    for info in tt.models().values():
        assert info.license
        assert info.source_data
        assert info.url.startswith("https://")
        assert info.vram_gb > 0
    with pytest.raises(ConfigError, match="unknown model 'nope'"):
        tt.model_info("nope")


def test_require_modules_names_the_extra() -> None:
    require_modules("text", "json")
    with pytest.raises(MissingExtra, match=r"no_such_module_xyz not installed.*hone-taste\[text\]"):
        require_modules("text", "json", "no_such_module_xyz")


def test_gpu_leases_pass_the_contract() -> None:
    contracts.check_gpu_lease(tt.NullGpuLease())
    contracts.check_gpu_lease(FakeGpuLease())


def test_scorer_passes_the_trace_to_the_lease_and_records_model_id() -> None:
    gpu = FakeGpuLease()
    with tt.recording(tt.MemorySink()) as sink:
        tt.audiobox(gpu=gpu, model=FakeTasteModel({"CE": 5.0, "CU": 5.0, "PC": 5.0, "PQ": 5.0}))("a.wav")
    span = sink.spans[0]
    assert gpu.calls[0]["trace"]["traceparent"] == f"00-{span['trace_id']}-{span['span_id']}-01"
    assert span["attributes"]["hone.taste.model_id"] == "facebook/audiobox-aesthetics"
    assert span["attributes"]["hone.taste.family"] == "taste_model"


def test_fake_taste_model_refuses_predict_before_load() -> None:
    with pytest.raises(RuntimeError, match="before load"):
        FakeTasteModel().predict("x")


@pytest.mark.parametrize("weights", [{"tempo": 1.0}, {"coherence": -1.0}, {"coherence": 0.0}])
def test_bad_dimension_weights(weights: dict[str, float]) -> None:
    with pytest.raises(ConfigError, match="weights must name dimensions"):
        tt.songeval(weights=weights, model=FakeTasteModel(), accept_license=True)


def test_reward_model_options(tmp_path: Path) -> None:
    fake = FakeTasteModel({"reward": 1.0})
    with pytest.raises(ConfigError, match="pass reference="):
        tt.reward_model("my/other-rm", model=fake)
    ref = tmp_path / "ref.json"
    ref.write_text('{"good": [2, 3], "bad": [0, 1]}')
    other = tt.reward_model("my/other-rm", model=fake, reference=ref, normalization="percentile")
    assert other.model_id == "my/other-rm"
    assert "my/other-rm" in other.license
    assert other({"prompt": "p", "response": "r"}).details["normalization"] == "percentile"
    with pytest.raises(ConfigError, match="unknown normalization mode"):
        tt.reward_model(model=fake, reference=ref, normalization="zscore")


def test_image_preference_kinds(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="unknown image preference model"):
        tt.image_preference("clip")
    ref = tmp_path / "ref.json"
    ref.write_text('{"good": [9, 11], "bad": [2, 5]}')
    hps = tt.image_preference("hpsv3", model=FakeTasteModel({"score": 10.0}), reference=ref)
    assert hps.model_id == "MizzenAI/HPSv3"
    with pytest.raises(MissingExtra, match="pip install hpsv3"):
        tt.image_preference("hpsv3", reference=ref)


def test_download_is_atomic(tmp_path: Path) -> None:
    source = tmp_path / "weights.bin"
    source.write_bytes(b"\x00" * 10)
    target = tmp_path / "cache" / "model.safetensors"
    download(source.as_uri(), target)
    assert target.read_bytes() == b"\x00" * 10
    with pytest.raises(ModelUnavailable, match="could not download"):
        download((tmp_path / "missing.bin").as_uri(), tmp_path / "cache" / "other.safetensors")
    assert not list((tmp_path / "cache").glob("*.partial"))


def test_binoculars_needs_a_reachable_threshold() -> None:
    scorer = tt.binoculars(model=FakeTasteModel({"score": 0.5}), threshold=0.5)
    result = scorer("text")
    assert result.value == 0.5
    assert result.confidence is None  # the fake did not report a token count


def test_resolve_device() -> None:
    no_gpu = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
    gpu = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True))
    assert resolve_device(no_gpu, None) == "cpu"
    assert resolve_device(gpu, None) == "cuda"
    assert resolve_device(gpu, "cuda:1") == "cuda:1"


def test_real_backends_are_created_without_loading(tmp_path: Path) -> None:
    ref = tmp_path / "ref.json"
    ref.write_text('{"good": [2, 3], "bad": [0, 1]}')
    factories = [
        lambda: tt.songeval(accept_license=True, checkpoint=tmp_path / "none.safetensors"),
        tt.audiobox,
        lambda: tt.reward_model(reference=ref),
        lambda: tt.image_preference(reference=ref, accept_license=True),
        tt.binoculars,
        lambda: tt.character_consistency(tmp_path / "sheet.png"),
    ]
    for factory in factories:
        try:
            scorer = factory()
        except MissingExtra as exc:
            pytest.skip(str(exc))
        scorer.close()  # never loaded: nothing to release, nothing downloaded
    assert not (tmp_path / "none.safetensors").exists()


def test_a_user_chosen_binoculars_pair_is_not_gated_and_names_its_models() -> None:
    scorer = tt.binoculars("my/observer", "my/performer", model=FakeTasteModel({"score": 0.5}), threshold=0.5)
    assert scorer.model_id == "my/observer + my/performer"
    assert scorer.license == "see the model cards of my/observer + my/performer"
    assert scorer("some text").details["model_id"] == "my/observer + my/performer"


def test_reference_accepts_a_reference_set_object() -> None:
    ref = normalize.ReferenceSet("text", "mine", (0.0, 10.0))
    scorer = tt.reward_model(model=FakeTasteModel({"reward": 5.0}), reference=ref, normalization="percentile")
    result = scorer({"prompt": "p", "response": "r"})
    assert (result.value, result.details["reference_set"], result.details["raw"]) == (0.5, "text/mine", 5.0)


def test_songeval_checkpoint_lives_under_hone_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HONE_HOME", str(tmp_path))
    assert default_checkpoint() == tmp_path / "taste" / "models" / "songeval" / "model.safetensors"
