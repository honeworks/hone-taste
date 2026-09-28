"""AC-17: the core imports and works without torch / transformers installed."""

import subprocess
import sys

HEAVY = [
    "torch",
    "transformers",
    "librosa",
    "muq",
    "safetensors",
    "audiobox_aesthetics",
    "soundfile",
    "PIL",
    "numpy",
    "openai",
    "typer",
]

SCRIPT = f"""
import sys
for name in {HEAVY!r}:
    sys.modules[name] = None  # behaves as "not installed"
import hone_taste as tt
from hone_taste.testing import FakeTasteModel
assert tt.slop_score()("A plain sentence about rain.").value is not None
assert tt.audiobox(model=FakeTasteModel({{"CE": 5.0, "CU": 5.0, "PC": 5.0, "PQ": 5.0}}))("a.wav").value == 0.5
for factory, extra in [(tt.songeval, "songs"), (tt.audiobox, "songs"), (tt.reward_model, "text"),
                       (tt.binoculars, "detect"), (tt.image_preference, "images")]:
    try:
        factory(accept_license=True)
    except tt.errors.MissingExtra as exc:
        assert f"hone-taste[{{extra}}]" in str(exc), exc
    else:
        raise AssertionError(f"{{factory.__name__}} did not raise MissingExtra")
loaded = [m for m in {HEAVY!r} if sys.modules.get(m) is not None]
assert not loaded, loaded
print("ok")
"""


def test_ac17_import_boundaries() -> None:
    result = subprocess.run([sys.executable, "-c", SCRIPT], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"
