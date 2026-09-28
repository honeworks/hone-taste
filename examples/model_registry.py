"""The model registry: whose taste each wrapped model learned, its license, and what it needs.

What: `tt.models()` returns every heavy model hone-taste can wrap as a `tt.ModelInfo` (`model_id`, `url`,
      `license`, `license_clear`, `source_data`, `vram_gb`, `extra`); `tt.model_info(name)` returns one.
      The data ships with the package (`hone_taste/data/models.toml`) and is the same table the README
      and `hone-taste models` print.
How:  1. list the models and print whose ratings each one learned and under which license,
      2. find the ones that need `accept_license=True` (`license_clear` is False),
      3. look up one model before building its scorer: which extra to install, how much VRAM to lease,
      4. an unknown name raises `ConfigError` listing the known ones.
Why:  a taste model is only as good a stand-in as the people it learned from, and some weights may not
      be used commercially. Checking this in code (for example in a startup check or a CI job that fails
      on non-commercial models) is better than remembering it. Reading the registry imports no heavy
      dependency and downloads nothing.

Run: uv run python examples/model_registry.py
"""

import textwrap

import hone_taste as tt
from hone_taste.errors import ConfigError

# 1. Every wrapped model, with whose taste it learned.
for name, info in sorted(tt.models().items()):
    print(f"{name:13} {info.model_id:38} extra={info.extra:7} vram={info.vram_gb:>4} GB")
    print(f"{'':13} learned from: {textwrap.shorten(info.source_data, 80)}")

# 2. The models that need accept_license=True before use.
gated = sorted(name for name, info in tt.models().items() if not info.license_clear)
print("needs accept_license=True:", gated)
assert gated == ["pickscore", "songeval"]

# 3. One model before building its scorer: what to install and how much GPU memory to reserve.
info = tt.model_info("reward_model")
print(
    f"reward_model: pip install 'hone-taste[{info.extra}]', lease {info.vram_gb} GB, license {info.license}"
)
assert info.license_clear and info.extra == "text"

# A policy check you could run in CI: no non-commercial weights in this product.
non_commercial = [name for name, info in tt.models().items() if "non-commercial" in info.license.lower()]
print("non-commercial terms mentioned:", non_commercial)
assert non_commercial == ["songeval"]

# 4. Unknown names fail loudly with the list of known ones.
try:
    tt.model_info("musecritic")
except ConfigError as exc:
    print("unknown:", exc)
else:
    raise AssertionError("expected a ConfigError")
