"""Run the README quickstart with only the wheel installed (called by scripts/check.sh)."""

import re
import tempfile
from pathlib import Path

readme = (Path(__file__).parents[1] / "README.md").read_text(encoding="utf-8")
code = re.search(r"## Quickstart\n```python\n(.*?)```", readme, re.DOTALL)
assert code, "README has no quickstart block"
import os  # noqa: E402

os.environ["HONE_HOME"] = tempfile.mkdtemp(prefix="hone-taste-smoke-")
exec(compile(code.group(1), "README quickstart", "exec"), {})  # noqa: S102
print("quickstart ok")
