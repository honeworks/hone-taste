"""The `hone-taste` command line: score files, ask for picks, check agreement, list models.

What: the CLI (extra `cli`: `pip install "hone-taste[cli]"`) runs the same scorers on files:
      `hone-taste score`, `hone-taste profile ask`, `hone-taste agreement` and `hone-taste models`.
      Exit codes: 0 ok, 1 when a file could not be scored, 2 for usage or configuration errors. This
      example runs it as `python -m hone_taste.cli ...` (the same program as `hone-taste ...`) in a
      subprocess, the way a script or CI job would.
How:  1. `score --scorer slop --domain lyrics FILES` prints value, file and reason per line; `--json` prints
         rows for other tools; a file that cannot be scored gives exit code 1,
      2. `profile ask --name NAME --items DIR` asks about the closest calls in a terminal; without a
         terminal it refuses (exit 2) instead of hanging, so scripts record picks from Python,
      3. `agreement --name NAME --scorer slop --json` checks a scorer against that profile's picks,
      4. `models` prints whose taste each model learned and its license.
      CLI and Python share `${HONE_HOME:-.hone}`: profiles written by one are read by the other.
Why:  quick checks on a folder of drafts, shell pipelines and CI gates ("fail if any lyric scores below
      0.5") without writing Python. Pitfall: `reward_model` and `image_preference` need prompt pairs, so
      they are Python-only; heavy scorers need their extra and, for SongEval, `--accept-license`.

Run: uv run python examples/command_line.py
"""

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import hone_taste as tt

NEON = "Neon echoes whisper through the tapestry of night"
BOOTS = "Dad's boots by the door still smell like diesel"
KETTLE = "The kettle clicks off and nobody gets up to pour it"

home = tempfile.TemporaryDirectory()  # kept for the whole script, removed at the end
os.environ["HONE_HOME"] = home.name  # the CLI subprocesses and tt.profile below use the same folder
drafts = Path(home.name, "drafts")
drafts.mkdir()
for name, text in {"a.txt": NEON, "b.txt": BOOTS, "c.txt": KETTLE}.items():
    Path(drafts, name).write_text(text, encoding="utf-8")


def hone_taste(*args: str) -> subprocess.CompletedProcess[str]:
    """Run `hone-taste ARGS` without a terminal attached, like a script or CI job."""
    command = [sys.executable, "-m", "hone_taste.cli", *args]
    return subprocess.run(command, capture_output=True, text=True, stdin=subprocess.DEVNULL, check=False)


# 1. Score files: one line per file, or JSON rows.
files = sorted(str(p) for p in drafts.glob("*.txt"))
result = hone_taste("score", "--scorer", "slop", "--domain", "lyrics", *files)
print(result.stdout.replace(str(drafts) + os.sep, ""), end="")
assert result.returncode == 0 and len(result.stdout.splitlines()) == 3

rows = json.loads(hone_taste("score", "--scorer", "slop", "--json", *files).stdout)
print("json row:", {k: rows[0][k] for k in ("value", "reason", "error")})
assert rows[0]["value"] < rows[1]["value"]

Path(drafts, "empty.txt").write_text("   ", encoding="utf-8")
failed = hone_taste("score", str(Path(drafts, "empty.txt")))
print(f"empty file: exit {failed.returncode}: {failed.stdout.strip()}")
assert failed.returncode == 1
assert "no words to score" in failed.stdout  # the reason is printed where the value would be

# 2. Asking needs a terminal; without one it refuses with exit code 2 and says what to do instead.
refused = hone_taste("profile", "ask", "--name", "ana", "--items", str(drafts), "--budget", "2")
print(f"profile ask without a terminal: exit {refused.returncode}: {refused.stderr.strip()[:70]}...")
assert refused.returncode == 2 and "TTY" in refused.stderr

me = tt.profile("ana")  # the same file the CLI uses: ${HONE_HOME}/taste/profiles/ana.json
me.add_pick(winner=BOOTS, loser=NEON)
me.add_pick(winner=KETTLE, loser=NEON)

# 3. How often does slop agree with this person's picks?
checked = hone_taste("agreement", "--name", "ana", "--scorer", "slop", "--json")
assert checked.returncode == 0, checked.stderr
report = json.loads(checked.stdout)
print("agreement:", report["scorers"]["slop"])
assert report["pairs"] == 2 and report["scorers"]["slop"]["rate"] == 1.0

# 4. The model table, and a usage error (exit code 2).
models = hone_taste("models")
print(models.stdout.splitlines()[0])
assert models.returncode == 0 and "accept-license" in models.stdout
assert hone_taste("score", "--scorer", "nope", files[0]).returncode == 2

home.cleanup()
