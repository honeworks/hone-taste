"""AC-20: every examples/*.py runs, opens with a What / How / Why docstring and is listed in the index."""

import ast
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

import hone_taste as tt

EXAMPLES = Path(__file__).parents[2] / "examples"
FILES = sorted(EXAMPLES.glob("*.py"))
DESIGN = Path(__file__).parents[2] / "design" / "current.md"
DESIGN_LINK = re.compile(r"\]\(\.\./design/current\.md#[\w-]+\)")

INDEX_ROW = re.compile(r"^\| \[`([\w.]+\.py)`\]\(\1\) \|.*$", re.MULTILINE)
PUBLIC_MODULES = {
    "hone_taste",
    "hone_taste.testing",
    "hone_taste.errors",
    "hone_taste.normalize",
    "hone_taste.adapters.openai",
}
# Variables that would change how an example runs (PYTHONOPTIMIZE strips every assert).
UNSAFE_ENV = ("HONE_", "PYTHONOPTIMIZE", "PYTHONWARNINGS", "PYTHONSTARTUP", "PYTHONPATH")


def required_examples() -> set[str]:
    """The examples table in design/current.md section 11, plus the entry point every reader starts from."""
    section = DESIGN.read_text(encoding="utf-8").split("## 11. Examples", 1)[1].split("\n## ", 1)[0]
    return set(re.findall(r"^\| `(\w+\.py)` \|", section, flags=re.MULTILINE)) | {"quickstart.py"}


def run_python(args: list[str], tmp_path: Path) -> subprocess.CompletedProcess[str]:
    """Run Python like a reader would: an empty folder, its own HONE_HOME, no terminal on stdin."""
    env = {k: v for k, v in os.environ.items() if not k.startswith(UNSAFE_ENV)}
    env["HONE_HOME"] = str(tmp_path / "hone")
    return subprocess.run(
        [sys.executable, *args],
        cwd=tmp_path,
        env=env,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )


def test_examples_cover_the_design_table() -> None:
    required = required_examples()
    assert len(required) >= 9
    assert {p.name for p in FILES} >= required


def test_the_runner_keeps_asserts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PYTHONOPTIMIZE", "1")
    assert run_python(["-c", "assert False"], tmp_path).returncode != 0


@pytest.mark.parametrize("path", FILES, ids=lambda p: p.name)
def test_ac20_example_runs(path: Path, tmp_path: Path) -> None:
    result = run_python([str(path)], tmp_path)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip(), "an example prints what it shows"


def test_quickstart_records_to_the_default_store(tmp_path: Path) -> None:
    assert run_python([str(EXAMPLES / "quickstart.py")], tmp_path).returncode == 0
    store = tt.SqliteSpanSink(tmp_path / "hone" / "taste" / "spans.db")
    names = [span["name"] for span in store.spans()]
    store.close()
    assert len(names) >= 4
    assert set(names) == {"hone.taste.score"}


@pytest.mark.parametrize("path", FILES, ids=lambda p: p.name)
def test_ac20_example_has_what_how_why_docstring(path: Path) -> None:
    docstring = ast.get_docstring(ast.parse(path.read_text(encoding="utf-8")))
    assert docstring, f"{path.name} must open with a docstring"
    parts = re.split(r"^(What|How|Why|Run):", docstring, flags=re.MULTILINE)
    sections = dict(zip(parts[1::2], parts[2::2], strict=True))
    assert list(sections) == ["What", "How", "Why", "Run"], f"{path.name}: What / How / Why / Run, in order"
    for name in ("What", "How", "Why"):
        assert len(sections[name].strip()) >= 40, f"{path.name}: the {name} section is too short"
    assert sections["Run"].strip() == f"uv run python examples/{path.name}"


def test_ac20_readme_indexes_every_example_once() -> None:
    rows = INDEX_ROW.findall((EXAMPLES / "README.md").read_text(encoding="utf-8"))
    assert sorted(rows) == [p.name for p in FILES]
    assert len(rows) == len(set(rows))
    order = rows.index
    assert order("quickstart.py") == 0
    assert order("lyrics_taste.py") == len(rows) - 1  # the example that combines everything comes last


def test_ac20_readme_rows_have_concept_sentence_and_design_section() -> None:
    text = (EXAMPLES / "README.md").read_text(encoding="utf-8")
    for match in INDEX_ROW.finditer(text):
        cells = [cell.strip() for cell in match.group(0).strip("|").split("|")]
        assert len(cells) == 4, match.group(0)
        assert all(cells), match.group(0)
        assert cells[2].endswith("."), "the description is one sentence"
        assert DESIGN_LINK.search(cells[3]), "a link to the design section (design/current.md#...)"


def _is_private(name: str) -> bool:
    return name.startswith("_") and not name.startswith("__")


@pytest.mark.parametrize("path", FILES, ids=lambda p: p.name)
def test_ac20_example_uses_only_the_public_api(path: Path) -> None:
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("hone_taste"):
            assert node.module in PUBLIC_MODULES, f"{path.name} imports {node.module}"
            for alias in node.names:
                assert not _is_private(alias.name), f"{path.name} imports {alias.name}"
                if node.module == "hone_taste":
                    assert alias.name in tt.__all__, f"{path.name} imports {alias.name}"
        if isinstance(node, ast.Import):
            modules = {alias.name for alias in node.names if alias.name.startswith("hone_taste")}
            assert modules <= PUBLIC_MODULES, f"{path.name} imports {modules}"
        if isinstance(node, ast.Attribute):
            assert not _is_private(node.attr), f"{path.name} uses the private name {node.attr}"
