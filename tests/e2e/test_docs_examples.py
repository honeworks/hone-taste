"""README and docs/*.md code blocks run as written (with fakes); examples/ are in test_ac20_examples.py."""

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[2]
BLOCK = re.compile(r"(<!-- not executed[^>]*-->\s*)?```python\n(.*?)```", re.DOTALL)
DOCS = [ROOT / "README.md", *sorted((ROOT / "docs").glob("*.md"))]


def python_blocks(path: Path) -> list[str]:
    return [code for skip, code in BLOCK.findall(path.read_text(encoding="utf-8")) if not skip]


@pytest.mark.parametrize("path", DOCS, ids=lambda p: p.name)
def test_doc_blocks_run(path: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HONE_HOME", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    for code in python_blocks(path):
        exec(compile(code, str(path), "exec"), {})  # noqa: S102 - running our own documentation


def test_every_doc_has_runnable_code() -> None:
    assert python_blocks(ROOT / "README.md")
    assert {p.name for p in DOCS} >= {"README.md", "guide.md", "records.md", "adapters.md", "cli.md"}


def test_readme_license_table_matches_the_registry() -> None:
    import hone_taste as tt

    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    table = readme.split("## Models and licenses", 1)[1].split("\n## ", 1)[0]
    rows = [line for line in table.splitlines() if line.startswith("| `")]
    assert len(rows) == len(tt.models())
    for info in tt.models().values():
        row = next(r for r in rows if info.model_id.split(" ")[0] in r)
        assert ("accept_license=True" in row) == (not info.license_clear), info.name
    for phrase in ("## Honesty rules", "signals, never gates", "Personas are one LLM", "Goodhart"):
        assert phrase in readme


def test_example_links_in_the_docs_point_to_existing_files() -> None:
    linked = {
        name for path in DOCS for name in re.findall(r"examples/(\w+\.py)", path.read_text(encoding="utf-8"))
    }
    assert "quickstart.py" in linked
    assert all((ROOT / "examples" / name).is_file() for name in linked), linked
