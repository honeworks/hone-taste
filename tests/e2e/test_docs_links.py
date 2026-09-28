"""Relative links in the README, contributor files, docs, examples index and design docs resolve."""

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[2]
PAGES = sorted(
    [
        *(ROOT / name for name in ("README.md", "CONTRIBUTING.md", "AGENTS.md", "CHANGELOG.md")),
        *(ROOT / "docs").glob("*.md"),
        ROOT / "examples" / "README.md",
        *(ROOT / "design").rglob("*.md"),
    ]
)
LINK = re.compile(r"\[[^\]]*\]\(([^)\s]+)\)")
FENCE = re.compile(r"^```.*?^```", re.MULTILINE | re.DOTALL)
# The README uses absolute GitHub links (PyPI shows it too); they are checked as local paths.
REPOSITORY = re.compile(r"https://github\.com/honeworks/hone-taste/(?:blob|tree)/main/")
REMOVED = ("SPEC.md", "KICKOFF.md", "STATUS.md", "DECISIONS.md", "design/family", "design/background")


def anchors(path: Path) -> set[str]:
    """GitHub-style heading anchors of a Markdown file."""
    text = FENCE.sub("", path.read_text(encoding="utf-8"))
    headings = re.findall(r"^#+ (.+)$", text, flags=re.MULTILINE)
    return {re.sub(r"[^\w\- ]", "", heading.strip().lower()).replace(" ", "-") for heading in headings}


def links(path: Path) -> list[str]:
    text = FENCE.sub("", path.read_text(encoding="utf-8"))
    targets = [REPOSITORY.sub("", target) for target in LINK.findall(text)]
    return [target for target in targets if not re.match(r"[a-z]+:", target)]


def test_pages_exist() -> None:
    assert all(page.is_file() for page in PAGES)
    assert {"current.md", "decisions.md", "0001-initial-design.md", "0000-research.md"} <= {
        p.name for p in PAGES
    }


@pytest.mark.parametrize("page", PAGES, ids=lambda p: str(p.relative_to(ROOT)))
def test_relative_links_resolve(page: Path) -> None:
    for target in links(page):
        file, _, anchor = target.partition("#")
        base = ROOT if page == ROOT / "README.md" else page.parent
        resolved = (base / file).resolve() if file else page
        assert resolved.exists(), f"{page.relative_to(ROOT)}: broken link {target}"
        if anchor and resolved.suffix == ".md":
            assert anchor in anchors(resolved), f"{page.relative_to(ROOT)}: missing anchor {target}"


def test_readme_links_are_absolute() -> None:
    """PyPI renders the README as the project description, where relative links break."""
    text = FENCE.sub("", (ROOT / "README.md").read_text(encoding="utf-8"))
    relative = [t for t in LINK.findall(text) if not re.match(r"[a-z]+:|#", t)]
    assert not relative, relative
    assert REPOSITORY.search(text)


@pytest.mark.parametrize("page", PAGES, ids=lambda p: str(p.relative_to(ROOT)))
def test_no_links_to_removed_build_files(page: Path) -> None:
    for target in links(page):
        assert not any(name in target for name in REMOVED), f"{page.relative_to(ROOT)}: {target}"
