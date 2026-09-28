"""`hone-taste` command line (extra `cli`): score files, ask for picks, check agreement, list models.

    hone-taste score --scorer slop --domain lyrics song.txt
    hone-taste profile ask --name ana --items lyrics/
    hone-taste agreement --name ana --scorer slop
    hone-taste models

Exit codes: 0 ok, 1 when some file could not be scored, 2 for usage / configuration errors.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Annotated, Any

try:
    import typer
except ImportError:  # pragma: no cover - the console script is installed even without the extra
    sys.exit("the hone-taste command needs the cli extra: pip install 'hone-taste[cli]'")

import hone_taste as tt
from hone_taste.errors import HoneTasteError

app = typer.Typer(help="Stand-in scorers for human judgment.", no_args_is_help=True)
profile_app = typer.Typer(help="Personal taste profiles.", no_args_is_help=True)
app.add_typer(profile_app, name="profile")

TEXT_SUFFIXES = {".txt", ".md"}

# name -> factory(domain, accept_license)
SCORERS: dict[str, Callable[[str, bool], tt.Scorer]] = {
    "slop": lambda domain, _: tt.slop_score(domain=domain),
    "binoculars": lambda _, accept: tt.binoculars(accept_license=accept),
    "songeval": lambda _, accept: tt.songeval(accept_license=accept),
    "audiobox": lambda _, accept: tt.audiobox(accept_license=accept),
}

ScorerOption = Annotated[str, typer.Option("--scorer", help=f"one of {', '.join(SCORERS)}")]
DomainOption = Annotated[str, typer.Option(help="slop preset: general, lyrics or email")]
JsonOption = Annotated[bool, typer.Option("--json", help="print JSON")]


def _fail(message: str) -> typer.Exit:
    typer.echo(f"error: {message}", err=True)
    return typer.Exit(2)


def _scorer(name: str, domain: str = "general", accept_license: bool = False) -> tt.Scorer:
    if name not in SCORERS:
        raise _fail(f"unknown scorer {name!r}; choose one of {', '.join(SCORERS)}")
    try:
        return SCORERS[name](domain, accept_license)
    except HoneTasteError as exc:
        raise _fail(str(exc)) from exc


def _score_file(scorer: tt.Scorer, path: Path) -> dict[str, Any]:
    """One output row; a text file that is not UTF-8 is reported, not raised."""
    try:
        content = path.read_text(encoding="utf-8") if "text" in scorer.accepts else str(path)
    except UnicodeDecodeError as exc:
        return {"file": str(path), "value": None, "reason": "", "error": f"not UTF-8 text: {exc}"}
    result = scorer(content)
    return {"file": str(path), "value": result.value, "reason": result.reason, "error": result.error}


@app.command()
def score(
    files: Annotated[list[Path], typer.Argument(exists=True, dir_okay=False, help="files to score")],
    scorer: ScorerOption = "slop",
    domain: DomainOption = "general",
    accept_license: Annotated[bool, typer.Option(help="accept an unclear model license")] = False,
    as_json: JsonOption = False,
) -> None:
    """Score text files (slop, binoculars) or audio files (songeval, audiobox)."""
    chosen = _scorer(scorer, domain, accept_license)
    rows = [_score_file(chosen, path) for path in files]
    if as_json:
        typer.echo(json.dumps(rows, indent=2))
    else:
        for row in rows:
            value = "  -  " if row["value"] is None else f"{row['value']:.3f}"
            typer.echo(f"{value}  {row['file']}  {row['error'] or row['reason']}")
    if any(row["value"] is None for row in rows):
        raise typer.Exit(1)


def _items(folder: Path) -> list[str]:
    files = sorted(p for p in folder.iterdir() if p.suffix in TEXT_SUFFIXES)
    if len(files) < 2:
        raise _fail(f"{folder} needs at least two .txt / .md files to compare")
    return [p.read_text(encoding="utf-8").strip() for p in files]


@profile_app.command("ask")
def profile_ask(
    name: Annotated[str, typer.Option(help="profile name")],
    items: Annotated[Path, typer.Option(exists=True, file_okay=False, help="folder of .txt / .md items")],
    scorer: ScorerOption = "slop",
    domain: DomainOption = "general",
    budget: Annotated[int, typer.Option(help="how many pairs to ask about")] = 5,
) -> None:
    """Ask about the closest-call pairs among the items and record the picks."""
    chosen = _scorer(scorer, domain)
    try:
        me = tt.profile(name)
        pairs = me.questions(_items(items), {scorer: chosen}, budget=budget)
        picks = me.ask(pairs)
    except HoneTasteError as exc:
        raise _fail(str(exc)) from exc
    typer.echo(f"recorded {len(picks)} pick(s) in profile {name!r} ({len(me.picks)} in total)")


@app.command()
def agreement(
    name: Annotated[str, typer.Option(help="profile whose picks to compare with")],
    scorer: Annotated[list[str], typer.Option("--scorer", help="scorer(s) to check")] = ["slop"],  # noqa: B006 - typer reads the default
    domain: DomainOption = "general",
    as_json: JsonOption = False,
) -> None:
    """How often each scorer prefers what the person picked (text picks only)."""
    scorers = {n: _scorer(n, domain) for n in scorer}
    try:
        pairs = tt.profile(name).pairs
        if not pairs:
            raise HoneTasteError(f"profile {name!r} has no picks; run: hone-taste profile ask --name {name}")
        report = tt.agreement(scorers, pairs)
    except HoneTasteError as exc:
        raise _fail(str(exc)) from exc
    typer.echo(report.to_json() if as_json else str(report))


@app.command()
def models(as_json: JsonOption = False) -> None:
    """The wrapped models: whose taste they learned and their license."""
    infos = sorted(tt.models().values(), key=lambda info: info.name)
    if as_json:
        keys = ("name", "model_id", "license", "license_clear", "source_data", "extra")
        typer.echo(json.dumps([{k: getattr(i, k) for k in keys} for i in infos], indent=2))
        return
    for info in infos:
        gate = "" if info.license_clear else "  [needs --accept-license]"
        typer.echo(f"{info.name} ({info.model_id}), extra '{info.extra}'{gate}\n  license: {info.license}")
        typer.echo(f"  learned from: {info.source_data}")


if __name__ == "__main__":  # pragma: no cover
    app()
