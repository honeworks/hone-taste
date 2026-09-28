"""The `hone-taste` CLI, in-process through typer's runner."""

import json
import subprocess
import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

import hone_taste as tt
from hone_taste.cli import app

runner = CliRunner()
SLOP = "It's not just a song, but a journey through the tapestry of time. Let us delve into neon echoes."
FRESH = "Rain on the tin roof, my father's boots by the door."


@pytest.fixture
def files(tmp_path: Path) -> Path:
    folder = tmp_path / "items"
    folder.mkdir()
    (folder / "a.txt").write_text(SLOP)
    (folder / "b.txt").write_text(FRESH)
    (folder / "c.md").write_text("Neon dreams and whispered echoes, a tapestry of light.")
    (folder / "ignored.wav").write_bytes(b"")
    return folder


def test_help() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for command in ("score", "profile", "agreement", "models"):
        assert command in result.output


def test_score_text_and_json(files: Path) -> None:
    result = runner.invoke(app, ["score", "--domain", "lyrics", str(files / "a.txt"), str(files / "b.txt")])
    assert result.exit_code == 0, result.output
    lines = result.output.strip().splitlines()
    assert len(lines) == 2
    assert float(lines[0].split()[0]) < float(lines[1].split()[0])

    result = runner.invoke(app, ["score", "--json", str(files / "b.txt")])
    rows = json.loads(result.output)
    assert rows[0]["file"].endswith("b.txt")
    assert 0.0 <= rows[0]["value"] <= 1.0


def test_score_errors(files: Path, tmp_path: Path) -> None:
    empty = tmp_path / "empty.txt"
    empty.write_text("")
    result = runner.invoke(app, ["score", str(empty)])
    assert result.exit_code == 1  # could not score: value None, never 0
    assert "no words" in result.output

    result = runner.invoke(app, ["score", "--json", str(empty)])
    assert result.exit_code == 1
    assert json.loads(result.stdout)[0] == {
        "file": str(empty),
        "value": None,
        "reason": "",
        "error": "no words to score",
    }  # stdout stays pure JSON; None, never 0

    latin1 = tmp_path / "latin1.txt"
    latin1.write_bytes("caf\xe9".encode("latin-1"))
    result = runner.invoke(app, ["score", str(latin1)])
    assert result.exit_code == 1
    assert "not UTF-8" in result.output

    result = runner.invoke(app, ["score", "--scorer", "vibes", str(files / "a.txt")])
    assert result.exit_code == 2
    assert result.stdout == ""
    assert "unknown scorer 'vibes'" in result.stderr

    assert runner.invoke(app, ["score", str(tmp_path / "missing.txt")]).exit_code == 2

    result = runner.invoke(app, ["score", "--scorer", "songeval", str(files / "a.txt")])
    assert result.exit_code == 2
    assert "accept_license=True" in result.output


def test_profile_ask_refuses_without_a_terminal(files: Path) -> None:
    result = runner.invoke(app, ["profile", "ask", "--name", "cli-test", "--items", str(files)])
    assert result.exit_code == 2
    assert "not a TTY" in result.output


def test_profile_ask_records_picks(files: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tt.ConsoleIO, "ask", lambda self, prompt: input(prompt))  # as if in a terminal
    result = runner.invoke(
        app, ["profile", "ask", "--name", "cli-ask", "--items", str(files), "--budget", "2"], input="1\n2\n"
    )
    assert result.exit_code == 0, result.output
    assert "recorded 2 pick(s)" in result.output
    picks = tt.profile("cli-ask").picks
    assert len(picks) == 2
    first_pair = result.output.split("Pick 1/2")[1].split("Pick 2/2")[0]
    option_1 = first_pair.split("[1] ")[1].split("\n")[0]
    assert picks[0]["winner"] == option_1  # "1" picked the first option shown


def test_profile_ask_needs_two_items(tmp_path: Path) -> None:
    (tmp_path / "only.txt").write_text(FRESH)
    result = runner.invoke(app, ["profile", "ask", "--name", "x", "--items", str(tmp_path)])
    assert result.exit_code == 2
    assert "at least two" in result.output


def test_agreement() -> None:
    result = runner.invoke(app, ["agreement", "--name", "cli-nobody"])
    assert result.exit_code == 2
    assert "has no picks" in result.output

    me = tt.profile("cli-agree")
    me.add_pick(winner=FRESH, loser=SLOP)
    result = runner.invoke(app, ["agreement", "--name", "cli-agree", "--json"])
    assert result.exit_code == 0, result.output
    report = json.loads(result.output)
    assert report["scorers"]["slop"]["rate"] == 1.0

    result = runner.invoke(app, ["agreement", "--name", "cli-agree"])  # the plain-text table
    assert result.exit_code == 0, result.output
    assert "slop" in result.output
    assert runner.invoke(app, ["agreement", "--name", "cli-agree", "--scorer", "vibes"]).exit_code == 2


def test_models() -> None:
    result = runner.invoke(app, ["models"])
    assert result.exit_code == 0
    assert "pickscore" in result.output
    assert "needs --accept-license" in result.output
    rows = json.loads(runner.invoke(app, ["models", "--json"]).output)
    assert {row["name"] for row in rows} == set(tt.models())


def test_console_script_module_runs() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "hone_taste.cli", "--help"], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
    assert "score" in result.stdout
