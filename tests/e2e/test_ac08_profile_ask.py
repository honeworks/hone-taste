"""AC-8: profile.ask records picks from scripted IO; without a TTY and without io= it fails clearly."""

import io
from pathlib import Path

import pytest

import hone_taste as tt
from hone_taste.testing import ScriptedIO


def test_ac8_profile_ask(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    me = tt.profile("asker", path=tmp_path / "asker.json")
    pairs = [("fresh line", "neon echoes"), ("rain on tin", "tapestry of time"), ("x", "y"), ("p", "q")]
    scripted = ScriptedIO(["1", "2", "maybe", "s", "q"])

    recorded = me.ask(pairs, io=scripted)

    assert recorded == [("fresh line", "neon echoes"), ("tapestry of time", "rain on tin")]
    assert "Pick 1/4" in scripted.prompts[0]
    assert "[1] fresh line" in scripted.prompts[0]
    assert "[2] neon echoes" in scripted.prompts[0]
    assert len(scripted.prompts) == 5  # "maybe" is asked again, then skipped with "s"; "q" stops
    reloaded = tt.profile("asker", path=tmp_path / "asker.json")
    assert reloaded.pairs == recorded

    quitter = tt.profile("quitter", path=tmp_path / "quitter.json")
    assert quitter.ask(pairs, io=ScriptedIO(["q"])) == []
    assert not (tmp_path / "quitter.json").exists()
    nothing = ScriptedIO([])
    assert me.ask([], io=nothing) == []
    assert nothing.prompts == []

    monkeypatch.setattr("sys.stdin", io.StringIO(""))
    with pytest.raises(tt.errors.HoneTasteError, match="not a TTY"):
        me.ask(pairs)
    assert len(tt.profile("asker", path=tmp_path / "asker.json").picks) == 2
