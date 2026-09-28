import io
import json
from pathlib import Path
from typing import Any

import pytest

from hone_taste.errors import ConfigError, HoneTasteError
from hone_taste.profile import ConsoleIO, profile
from hone_taste.testing import ScriptedIO
from hone_taste.types import FunctionScorer, Score


def lookup(name: str, values: dict[str, float | None]) -> FunctionScorer:
    return FunctionScorer(name, "human_likeness", frozenset({"text"}), lambda t: Score(values[t]))


def test_name_must_be_file_safe(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="profile name"):
        profile("../evil", path=tmp_path / "x.json")


@pytest.mark.parametrize(
    "content", ["{not json", '{"fitted": {"weights": {}, "bogus": 1}}', '{"fitted": {"weights": {}}}']
)
def test_broken_file_raises_config_error(tmp_path: Path, content: str) -> None:
    path = tmp_path / "p.json"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ConfigError, match="not valid"):
        profile("p", path=path)


def test_non_json_items_are_rejected_and_paths_become_strings(tmp_path: Path) -> None:
    me = profile("p", path=tmp_path / "p.json")
    with pytest.raises(ConfigError, match="JSON-compatible"):
        me.add_pick(winner=object(), loser="b")
    me.add_pick(winner=tmp_path / "a.wav", loser={"prompt": "p", "response": "r"}, context={"round": 1})
    data = json.loads((tmp_path / "p.json").read_text(encoding="utf-8"))
    assert data["picks"][0]["winner"] == str(tmp_path / "a.wav")
    assert data["picks"][0]["context"] == {"round": 1}
    assert data["created_at"]
    assert data["updated_at"]
    assert not list(tmp_path.glob("*.tmp"))  # atomic write leaves no temp file


def test_new_profile_is_not_written_until_changed(tmp_path: Path) -> None:
    me = profile("p", path=tmp_path / "p.json")
    assert me.picks == []
    assert me.fitted is None
    assert not (tmp_path / "p.json").exists()


def test_fit_and_as_scorer_errors(tmp_path: Path) -> None:
    me = profile("p", path=tmp_path / "p.json")
    scorers = {"s": lookup("s", {"a": 0.9, "b": 0.1})}
    with pytest.raises(ConfigError, match="no picks"):
        me.fit(scorers)
    with pytest.raises(ConfigError, match="call fit"):
        me.as_scorer(scorers)


def test_as_scorer_is_a_personal_weighted_combination(tmp_path: Path) -> None:
    me = profile("p", path=tmp_path / "p.json")
    scorers = {"s": lookup("s", {"a": 0.9, "b": 0.1}), "t": lookup("t", {"a": 0.1, "b": 0.9})}
    me.add_pick(winner="a", loser="b")
    me.fit(scorers)
    mine = me.as_scorer(scorers)
    assert mine.name == "profile:p"
    assert mine.family == "personal"
    assert mine("a").value == pytest.approx(0.9)


S = {"a": 0.875, "b": 0.125, "c": 0.625, "e": 0.375, "d": None}  # binary-exact: no float ties broken
T = {"a": 0.125, "b": 0.875, "c": 0.125, "e": 0.875, "d": None}


def test_questions_use_fitted_weights(tmp_path: Path) -> None:
    me = profile("p", path=tmp_path / "p.json")
    scorers = {"s": lookup("s", S), "t": lookup("t", T)}
    me.add_pick(winner="a", loser="b")
    # equal weights: four pairs at margin 0.125; (a, e) has the largest scorer disagreement
    assert me.questions(list(S), scorers, budget=1) == [("a", "e")]
    assert me.fit(scorers).weights == {"s": 1.0, "t": 0.0}
    # only s counts now: (c, e) is among the closest and the scorers disagree most on it
    assert me.questions(list(S), scorers, budget=1) == [("c", "e")]
    assert all("d" not in pair for pair in me.questions(list(S), scorers, budget=20))
    with pytest.raises(ConfigError, match="budget"):
        me.questions(["a"], scorers, budget=-1)


def test_questions_ignore_weights_fitted_on_other_scorers(tmp_path: Path) -> None:
    me = profile("p", path=tmp_path / "p.json")
    me.add_pick(winner="a", loser="b")
    me.fit({"s": lookup("s", S)})
    assert me.questions(list(S), {"s": lookup("s", S), "t": lookup("t", T)}, budget=1) == [("a", "e")]


def test_ask_keeps_picks_made_before_an_interrupt(tmp_path: Path) -> None:
    class Interrupted:
        def __init__(self) -> None:
            self.replies = iter(["1"])

        def ask(self, prompt: str) -> str:
            reply = next(self.replies, None)
            if reply is None:
                raise KeyboardInterrupt
            return reply

    me = profile("p", path=tmp_path / "p.json")
    with pytest.raises(KeyboardInterrupt):
        me.ask([("x", "y"), ("u", "v")], io=Interrupted())
    assert profile("p", path=tmp_path / "p.json").pairs == [("x", "y")]


def test_failed_save_leaves_the_previous_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    me = profile("p", path=tmp_path / "p.json")
    me.add_pick(winner="a", loser="b")

    def broken_replace(self: Path, target: Path) -> Path:
        raise OSError("disk full")

    monkeypatch.setattr(Path, "replace", broken_replace)
    with pytest.raises(OSError, match="disk full"):
        me.add_pick(winner="c", loser="d")
    monkeypatch.undo()
    assert profile("p", path=tmp_path / "p.json").pairs == [("a", "b")]


def test_ask_skips_after_three_unclear_replies(tmp_path: Path) -> None:
    me = profile("p", path=tmp_path / "p.json")
    scripted = ScriptedIO(["?", "maybe", "3", "2"])
    assert me.ask([("x", "y"), ("u", "v")], io=scripted) == [("v", "u")]
    assert len(scripted.prompts) == 4


def test_console_io_reads_input_on_a_tty(monkeypatch: pytest.MonkeyPatch) -> None:
    class Tty(io.StringIO):
        def isatty(self) -> bool:
            return True

    monkeypatch.setattr("sys.stdin", Tty("1\n"))
    assert ConsoleIO().ask("pick: ") == "1"
    monkeypatch.setattr("sys.stdin", io.StringIO("1\n"))
    with pytest.raises(HoneTasteError, match="ScriptedIO"):
        ConsoleIO().ask("pick: ")


def test_prompt_examples(tmp_path: Path) -> None:
    me = profile("p", path=tmp_path / "p.json")
    assert me.prompt_examples() == ""
    me.examples(liked=["rain on tin", "x" * 300], disliked=["neon echoes"])
    block = me.prompt_examples(limit=1, max_chars=10)
    assert block == "Examples this person liked:\n- xxxxxxxxxx\nExamples this person disliked:\n- neon echoe"
    assert profile("p", path=tmp_path / "p.json").liked == ["rain on tin", "x" * 300]


def test_fitted_weights_round_trip(tmp_path: Path) -> None:
    me = profile("p", path=tmp_path / "p.json")
    me.add_pick(winner="a", loser="b")
    fitted = me.fit({"s": lookup("s", {"a": 0.9, "b": 0.1})})
    reloaded: Any = profile("p", path=tmp_path / "p.json")
    assert reloaded.fitted == fitted


def test_questions_with_no_shared_scorer_do_not_crash(tmp_path: Path) -> None:
    me = profile("p", path=tmp_path / "p.json")
    scorers = {"x": lookup("x", {"A": 0.3, "B": None}), "y": lookup("y", {"A": None, "B": 0.6})}
    assert me.questions(["A", "B"], scorers) == [("A", "B")]


def test_items_are_stored_as_json(tmp_path: Path) -> None:
    me = profile("p", path=tmp_path / "p.json")
    me.add_pick(winner=("t", 1), loser=("u", 2))
    assert me.pairs == profile("p", path=tmp_path / "p.json").pairs == [(["t", 1], ["u", 2])]


def test_pick_without_loser_is_invalid(tmp_path: Path) -> None:
    path = tmp_path / "p.json"
    path.write_text('{"picks": [{"winner": "a"}]}', encoding="utf-8")
    with pytest.raises(ConfigError, match="winner and a loser"):
        profile("p", path=path)
