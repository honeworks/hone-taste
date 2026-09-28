"""Personal taste profiles: a few pairwise picks re-weight the other scorers for one person.

Stored as JSON at `${HONE_HOME:-.hone}/taste/profiles/<name>.json` (picks, examples, fitted weights, dates).
"""

from __future__ import annotations

import dataclasses
import itertools
import json
import os
import re
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path, PurePath
from typing import Any, Protocol

from hone_taste.combine import aggregate_function, combine
from hone_taste.errors import ConfigError, HoneTasteError
from hone_taste.fitting import Weights, fit_weights, item_key, score_table
from hone_taste.types import FunctionScorer, Scorer

NAME = re.compile(r"[A-Za-z0-9_.-]+")
PICK_PROMPT = "Which do you prefer? [1/2, s = skip, q = quit]: "


class PickIO(Protocol):
    """How `Profile.ask` talks to a person: show a prompt, return their reply."""

    def ask(self, prompt: str) -> str: ...


class ConsoleIO:
    """Asks in the terminal; refuses to run when stdin is not a TTY (never hang a pipeline)."""

    def ask(self, prompt: str) -> str:
        if not sys.stdin.isatty():
            raise HoneTasteError(
                "profile.ask needs a terminal but stdin is not a TTY; pass io= (e.g. "
                "hone_taste.testing.ScriptedIO) or record picks with profile.add_pick()"
            )
        return input(prompt)


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _jsonable(item: Any) -> Any:
    item = str(item) if isinstance(item, PurePath) else item
    try:
        return json.loads(json.dumps(item))  # in memory exactly as on disk (tuples become lists)
    except (TypeError, ValueError) as exc:
        raise ConfigError(
            f"profile items must be text, paths or JSON-compatible data; got {type(item).__name__}"
        ) from exc


def default_path(name: str) -> Path:
    return Path(os.environ.get("HONE_HOME", ".hone")) / "taste" / "profiles" / f"{name}.json"


@dataclass
class Profile:
    """One person's picks, liked / disliked examples and fitted weights. Every change is saved."""

    name: str
    path: Path
    picks: list[dict[str, Any]] = field(default_factory=list[dict[str, Any]])
    liked: list[Any] = field(default_factory=list[Any])
    disliked: list[Any] = field(default_factory=list[Any])
    fitted: Weights | None = None
    created_at: str = field(default_factory=_now)
    updated_at: str = ""

    @property
    def pairs(self) -> list[tuple[Any, Any]]:
        """The picks as (winner, loser) pairs."""
        return [(p["winner"], p["loser"]) for p in self.picks]

    def add_pick(self, winner: Any, loser: Any, *, context: Any = None) -> None:
        """Record that this person preferred `winner` over `loser`."""
        pick = {
            "winner": _jsonable(winner),
            "loser": _jsonable(loser),
            "context": _jsonable(context),
            "at": _now(),
        }
        self.picks.append(pick)
        self.save()

    def examples(self, liked: Sequence[Any] = (), disliked: Sequence[Any] = ()) -> None:
        """Add examples this person liked / disliked (used by `prompt_examples()`)."""
        self.liked.extend(_jsonable(x) for x in liked)
        self.disliked.extend(_jsonable(x) for x in disliked)
        self.save()

    def fit(self, scorers: Mapping[str, Scorer], items: Sequence[Any] = ()) -> Weights:
        """Fit weights over `scorers` from the picks (`items`: extra examples to measure score spread)."""
        pairs = self.pairs
        if not pairs:  # check before scoring: scorers may be slow (LLM panels)
            raise ConfigError(f"profile {self.name!r} has no picks; add picks with add_pick() or ask()")
        table = score_table(scorers, [*items, *(item for pair in pairs for item in pair)])
        self.fitted = fit_weights(table, list(scorers), pairs)
        self.save()
        return self.fitted

    def as_scorer(self, scorers: Mapping[str, Scorer]) -> FunctionScorer:
        """The weighted combination of `scorers` tuned to this person (call `fit` first)."""
        if self.fitted is None:
            raise ConfigError(f"profile {self.name!r} has no fitted weights; call fit(scorers, items) first")
        combined = combine(self.fitted.weights, scorers)
        return dataclasses.replace(combined, name=f"profile:{self.name}", family="personal")

    def questions(
        self, candidates: Sequence[Any], scorers: Mapping[str, Scorer], budget: int = 5
    ) -> list[tuple[Any, Any]]:
        """The most informative pairs to ask about: closest predicted margin first, then the largest
        disagreement between scorers. Pairs already picked are never repeated."""
        if budget < 0:
            raise ConfigError(f"budget must be >= 0; got {budget}")
        if self.fitted is not None and set(self.fitted.weights) == set(scorers):
            weights = self.fitted.weights
        else:
            weights = dict.fromkeys(scorers, 1.0)
        table = score_table(scorers, candidates)
        mean = aggregate_function("weighted_mean")  # ignores None values; None when nothing scored
        utility = {key: mean([(row[n], weights[n]) for n in scorers]) for key, row in table.items()}
        asked = {frozenset((item_key(w), item_key(lo))) for w, lo in self.pairs}
        unique = list({item_key(c): c for c in candidates}.items())
        ranked: list[tuple[tuple[float, float], Any, Any]] = []
        for (ka, a), (kb, b) in itertools.combinations(unique, 2):
            ua, ub = utility[ka], utility[kb]
            if ua is not None and ub is not None and frozenset((ka, kb)) not in asked:
                ranked.append(((abs(ua - ub), -_disagreement(table[ka], table[kb])), a, b))
        ranked.sort(key=lambda r: r[0])
        return [(a, b) for _, a, b in ranked[:budget]]

    def ask(self, pairs: Sequence[tuple[Any, Any]], *, io: PickIO | None = None) -> list[tuple[Any, Any]]:
        """Ask the person about each pair (1 / 2 / skip / quit) and record the picks. Returns them."""
        io = io or ConsoleIO()
        recorded: list[tuple[Any, Any]] = []
        for number, (a, b) in enumerate(pairs, 1):
            reply = _read_choice(io, f"\nPick {number}/{len(pairs)}\n[1] {a}\n[2] {b}\n{PICK_PROMPT}")
            if reply == "q":
                break
            if reply in ("1", "2"):
                winner, loser = (a, b) if reply == "1" else (b, a)
                self.add_pick(winner, loser)
                recorded.append((winner, loser))
        return recorded

    def prompt_examples(self, limit: int = 5, max_chars: int = 200) -> str:
        """A short block of liked / disliked examples to paste into a panel or judge prompt."""

        def block(title: str, items: list[Any]) -> list[str]:
            return [title, *(f"- {str(x)[:max_chars]}" for x in items[-limit:])] if items else []

        lines = block("Examples this person liked:", self.liked) + block(
            "Examples this person disliked:", self.disliked
        )
        return "\n".join(lines)

    def save(self) -> None:
        """Write the profile atomically (temp file + rename)."""
        self.updated_at = _now()
        data = {k: v for k, v in dataclasses.asdict(self).items() if k != "path"}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(data, indent=1, ensure_ascii=False), encoding="utf-8")
        tmp.replace(self.path)


def _disagreement(row_a: dict[str, float | None], row_b: dict[str, float | None]) -> float:
    """How much the scorers disagree about a pair: the spread of their score differences."""
    diffs = [a - b for n, a in row_a.items() if a is not None and (b := row_b[n]) is not None]
    return max(diffs) - min(diffs) if diffs else 0.0


def _read_choice(io: PickIO, prompt: str) -> str:
    for _ in range(3):
        reply = io.ask(prompt).strip().lower()
        if reply in ("1", "2", "s", "q"):
            return reply
    return "s"  # three unclear replies: skip this pair


def profile(name: str, *, path: str | Path | None = None) -> Profile:
    """Load the profile `name` (or start an empty one; it is saved on the first change).

    >>> import tempfile, pathlib
    >>> me = profile("ana", path=pathlib.Path(tempfile.mkdtemp()) / "ana.json")
    >>> me.add_pick(winner="a lyric with a real image", loser="a lyric about neon echoes")
    >>> len(profile("ana", path=me.path).picks)
    1
    """
    if not NAME.fullmatch(name):
        raise ConfigError(f"profile name {name!r} may only use letters, digits, '_', '-' and '.'")
    file = Path(path) if path is not None else default_path(name)
    if not file.exists():
        return Profile(name, file)
    try:
        data = json.loads(file.read_text(encoding="utf-8"))
        fitted = Weights(**data["fitted"]) if data.get("fitted") else None
        if any("winner" not in p or "loser" not in p for p in data.get("picks", [])):
            raise ValueError("every pick needs a winner and a loser")
    except (ValueError, TypeError) as exc:
        raise ConfigError(f"profile file {file} is not valid: {exc}") from exc
    return Profile(
        name=name,
        path=file,
        picks=data.get("picks", []),
        liked=data.get("liked", []),
        disliked=data.get("disliked", []),
        fitted=fitted,
        created_at=data.get("created_at", _now()),
        updated_at=data.get("updated_at", ""),
    )
