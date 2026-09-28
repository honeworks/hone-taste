"""Deterministic, scriptable fakes. Each records its calls in `.calls`."""

from __future__ import annotations

import json
from collections.abc import Callable, Generator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

from hone_taste.errors import ConfigError
from hone_taste.ports import Answer, Question, TextResult, TraceContext

AnswerRule = Callable[[str | Mapping[str, Any], str, Question], Answer]


def default_answer(state: str | Mapping[str, Any], name: str, question: Question) -> dict[str, Any]:
    """A neutral, valid answer: "yes" at 0.5, the first option, or the middle of the scale."""
    kind = question["type"]
    answer: dict[str, Any] = {"type": kind, "calibrated": False, "rationale": f"fake answer to {name}"}
    if kind == "yes_no":
        return {**answer, "value": 0.5, "probabilities": {"yes": 0.5, "no": 0.5}}
    if kind == "choice":
        return {**answer, "value": 1.0, "choice": question["options"][0]}
    low, high = question.get("scale", (1, 5))
    return {**answer, "value": 0.5, "raw": (low + high) / 2}


@dataclass
class FakeDecisionClient:
    """A `DecisionClient` answering from a rule `(state, question_name, question) -> Answer`.

    `answers` may also be a mapping from question name to a fixed answer. A rule that raises makes the
    whole `decide` call raise (like a transport error).

    >>> fake = FakeDecisionClient(answers={"q": {"type": "score", "value": 1.0, "raw": 5}})
    >>> fake.decide("text", {"q": {"type": "score", "instructions": "Good?"}})["q"]["value"]
    1.0
    >>> len(fake.calls)
    1
    """

    answers: Mapping[str, Answer] | AnswerRule = default_answer
    calls: list[dict[str, Any]] = field(default_factory=list[dict[str, Any]])

    def decide(
        self,
        state: str | Mapping[str, Any],
        questions: Mapping[str, Question],
        *,
        images: Sequence[str] = (),
        trace: TraceContext | None = None,
    ) -> dict[str, Answer]:
        self.calls.append(
            {"state": state, "questions": dict(questions), "images": list(images), "trace": trace}
        )
        answers = self.answers
        if callable(answers):
            return {name: answers(state, name, q) for name, q in questions.items()}
        return {name: answers[name] for name in questions if name in answers}


@dataclass
class FakeTextClient:
    """A `TextClient` returning scripted responses in order (the last one repeats).

    Each response is a string or a function `(messages, schema) -> str`. With a `schema`, the text is
    parsed as JSON into `parsed`, or `error` is set.

    >>> FakeTextClient(['{"ok": true}']).complete([{"role": "user", "content": "hi"}], schema={}).parsed
    {'ok': True}
    """

    responses: Sequence[str | Callable[[Sequence[Mapping[str, Any]], Mapping[str, Any] | None], str]] = (
        "OK",
    )
    model: str = "fake-text"
    calls: list[dict[str, Any]] = field(default_factory=list[dict[str, Any]])

    def __post_init__(self) -> None:
        if not self.responses:
            raise ConfigError("FakeTextClient needs at least one response")

    def complete(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        schema: Mapping[str, Any] | None = None,
        trace: TraceContext | None = None,
        **params: Any,
    ) -> TextResult:
        self.calls.append({"messages": list(messages), "schema": schema, "trace": trace, "params": params})
        response = self.responses[min(len(self.calls), len(self.responses)) - 1]
        text = response(messages, schema) if callable(response) else response
        parsed, error = None, None
        if schema is not None:
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                error = "response is not valid JSON"
        return TextResult(text=text, parsed=parsed, error=error, model=self.model, finish_reason="stop")


@dataclass
class ScriptedIO:
    """Answers `Profile.ask` prompts from a list of replies ("1", "2", "s", "q"); "q" once they run out.

    >>> io = ScriptedIO(["2"])
    >>> io.ask("Which do you prefer? "), io.ask("Which do you prefer? ")
    ('2', 'q')
    >>> len(io.prompts)
    2
    """

    replies: Sequence[str]
    prompts: list[str] = field(default_factory=list[str])

    def ask(self, prompt: str) -> str:
        self.prompts.append(prompt)
        index = len(self.prompts) - 1
        return self.replies[index] if index < len(self.replies) else "q"


@dataclass
class FakeTasteModel:
    """A `TasteModel` backend for any taste-model scorer (`model=`): no torch, no weights.

    `outputs` is a fixed `{name: raw value}` or a function of the input. `events` logs "load", "predict"
    and "unload"; share the list with a `FakeGpuLease` to check that the lease wraps them.

    >>> import hone_taste as tt
    >>> fake = FakeTasteModel({"CE": 8.0, "CU": 6.0, "PC": 2.0, "PQ": 7.0})
    >>> scorer = tt.audiobox(model=fake)
    >>> round(scorer("song.wav").value, 2), fake.events
    (0.7, ['load', 'predict'])
    >>> scorer.close(); fake.events[-1]
    'unload'
    """

    outputs: Mapping[str, float] | Callable[[Any], Mapping[str, float]] = field(
        default_factory=lambda: {"score": 0.5}
    )
    events: list[str] = field(default_factory=list[str])
    calls: list[Any] = field(default_factory=list[Any])
    options: list[dict[str, Any]] = field(default_factory=list[dict[str, Any]])
    loaded: bool = False

    def load(self) -> None:
        self.events.append("load")
        self.loaded = True

    def predict(self, input: Any, **options: Any) -> Mapping[str, float]:
        """`options` (e.g. SongEval's `max_seconds`) are logged in `self.options`, one dict per call."""
        if not self.loaded:
            raise RuntimeError("predict() called before load()")
        self.events.append("predict")
        self.calls.append(input)
        self.options.append(options)
        return self.outputs(input) if callable(self.outputs) else self.outputs

    def unload(self) -> None:
        self.events.append("unload")
        self.loaded = False


@dataclass
class FakeGpuLease:
    """A `GpuLease` that logs `"lease <name>"` / `"release <name>"` in `events` (reentrant, never waits)."""

    events: list[str] = field(default_factory=list[str])
    calls: list[dict[str, Any]] = field(default_factory=list[dict[str, Any]])

    @contextmanager
    def lease(
        self, name: str, vram_gb: float, *, timeout_s: float | None = None, trace: TraceContext | None = None
    ) -> Generator[None]:
        self.calls.append({"name": name, "vram_gb": vram_gb, "timeout_s": timeout_s, "trace": trace})
        self.events.append(f"lease {name}")
        try:
            yield
        finally:
            self.events.append(f"release {name}")


@dataclass
class FakeEmbedder:
    """An `ImageEmbedder` for `character_consistency(model=...)`: no torch, no weights, no files read.

    `vectors` maps an image path to its vector (`"sheet.png#1"` for the second strip of a split image), or
    is a function `(path, (index, count)) -> vector`. An unknown path raises `KeyError`, which the scorer
    turns into an error score, like a missing file. `events` logs "load" and "unload".

    >>> fake = FakeEmbedder({"a.png": [1.0, 0.0], "a.png#1": [0.0, 1.0]})
    >>> fake.load(); fake.embed("a.png"), fake.embed("a.png", (1, 2))
    ([1.0, 0.0], [0.0, 1.0])
    """

    vectors: Mapping[str, Sequence[float]] | Callable[[str, tuple[int, int]], Sequence[float]]
    events: list[str] = field(default_factory=list[str])
    calls: list[tuple[str, tuple[int, int]]] = field(default_factory=list[tuple[str, tuple[int, int]]])
    loaded: bool = False

    def load(self) -> None:
        self.events.append("load")
        self.loaded = True

    def embed(self, image: str, part: tuple[int, int] = (0, 1)) -> Sequence[float]:
        if not self.loaded:
            raise RuntimeError("embed() called before load()")
        self.calls.append((image, part))
        if callable(self.vectors):
            return self.vectors(image, part)
        return self.vectors[image if part == (0, 1) else f"{image}#{part[0]}"]

    def unload(self) -> None:
        self.events.append("unload")
        self.loaded = False
