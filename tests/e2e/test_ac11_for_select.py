"""AC-11: for_select reads candidate.data[field] / candidate.files[file] and returns a ScoreLike,
with a minimal fake candidate (no hone-select import)."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

import hone_taste as tt


@dataclass
class Candidate:
    id: str
    data: Any
    files: dict[str, str] = field(default_factory=dict[str, str])
    meta: dict[str, Any] = field(default_factory=dict[str, Any])


@dataclass
class Echo:
    """Scores the length of whatever it is given and remembers it."""

    name: str = "echo"
    family: str = "human_likeness"
    accepts: frozenset[str] = frozenset({"text", "audio", "text_with_prompt"})
    license: str = "Apache-2.0"
    source_data: str = "none"
    seen: list[Any] = field(default_factory=list[Any])

    def __call__(self, input: Any, /) -> tt.Score:
        self.seen.append(input)
        return tt.Score(0.5, reason="echo")


def test_ac11_for_select(tmp_path: Path) -> None:
    slop = tt.for_select(tt.slop_score(domain="lyrics"), field="lyrics")
    fresh = Candidate("c1", {"lyrics": "Rain on the tin roof, my father's boots by the door."})
    cliched = Candidate("c2", {"lyrics": "Neon echoes whisper through the tapestry of time."})
    good, bad = slop(fresh), slop(cliched)
    for score in (good, bad):  # ScoreLike: attribute object with the candidate-scorer fields
        assert all(hasattr(score, a) for a in ("value", "confidence", "reason", "details", "error"))
    assert good.value is not None
    assert bad.value is not None
    assert good.value > bad.value
    assert slop.name == "slop_lyrics"

    echo = Echo()
    audio = tmp_path / "song.wav"
    tt.for_select(echo, file="audio")(Candidate("c3", {}, files={"audio": str(audio)}))
    tt.for_select(echo)(Candidate("c4", "the whole data"))
    tt.for_select(echo, field="answer", prompt_field="question")(
        Candidate("c5", {"question": "Why?", "answer": "Because."})
    )
    mapping_candidate = {"data": {"lyrics": "la"}, "files": {}, "meta": {}}  # a Mapping, without an id
    tt.for_select(echo, field="lyrics")(mapping_candidate)
    image = str(tmp_path / "frame.png")
    tt.for_select(echo, file="frame", prompt_field="prompt")(
        Candidate("c6", {"prompt": "a red barn"}, files={"frame": image})
    )
    assert echo.seen == [
        str(audio),
        "the whole data",
        {"prompt": "Why?", "response": "Because."},
        "la",
        {"prompt": "a red barn", "image": image},
    ]

    missing = slop(Candidate("c7", {"title": "no lyrics here"}))
    assert missing.value is None
    assert "lyrics" in missing.error

    with pytest.raises(tt.errors.ConfigError, match="not both"):
        tt.for_select(echo, field="lyrics", file="audio")
    with pytest.raises(tt.errors.ConfigError, match="does not take files"):
        tt.for_select(tt.slop_score(), file="audio")  # would score the path string as text
    assert slop.version == tt.__version__

    with tt.recording(tt.MemorySink()) as sink:
        slop(fresh)
        tt.for_select(tt.slop_score(), field="lyrics", record=False)(fresh)
    assert len(sink.spans) == 1  # record=False wrote nothing
    assert sink.spans[0]["attributes"]["hone.candidate_id"] == "c1"
    assert sink.spans[0]["attributes"]["hone.scorer"] == "slop_lyrics"

    trace_id, parent = "4bf92f3577b34da6a3ce929d0e0e4736", "00f067aa0ba902b7"
    with tt.recording(tt.MemorySink()) as sink:
        slop(fresh, trace={"traceparent": f"00-{trace_id}-{parent}-01", "hone.run_id": "r1"})
    span = sink.spans[0]
    assert (span["trace_id"], span["parent_span_id"]) == (trace_id, parent)  # hone-select's trace
    assert span["attributes"]["hone.run_id"] == "r1"
