"""Use hone-taste scorers inside hone-select without importing hone-select: `tt.for_select`.

What: hone-select calls each scorer in its registry with a candidate that has `id`, `data` (a dict),
      `files` (name -> path) and `meta`. `tt.for_select(scorer, ...)` adapts any hone-taste scorer to
      that shape: it reads the input from the candidate, links the score to the caller's trace and returns
      the `Score` (hone-select's ScoreLike: `value`, `confidence`, `reason`, `details`, `error`).
How:  1. `field="lyrics"` reads `candidate.data["lyrics"]`; no field passes all of `candidate.data`,
      2. `file="audio"` reads `candidate.files["audio"]` (audio / image scorers only),
      3. `prompt_field="prompt"` builds `{"prompt": data["prompt"], "response": <field>}` for
         `text_with_prompt` scorers (and `{"prompt", "image"}` with `file=`),
      4. `record=False` leaves recording to hone-select; `trace=` (hone-select passes it) links the span.
      Any object or mapping with those fields works as a candidate; this example builds a tiny one.
Why:  hone-taste stays useful alone and hone-select stays free of hone-taste: they meet only at this
      shape. Pitfall: a candidate without the field is "could not score"
      (`value=None` with an error), never 0, so hone-select can tell a bad candidate from a broken input.
      With hone-select: `Engine(cfg, registry=[write_lyric, tt.for_select(slop, field="lyrics")])`.

Run: uv run python examples/select_bridge.py
"""

from dataclasses import dataclass, field

import hone_taste as tt
from hone_taste.errors import ConfigError
from hone_taste.testing import FakeTasteModel


@dataclass
class Candidate:  # the shape hone-select passes; its own class is not needed
    id: str
    data: dict
    files: dict = field(default_factory=dict)
    meta: dict = field(default_factory=dict)


# 1. A text field.
lyric_scorer = tt.for_select(tt.slop_score(domain="lyrics"), field="lyrics")
candidates = [
    Candidate("a", {"lyrics": "Neon echoes whisper through the tapestry of time"}),
    Candidate("b", {"lyrics": "Rain on the tin roof, my father's boots by the door"}),
    Candidate("c", {"title": "no lyrics yet"}),
]
values = {}
for candidate in candidates:
    score = lyric_scorer(candidate)
    values[candidate.id] = score.value
    print(f"{candidate.id}: value={score.value} {score.error or score.reason}")
assert values["a"] < values["b"] and values["c"] is None
print(f"scorer name={lyric_scorer.name} version={lyric_scorer.version}")  # hone-select records both

# 2. A file: audio scorers read candidate.files[...] (a fake Audiobox backend here).
audio_scorer = tt.for_select(
    tt.audiobox(model=FakeTasteModel({"CE": 8.0, "CU": 7.0, "PC": 5.0, "PQ": 9.0})), file="audio"
)
song = Candidate("s1", {"lyrics": "..."}, files={"audio": "renders/s1.wav"})
audio = audio_scorer(song)
print(f"audio: value={audio.value:.2f}")
assert audio.value == (0.8 + 0.7 + 0.9) / 3  # mean of CE, CU, PQ on a 0-10 scale

try:
    tt.for_select(tt.slop_score(), file="audio")  # a text scorer cannot read a file path
except ConfigError as exc:
    print("config error:", exc)
else:
    raise AssertionError("expected a ConfigError")

# 3. Prompt + response for a reward model; a plain dict works as a candidate too.
judge = tt.for_select(
    tt.reward_model(model=FakeTasteModel(lambda item: {"reward": float(len(item["response"]))})),
    field="answer",
    prompt_field="prompt",
)
answer = {"id": "q1", "data": {"prompt": "Name a blues scale note.", "answer": "The flattened fifth."}}
reward = judge(answer)
print(f"reward: value={reward.value:.2f} raw={reward.details['raw']}")
assert reward.details["raw"] == len("The flattened fifth.")

# 4. Records: the span carries the candidate id and the scorer name; record=False records nothing.
with tt.recording(tt.MemorySink()) as sink:
    lyric_scorer(candidates[1], trace={"hone.run_id": "run-7"})
    tt.for_select(tt.slop_score(), field="lyrics", record=False)(candidates[1])
attributes = sink.spans[0]["attributes"]
print("span:", {k: attributes[k] for k in ("hone.run_id", "hone.candidate_id", "hone.scorer")})
assert len(sink.spans) == 1 and attributes["hone.candidate_id"] == "b"
