"""Putting it together: pick the best lyric draft the way a song pipeline does, with fakes.

What: a lyric selection that combines two families and one person: the slop score (lyrics preset) for
      clichés, a two-persona listener panel for appeal, and a profile that learns from a few of the
      person's picks how much each should count. The panel's client is a scripted
      `FakeDecisionClient`, so the example runs offline; everything is recorded to one SQLite file.
How:  1. build the scorers: `tt.slop_score(domain="lyrics")` and `tt.audience(personas, question, client)`,
      2. `me.questions(drafts, scorers, budget=2)` finds the closest calls; `me.ask(...)` records the
         person's answers (scripted here; in a terminal just `me.ask(pairs)`); add known picks directly,
      3. `me.fit(scorers, items=drafts)` learns the weights; `me.as_scorer(scorers)` ranks the drafts,
      4. `tt.recording(tt.SqliteSpanSink(path))` keeps every score for later inspection (hone-lens).
Why:  each family alone is easy to fool (slop misses dull lines, a panel is one LLM pretending); combined
      and tuned to the person who decides, they pick drafts that person would pick. For real runs swap the
      fake client for a real one, e.g.
      `OpenAITextClient("gemma3:12b", base_url="http://127.0.0.1:11434/v1", api_key="ollama")`
      from `hone_taste.adapters.openai`, and drop `path=` so the profile persists in `${HONE_HOME}`.

Run: uv run python examples/lyrics_taste.py
"""

import tempfile
from pathlib import Path

import hone_taste as tt
from hone_taste.testing import FakeDecisionClient, ScriptedIO

DRAFTS = [
    "Neon echoes whisper through the tapestry of night, not just a song but a journey",
    "Dad's work boots by the door still smell like diesel and rain",
    "Baby baby baby, you're my baby, you're my baby tonight",
    "The kettle clicks off and nobody gets up to pour it",
]


def listener(state, name, question):
    """A scripted persona: likes concrete images, dislikes stock phrases (1-5 scale)."""
    concrete = any(word in state for word in ("boots", "kettle", "diesel"))
    raw = 5 if concrete else 2
    rationale = "concrete images" if concrete else "stock phrases"
    return {"type": "score", "value": (raw - 1) / 4, "raw": raw, "rationale": rationale}


home = tempfile.TemporaryDirectory()  # kept for the whole script, removed at the end
sink = tt.SqliteSpanSink(Path(home.name, "spans.db"))
with tt.recording(sink):
    # 1. Two families of scorers.
    slop = tt.slop_score(domain="lyrics")
    panel = tt.audience(
        ["A 25-year-old who skips songs within 20 seconds", "A 55-year-old blues fan who hates cliches"],
        question="Would you keep listening past the first chorus?",
        client=FakeDecisionClient(listener),
    )
    scorers = {"slop": slop, "panel": panel}

    # 2. A few picks from the person who decides: the two closest calls, plus one known pick.
    me = tt.profile("songwriter", path=Path(home.name, "songwriter.json"))
    pairs = me.questions(DRAFTS, scorers, budget=2)
    me.ask(pairs, io=ScriptedIO(["1", "2"]))
    me.add_pick(winner=DRAFTS[1], loser=DRAFTS[0])
    assert len(me.picks) == 3

    # 3. Weights tuned to the person, then a ranking.
    weights = me.fit(scorers, items=DRAFTS)
    mine = me.as_scorer(scorers)
    scores = {draft: mine(draft) for draft in DRAFTS}  # score once: a real panel costs LLM calls
    # Rank only what could be scored; a None value is "no opinion", not a low score.
    values = {draft: s.value for draft, s in scores.items() if s.value is not None}
    ranked = sorted(values, key=lambda draft: values[draft], reverse=True)
    for draft in ranked:
        print(f"{values[draft]:.2f}  {draft}")

print("weights:", {name: round(w, 2) for name, w in weights.weights.items()})
# 4. Every score above, including the panel's and slop's parts, is in the SQLite file.
spans = sink.spans()
print(
    "spans recorded:", len(spans), "scorers:", sorted({s["attributes"]["hone.taste.scorer"] for s in spans})
)
assert ranked[0] in (DRAFTS[1], DRAFTS[3]) and ranked[-1] in (DRAFTS[0], DRAFTS[2])
assert weights.picks == 3
assert {"slop_lyrics", "audience", "profile:songwriter"} <= {
    s["attributes"]["hone.taste.scorer"] for s in spans
}
sink.close()
home.cleanup()
