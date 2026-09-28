"""Human-likeness without a model: the slop score, its domain presets, and your own banned patterns.

What: `tt.slop_score(domain=...)` measures how much a text leans on phrasing LLMs overuse (overused words
      and phrases 60%, "not X, but Y" contrasts 25%, overused trigrams 15%, per 1,000 words).
      `tt.patterns([...])` scores a text against your own list of banned literals and regexes.
How:  1. build one slop scorer per domain: "general", "lyrics" (adds "neon", "echoes of" ...), "email"
         (adds "I hope this email finds you well", "circle back" ...),
      2. score the same kinds of text with each and compare the values and hits,
      3. point `wordlists=` at a folder of upstream-format lists to use your own vocabulary,
      4. build `tt.patterns(...)` from literals (whole phrase, any case) and compiled regexes (as given);
         value = 1 - min(1, hits / max_hits), every hit listed with its span.
Why:  both are cheap (no GPU, no network), deterministic and explainable, so they run on every candidate
      before anything expensive. They are signals, never gates: formal, templated or non-native writing
      also uses these phrases. Use them to rank or flag, and combine them with other families.

Run: uv run python examples/slop_and_patterns.py
"""

import json
import re
import tempfile
from pathlib import Path

import hone_taste as tt

lyric = "Neon lights and echoes of the night, we dance in the rain until the city lights fade"
email = "I hope this email finds you well. Let's circle back and leverage our synergy moving forward."

# 1-2. The same text, scored by each preset. A domain preset only adds clichés; it never removes any.
presets = {domain: tt.slop_score(domain=domain) for domain in ("general", "lyrics", "email")}
for slop in presets.values():
    on_lyric, on_email = slop(lyric), slop(email)
    print(
        f"{slop.name:13} lyric={on_lyric.value:.2f} ({len(on_lyric.details['hits'])} hits)  "
        f"email={on_email.value:.2f} ({len(on_email.details['hits'])} hits)"
    )

lyric_hits = [h.get("phrase") or h.get("word") for h in presets["lyrics"](lyric).details["hits"]]
print("lyrics preset found:", lyric_hits)
assert "neon lights" in lyric_hits and "echoes of" in lyric_hits
lyrics_preset, general_preset = presets["lyrics"](lyric).value, presets["general"](lyric).value
assert lyrics_preset is not None and general_preset is not None
assert lyrics_preset < general_preset  # the lyrics preset finds more lyric clichés

email_hits = len(presets["email"](email).details["hits"])
assert email_hits > len(presets["general"](email).details["hits"])  # the email preset finds email clichés

# The per-component rates explain the value (hits per 1,000 words).
print(
    "email rates:", {k: round(v, 1) for k, v in presets["email"](email).details["rates_per_1k_words"].items()}
)

# 3. Your own word lists, in the upstream slop-score format (JSON arrays of [phrase, ...] rows).
with tempfile.TemporaryDirectory() as folder:
    Path(folder, "slop_list.json").write_text(json.dumps([["synergy"], ["paradigm"]]))
    Path(folder, "slop_list_trigrams.json").write_text(json.dumps([["move needle forward"]]))
    house_style = tt.slop_score(wordlists=folder)
    result = house_style("A new paradigm of synergy.")
print("own lists:", [h["word"] for h in result.details["hits"]])
assert [h["word"] for h in result.details["hits"]] == ["paradigm", "synergy"]

# 4. Your own banned patterns: literals match whole phrases in any case; regexes are used as given.
banned = tt.patterns(
    ["as an AI", "in conclusion", re.compile(r"\b(?:very|really) (?:unique|important)\b", re.IGNORECASE)],
    name="house_rules",
    max_hits=4,  # 4 or more hits -> 0.0
)
checked = banned("In conclusion, as an AI I think this is really important. AI-generated, not 'as an AIM'.")
for hit in checked.details["hits"]:
    print(f"  {hit['pattern']!r} matched {hit['match']!r} at {hit['span']}")
print(f"{banned.name}: value={checked.value:.2f} reason={checked.reason!r}")
assert [h["match"] for h in checked.details["hits"]] == ["In conclusion", "as an AI", "really important"]
assert checked.value == 1 - 3 / 4
assert banned("Plain words.").value == 1.0
