"""Slop score: how much a text leans on phrasing that LLMs overuse (higher value = more human-like).

A port of the Slop Score idea (github.com/sam-paech/slop-score, MIT): a weighted composite of overused
words and phrases (60%), "not X, but Y" contrast patterns (25%) and overused trigrams (15%), each counted
per 1,000 words and scaled by a reference maximum. `value = 1 - composite`.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from importlib import resources
from importlib.resources.abc import Traversable
from pathlib import Path
from typing import Any

from hone_taste.errors import ConfigError
from hone_taste.scorers.patterns import compile_pattern, find_hits
from hone_taste.types import FunctionScorer, Score

WEIGHTS = {"words": 0.60, "contrast": 0.25, "trigrams": 0.15}
# Per-1,000-word rates that count as "maximally sloppy": about twice the worst model average on the
# upstream leaderboard (slop words ~40 / 1k words, trigrams ~1.1 / 1k words, contrast ~0.8 / 1k chars,
# i.e. ~4.5 / 1k words), because single short texts vary far more than averages over many texts.
REFERENCE_MAX = {"words": 80.0, "contrast": 9.0, "trigrams": 2.5}
FULL_CONFIDENCE_WORDS = 100  # shorter texts get proportionally lower confidence

PRON = r"(?:it|they|this|that)"
BE = r"(?:is|are|was|were)"
BE_NEG = r"(?:is\s+not|are\s+not|was\s+not|were\s+not|isn't|aren't|wasn't|weren't|ain't)"
SENTENCE_START = r"(?:^|(?<=[.?!]\s))\s*[\"']?"
# Surface forms of the contrast patterns, simplified from upstream js/regexes-stage1.js.
CONTRAST_PATTERNS = {
    "not X, but Y": (
        rf"\b(?:{BE_NEG}|not(?!\s+(?:that|only)\b))\s+(?:(?!\bbut\b)[^.?!]){{1,100}}?[,;:]\s*but\s+"
        r"(?!(?:when|while|which|who|if|that|as|because|although|though|until|unless|then|i|you|we|my)\b)"
        r"(?:(?:a|an|the)\s+)?\w+"
    ),
    "not X - it's Y": (
        rf"\b(?:\w+n't|not)\s+(?:(?:just|only|merely)\s+)?[^.?!]{{1,160}}?(?:\s-\s|--|\u2014|\u2013)\s*"
        rf"{PRON}(?:'s|'re|\s+{BE})\b"
    ),
    "it's not X. it's Y": (
        rf"{SENTENCE_START}(?:{PRON}\s+{BE}\s+not|{PRON}\s+{BE}n't|(?:it's|they're|that's)\s+not)\b"
        rf"[^.?!]{{0,160}}[.;:?!]\s*[\"']?{PRON}(?:'s|'re|\s+{BE})\b(?!\s+not\b)"
    ),
}
CONTRAST = [(name, re.compile(pattern, re.IGNORECASE)) for name, pattern in CONTRAST_PATTERNS.items()]
# NLTK English stopwords (as used upstream); trigrams are matched on the remaining content words.
STOPWORDS = frozenset(
    (  # noqa: SIM905 - a word list reads best as one string
        "i me my myself we our ours ourselves you your yours yourself yourselves he him his himself she her "
        "hers herself it its itself they them their theirs themselves what which who whom this that these "
        "those am is are was were be been being have has had having do does did doing a an the and but if or "
        "because as until while of at by for with about against between into through during before after "
        "above below to from up down in out on off over under again further then once here there when where "
        "why how all any both each few more most other some such no nor not only own same so than too very s "
        "t can will just don should now"
    ).split()
)
QUOTES = str.maketrans({"\u2019": "'", "\u2018": "'", "\u201c": '"', "\u201d": '"'})  # 1:1, spans unchanged
WORD = re.compile(r"[a-z]+(?:'[a-z]+)*", re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class Preset:
    """Extra single words and multi-word phrases a domain adds to the shared lists."""

    extra_words: tuple[str, ...] = ()
    extra_phrases: tuple[str, ...] = ()


PRESETS = {
    "general": Preset(),
    "lyrics": Preset(
        extra_words=("neon", "embers", "shattered", "symphony", "tapestry", "whispers", "echoes"),
        extra_phrases=(
            "echoes of",
            "neon lights",
            "city lights",
            "shattered dreams",
            "fire in my soul",
            "set me free",
            "dance in the rain",
            "into the night",
            "through the night",
            "in the silence",
            "burning bright",
            "hearts collide",
            "under the stars",
            "chasing dreams",
            "broken wings",
        ),
    ),
    "email": Preset(
        extra_words=("delve", "leverage", "synergy", "seamless", "seamlessly"),
        extra_phrases=(
            "i hope this email finds you well",
            "i hope this message finds you well",
            "i hope you are doing well",
            "i wanted to reach out",
            "please don't hesitate to reach out",
            "don't hesitate to contact",
            "at your earliest convenience",
            "circle back",
            "touch base",
            "rest assured",
            "i trust this",
            "thank you for your understanding",
            "looking forward to hearing from you",
            "moving forward",
        ),
    ),
}


def _load_list(path: Traversable) -> set[str]:
    """Read an upstream list file: a JSON array of `[phrase, ...]` rows (or plain strings)."""
    rows: list[Any] = json.loads(path.read_text(encoding="utf-8"))
    return {str(row[0] if isinstance(row, list) else row).lower() for row in rows if row}  # pyright: ignore[reportUnknownArgumentType]


def load_wordlists(directory: str | Path | None = None) -> tuple[set[str], set[str]]:
    """(words, trigrams) from upstream-format files `slop_list.json` / `slop_list_trigrams.json`."""
    folder: Path | Any = (
        Path(directory) if directory is not None else resources.files("hone_taste") / "data" / "slop"
    )
    try:
        return _load_list(folder / "slop_list.json"), _load_list(folder / "slop_list_trigrams.json")
    except FileNotFoundError as exc:
        raise ConfigError(
            f"slop word lists not found in {folder}: expected slop_list.json and slop_list_trigrams.json "
            "(the data/ folder of github.com/sam-paech/slop-score)"
        ) from exc


def _word_hits(
    tokens: list[re.Match[str]], words: set[str], phrase_hits: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Single-word hits, skipping words inside a phrase hit (so "echoes of" is counted once)."""
    inside = [range(h["span"][0], h["span"][1]) for h in phrase_hits]
    return [
        {"word": m.group(0).lower(), "span": [m.start(), m.end()]}
        for m in tokens
        if m.group(0).lower() in words and not any(m.start() in r for r in inside)
    ]


def _trigram_hits(tokens: list[re.Match[str]], trigrams: set[str]) -> list[dict[str, Any]]:
    content = [m for m in tokens if m.group(0).lower() not in STOPWORDS]
    hits: list[dict[str, Any]] = []
    for first, second, third in zip(content, content[1:], content[2:], strict=False):
        trigram = f"{first.group(0)} {second.group(0)} {third.group(0)}".lower()
        if trigram in trigrams:
            hits.append({"trigram": trigram, "span": [first.start(), third.end()]})
    return hits


def slop_score(domain: str = "general", wordlists: str | Path | None = None) -> FunctionScorer:
    """Score how human-like a text reads (1 = no slop found, 0 = saturated with LLM-isms).

    `domain` picks a preset ("general", "lyrics", "email") that adds domain clichés; `wordlists` is an
    optional folder with upstream slop-score list files to use instead of the packaged copy.

    >>> result = slop_score()("It's not just a song, but a journey through the tapestry of time.")
    >>> round(result.value, 2), sorted({h.get("word") or h.get("pattern") for h in result.details["hits"]})
    (0.17, ['not X, but Y', 'tapestry'])
    """
    if domain not in PRESETS:
        raise ConfigError(f"unknown slop domain {domain!r}; choose one of {sorted(PRESETS)}")
    preset = PRESETS[domain]
    words, trigrams = load_wordlists(wordlists)
    words |= set(preset.extra_words)
    phrases = [compile_pattern(p) for p in preset.extra_phrases]

    def score(text: str) -> Score:
        clean = text.translate(QUOTES)
        tokens = list(WORD.finditer(clean))
        if not tokens:
            return Score(None, error="no words to score")
        phrase_hits = find_hits(clean, phrases, label="phrase")
        word_hits = _word_hits(tokens, words, phrase_hits) + phrase_hits
        contrast_hits, trigram_hits = find_hits(clean, CONTRAST), _trigram_hits(tokens, trigrams)
        counts = {"words": len(word_hits), "contrast": len(contrast_hits), "trigrams": len(trigram_hits)}
        rates = {k: counts[k] / len(tokens) * 1000 for k in WEIGHTS}
        slop = sum(WEIGHTS[k] * min(1.0, rates[k] / REFERENCE_MAX[k]) for k in WEIGHTS)
        hits = sorted(word_hits + contrast_hits + trigram_hits, key=lambda h: h["span"])
        details = {"hits": hits, "rates_per_1k_words": rates, "words": len(tokens), "domain": domain}
        reason = f"{len(hits)} slop hit(s) in {len(tokens)} words" if hits else "no slop found"
        confidence = min(1.0, len(tokens) / FULL_CONFIDENCE_WORDS)
        return Score(1.0 - slop, confidence=confidence, reason=reason, details=details)

    return FunctionScorer(
        f"slop_{domain}" if domain != "general" else "slop",
        "human_likeness",
        frozenset({"text"}),
        score,
        license="MIT (slop-score word lists, Sam Paech); presets Apache-2.0",
        source_data="slop-score lists: words / trigrams LLMs overuse vs human writing "
        "(github.com/sam-paech/slop-score)",
    )
