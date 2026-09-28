"""Stylometry: does a text read like the reference texts, measured without any model?

`fingerprint(texts)` summarises an author's texts: the mean and spread of 15 surface features (sentence
length, paragraph length, I / we / you, contractions, questions, dashes, parentheses, word length,
headings, lists, code, bold) and the rates of 80 function words. `stylometry(...)` scores a text by its
closeness to that fingerprint: half from the surface features (z-scores), half from Burrows' Delta over
the function words. Blind to meaning, so it cannot be fooled by "sounding stylish" (design/changes/0004).
"""

from __future__ import annotations

import dataclasses
import math
import re
import statistics
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

from hone_taste.errors import ConfigError
from hone_taste.types import FunctionScorer, Score

# Frequent English function words: their rates are a classic authorship signal (Mosteller & Wallace,
# Burrows). Content words are left out on purpose: a new topic must not look like a new author.
FUNCTION_WORDS = (  # noqa: SIM905 - a few lines of words read better than an 80-line list
    "the a an and but or so if then than that this these those it its is are was were be been being "
    "have has had do does did not no of in on at to for from with by about as into through over i me my "
    "we our you your they them their he she which what when where why how all some any just only also "
    "even very really more most much can could would should will may might must"
).split()
SENTENCE_END = re.compile(r"(?<=[.!?])[\"')\]]*\s+(?=[A-Z0-9\"'(\[`*_])")
WORD = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
CONTRACTION = re.compile(r"\b[A-Za-z]+'(?:s|t|re|ve|ll|d|m)\b", re.IGNORECASE)
FIRST_PERSON = ("i", "me", "my", "i'm", "i've", "i'd", "i'll", "mine")
WE = ("we", "our", "us", "we're", "we've", "let's")
YOU = ("you", "your", "you're", "you'll", "you've")
Stat = tuple[float, float]  # (mean, spread)


def prose(text: str) -> str:
    """The text without fenced code blocks, with curly apostrophes made straight."""
    return re.sub(r"```.*?```", "", text, flags=re.DOTALL).replace("\u2019", "'")


def paragraphs(text: str) -> list[str]:
    """Prose paragraphs: not headings, lists, quotes, tables or code."""
    blocks = (b.strip() for b in re.split(r"\n\s*\n", prose(text)))
    return [" ".join(b.split()) for b in blocks if b and not re.match(r"#|>|\||[*\-+] |\d+\. |    ", b)]


def sentences(text: str) -> list[str]:
    return [s for p in paragraphs(text) for s in SENTENCE_END.split(p) if WORD.search(s)]


def _code_lines(text: str) -> int:
    count, inside = 0, False
    for line in text.splitlines():
        if line.strip().startswith("```"):
            inside = not inside
        elif inside and line.strip():
            count += 1
    return count


def features(text: str) -> dict[str, float]:
    """Per-text surface features; rates per 100 words unless the name says per 1,000."""
    body = prose(text)
    words = WORD.findall(body)
    if not words:
        raise ValueError("the text has no words")
    counts = Counter(w.lower() for w in words)
    lengths = [len(WORD.findall(s)) for s in sentences(text)] or [0]
    lines = body.splitlines()
    code = _code_lines(text)
    per100, per1000 = 100 / len(words), 1000 / len(words)
    return {
        "sentence_words": statistics.fmean(lengths),
        "sentence_words_sd": statistics.pstdev(lengths),
        "paragraph_sentences": len(lengths) / max(1, len(paragraphs(text))),
        "i_rate": sum(counts[w] for w in FIRST_PERSON) * per100,
        "we_rate": sum(counts[w] for w in WE) * per100,
        "you_rate": sum(counts[w] for w in YOU) * per100,
        "contractions": len(CONTRACTION.findall(body)) * per100,
        "questions_per_sentence": sum(s.rstrip().endswith("?") for s in sentences(text)) / len(lengths),
        "dashes": (body.count("—") + body.count(" -- ") + body.count(" - ")) * per100,
        "parentheses": body.count("(") * per100,
        "word_length": statistics.fmean(len(w) for w in words),
        "headings_per_1000": sum(line.startswith("#") for line in lines) * per1000,
        "list_items_per_1000": sum(bool(re.match(r"\s*([*\-+]|\d+\.) ", ln)) for ln in lines) * per1000,
        "code_share": code / max(1, code + sum(1 for ln in lines if ln.strip())),
        "bold_per_1000": len(re.findall(r"\*\*[^*]+\*\*", body)) * per1000,
    }


def word_rates(text: str) -> dict[str, float]:
    """Function-word rates per 1,000 words."""
    words = [w.lower() for w in WORD.findall(prose(text))]
    counts, n = Counter(words), max(1, len(words))
    return {w: counts[w] * 1000 / n for w in FUNCTION_WORDS}


def _stat(values: list[float]) -> Stat:
    return (statistics.fmean(values), statistics.pstdev(values))


def _spread(stat: Sequence[float]) -> float:
    """A spread never below 25 % of the mean (or 0.05): few texts must not make one feature brittle."""
    mean, sd = stat
    return max(sd, 0.25 * abs(mean), 0.05)


@dataclasses.dataclass(frozen=True, slots=True)
class Fingerprint:
    """An author's measurable style: `(mean, spread)` per feature and per function word. JSON-ready via
    `to_dict()` / `Fingerprint.from_dict()`, so it can be stored with a profile."""

    features: dict[str, Stat]
    function_words: dict[str, Stat]
    typical_delta: float  # mean Delta of the author's own texts to this fingerprint
    texts: int

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Fingerprint:
        def stats(table: Mapping[str, Sequence[float]]) -> dict[str, Stat]:
            return {k: (float(v[0]), float(v[1])) for k, v in table.items()}

        return cls(
            stats(data["features"]),
            stats(data["function_words"]),
            float(data["typical_delta"]),
            int(data["texts"]),
        )


def fingerprint(texts: Sequence[str]) -> Fingerprint:
    """The fingerprint of an author's texts (at least two; about ten give a stable one).

    >>> fp = fingerprint(["I think so. It is fine.", "I tried it. It worked, mostly."])
    >>> fp.texts, round(fp.features["sentence_words"][0], 2)
    (2, 3.0)
    """
    if len(texts) < 2:
        raise ConfigError("a fingerprint needs at least two reference texts (about ten give a stable one)")
    try:
        feats = [features(t) for t in texts]
    except ValueError as exc:
        raise ConfigError(f"every reference text needs words: {exc}") from exc
    rates = [word_rates(t) for t in texts]
    partial = Fingerprint(
        {k: _stat([f[k] for f in feats]) for k in feats[0]},
        {w: _stat([r[w] for r in rates]) for w in FUNCTION_WORDS},
        0.0,
        len(texts),
    )
    return dataclasses.replace(partial, typical_delta=statistics.fmean(delta(t, partial) for t in texts))


def delta(text: str, fp: Fingerprint) -> float:
    """Burrows' Delta: the mean absolute z-score of the function-word rates."""
    rates = word_rates(text)
    return statistics.fmean(abs(rates[w] - s[0]) / _spread(s) for w, s in fp.function_words.items())


def closeness(text: str, fp: Fingerprint) -> Score:
    """0-1 closeness of `text` to the fingerprint, with the per-feature z-scores furthest from it."""
    feats = features(text)
    z = {k: (feats[k] - s[0]) / _spread(s) for k, s in fp.features.items()}
    surface = statistics.fmean(max(0.0, 1 - abs(v) / 3) for v in z.values())
    d = delta(text, fp)
    words = math.exp(-max(0.0, d - fp.typical_delta) / max(fp.typical_delta, 1e-6))
    far = sorted(z.items(), key=lambda kv: -abs(kv[1]))[:4]
    details = {
        "surface": round(surface, 3),
        "function_words": round(words, 3),
        "delta": round(d, 3),
        "typical_delta": round(fp.typical_delta, 3),
        "furthest": {
            k: {"z": round(v, 2), "text": round(feats[k], 3), "reference": round(fp.features[k][0], 3)}
            for k, v in far
        },
        "reference_texts": fp.texts,
    }
    reason = "furthest: " + ", ".join(f"{k} z={v:.1f}" for k, v in far)
    confidence = min(1.0, len(WORD.findall(prose(text))) / 400)
    return Score(0.5 * surface + 0.5 * words, confidence=confidence, reason=reason, details=details)


def stylometry(reference: Sequence[str] | Fingerprint, *, name: str = "stylometry") -> FunctionScorer:
    """A scorer: how close is a text's measurable style to the reference texts (or a stored fingerprint)?

    >>> ref = ["I tried it. It broke. I fixed it.", "I wrote a test. It failed. I read the code."]
    >>> result = stylometry(ref)("I ran it. It crashed. I read the parser.")
    >>> 0 <= result.value <= 1, result.details["reference_texts"], sorted(result.details)[:2]
    (True, 2, ['delta', 'function_words'])
    """
    fp = reference if isinstance(reference, Fingerprint) else fingerprint(reference)
    return FunctionScorer(
        name,
        "personal",
        frozenset({"text"}),
        lambda text: closeness(text, fp),
        license="Apache-2.0",
        source_data=f"stylometry of {fp.texts} reference texts (no model, no ratings)",
    )
