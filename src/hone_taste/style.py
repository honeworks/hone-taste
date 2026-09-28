"""Style match: does a text read like this author? Stylometry plus a judge anchored on real excerpts.

The honesty rule for this scorer (design/changes/0004): a judge given a style *description* rewards
imitations of the description (persona-writer's style-sheet judge preferred its own imitations over the
real author in 3 of 3 cases, while stylometry ranked every pair right). So the judge here sees only the
author's real excerpts, and `calibrate_style` checks every scorer against held-out real texts before
you trust it or set the weights.
"""

from __future__ import annotations

import dataclasses
import json
import statistics
from collections.abc import Mapping, Sequence
from typing import Any

from hone_taste._tracing import current_trace
from hone_taste.agreement import AgreementReport, agreement
from hone_taste.audience import as_decision_client
from hone_taste.combine import combine
from hone_taste.errors import ConfigError
from hone_taste.ports import DecisionClient, TextClient, get_field
from hone_taste.scorers.stylometry import Fingerprint, paragraphs, stylometry
from hone_taste.types import FunctionScorer, Score, Scorer

MAX_JUDGED_WORDS = 2500
EXCERPT_WORDS = (30, 150)  # a good excerpt paragraph: long enough to carry a voice, short enough to read
DEFAULT_WEIGHTS = {"stylometry": 0.5, "style_judge": 0.5}


def pick_excerpts(texts: Sequence[str], count: int = 5) -> list[str]:
    """Up to `count` prose paragraphs, round robin over the texts, preferring 30-150 words.

    >>> pick_excerpts(["# Title\\n\\nShort.\\n\\n" + "word " * 40, "Another one."], count=2)[1]
    'Another one.'
    """
    per_text: list[list[str]] = []
    for text in texts:
        found = paragraphs(text)
        good = [p for p in found if EXCERPT_WORDS[0] <= len(p.split()) <= EXCERPT_WORDS[1]]
        per_text.append(good or [" ".join(p.split()[: EXCERPT_WORDS[1]]) for p in found[:1]])
    picked: list[str] = []
    for round_ in range(max(map(len, per_text), default=0)):
        picked += [found[round_] for found in per_text if round_ < len(found)]
    return picked[:count]


def _questions(author: str, excerpts: Sequence[str]) -> dict[str, dict[str, Any]]:
    """Anchored on the real excerpts; exaggeration counts against the text."""
    reference = "\n\n".join(f'"{e}"' for e in excerpts)
    ref = f"Reference passages, all written by {author}:\n{reference}\n\n"
    return {
        "voice": {
            "type": "score",
            "instructions": (
                f"{ref}Compared with the reference passages, how close is the text's voice: word choice, "
                "sentence rhythm, how plainly things are stated, how emphasis, headings and lists are used? "
                "Ignore the topic. Imitations usually exaggerate an author (more drama, vivid scene-setting, "
                "bold phrases, catchphrases, talking to 'you'): count exaggeration as a mismatch."
            ),
            "scale": [1, 5],
            "anchors": {"1": "a different writer", "3": "similar in places", "5": "indistinguishable"},
        },
        "plain": {
            "type": "yes_no",
            "instructions": f"{ref}Does the text state things as plainly and matter-of-factly as the "
            "reference?",
        },
        "exaggerates": {
            "type": "yes_no",
            "instructions": (
                f"{ref}Does the text add drama or flourishes the reference passages do not have (vivid "
                "scene-setting, rhetorical questions, bold phrases for effect, pep-talk endings)?"
            ),
        },
    }


def _judge_score(answers: Mapping[str, Any]) -> Score:
    """Mean of the voice score and the yes/no answers (`exaggerates` counts as 1 - P(yes))."""
    values: dict[str, float] = {}
    errors: list[str] = []
    for key in ("voice", "plain", "exaggerates"):
        answer = answers.get(key)
        value = get_field(answer, "value") if answer is not None else None
        if value is None:
            errors.append(f"{key}: {get_field(answer, 'error') or 'not answered'}")
        else:
            values[key] = 1 - value if key == "exaggerates" else value
    voice = answers.get("voice")
    rationale = str(get_field(voice, "rationale") or "") if voice is not None else ""
    details = {"answers": values, "voice_raw": get_field(voice, "raw") if voice is not None else None}
    if not values:
        return Score(None, details=details, error=f"the judge answered nothing usable ({'; '.join(errors)})")
    details = {**details, "voice_rationale": rationale, "errors": errors}
    return Score(statistics.fmean(values.values()), reason=rationale[:300], details=details)


def style_judge(
    reference_texts: Sequence[str],
    client: DecisionClient | TextClient,
    *,
    author: str = "the author",
    excerpts: int = 5,
) -> FunctionScorer:
    """An LLM judge given only real excerpts of the author (never a style description): a 1-5 "how close
    is the voice" question plus "as plain?" and "adds drama?" (inverted); value = their mean.

    >>> from hone_taste.testing import FakeDecisionClient
    >>> judge = style_judge(["I fixed the bug. It was a typo in the loop bound."], FakeDecisionClient())
    >>> judge("I found the bug.").value
    0.5
    """
    chosen = pick_excerpts(reference_texts, excerpts)
    if not chosen:
        raise ConfigError("style_judge needs reference texts with at least one prose paragraph")
    questions = _questions(author, chosen)
    decision_client = as_decision_client(client)

    def score(text: str) -> Score:
        clipped = " ".join(text.split(" ")[:MAX_JUDGED_WORDS])
        result = _judge_score(decision_client.decide(clipped, questions, trace=current_trace()))
        return dataclasses.replace(result, details={**result.details, "excerpts": len(chosen)})

    return FunctionScorer(
        "style_judge",
        "audience",
        frozenset({"text"}),
        score,
        license="depends on the client's model",
        source_data=f"an LLM judge given {len(chosen)} real excerpts by {author}",
    )


def style_match(
    reference_texts: Sequence[str],
    client: DecisionClient | TextClient,
    *,
    fingerprint: Fingerprint | None = None,
    weights: Mapping[str, float] | None = None,
    author: str = "the author",
    excerpts: int = 5,
) -> FunctionScorer:
    """Does a text read like the author? `stylometry` and `style_judge`, combined with `weights`
    (default 0.5 / 0.5; set them from `calibrate_style`). If the judge fails, the value is stylometry's
    alone and the judge's error is kept in `details["parts"]`, never a silent 0.

    >>> from hone_taste.testing import FakeDecisionClient
    >>> texts = ["I tried it. It broke. I fixed it.", "I wrote a test. It failed. I read the code."]
    >>> match = style_match(texts, FakeDecisionClient(), weights={"stylometry": 1, "style_judge": 1})
    >>> sorted(match("I ran it and it crashed.").details["parts"])
    ['style_judge', 'stylometry']
    """
    parts = {
        "stylometry": stylometry(fingerprint or reference_texts),
        "style_judge": style_judge(reference_texts, client, author=author, excerpts=excerpts),
    }
    combined = combine(dict(weights or DEFAULT_WEIGHTS), parts)
    sources = f"stylometry and an LLM judge over real texts by {author}"
    return dataclasses.replace(combined, name="style_match", family="personal", source_data=sources)


@dataclasses.dataclass(frozen=True, slots=True)
class StyleCalibration:
    """Real texts vs imitations: how often each scorer prefers the real one, and the weights that follow."""

    report: AgreementReport  # pairs are (real, imitation): the real text is the "human pick"
    weights: dict[str, float]  # each scorer by its correlation with the real texts (negatives count 0)
    means: dict[str, dict[str, float | None]]  # group ("real", then each imitation group) -> scorer -> mean
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "agreement": self.report.to_dict(),
            "weights": self.weights,
            "means": self.means,
            "note": self.note,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    def __str__(self) -> str:
        names = list(self.weights)
        header = [f"| text | {' | '.join(names)} |", "|---" * (len(names) + 1) + "|"]
        rows = [
            f"| {group} | " + " | ".join(_fmt(m[n]) for n in names) + " |" for group, m in self.means.items()
        ]
        weights = ", ".join(f"{n}={w:.2f}" for n, w in self.weights.items())
        agreement = str(self.report).replace("human picks", "(real, imitation) pairs", 1)
        table = "\n".join([*header, *rows])
        return f"{agreement}\n\nmean values:\n{table}\n\nweights: {weights} {self.note}".rstrip()


def _fmt(value: float | None) -> str:
    return "-" if value is None else f"{value:.2f}"


def _memo(scorer: Scorer, cache: dict[str, Score]) -> FunctionScorer:
    def score(text: str) -> Score:
        if text not in cache:
            cache[text] = scorer(text)
        return cache[text]

    return FunctionScorer(scorer.name, scorer.family, frozenset({"text"}), score)


def calibrate_style(
    scorers: Mapping[str, Scorer], real: Sequence[str], imitations: Mapping[str, Sequence[str]]
) -> StyleCalibration:
    """Check style scorers against held-out real texts: each `real[i]` should beat `group[i]` of every
    imitation group (e.g. a generic text and a ghostwritten one on the same subject). Every text is
    scored once per scorer. `weights` weight each scorer by its non-negative correlation; pass them to
    `style_match(..., weights=...)` when the scorers are named `stylometry` and `style_judge`.

    >>> from hone_taste.types import FunctionScorer, Score
    >>> calm = lambda text: Score(1 - text.count("!") / 5)
    >>> plain = FunctionScorer("plain", "personal", frozenset({"text"}), calm)
    >>> cal = calibrate_style({"plain": plain}, ["It works."], {"ghost": ["It works!!"]})
    >>> cal.report.scorers["plain"].rate, cal.weights
    (1.0, {'plain': 1.0})
    """
    if not scorers or not real or not imitations:
        raise ConfigError("calibrate_style needs scorers, real texts and at least one imitation group")
    bad = sorted(name for name, texts in imitations.items() if len(texts) != len(real))
    if bad:
        raise ConfigError(f"imitation groups {bad} must have one text per real text ({len(real)})")
    caches: dict[str, dict[str, Score]] = {name: {} for name in scorers}
    memo = {name: _memo(s, caches[name]) for name, s in scorers.items()}
    pairs = [(r, group[i]) for group in imitations.values() for i, r in enumerate(real)]
    report = agreement(memo, pairs)
    groups = {"real": list(real), **{k: list(v) for k, v in imitations.items()}}
    means = {
        g: {n: _mean([caches[n][t].value for t in texts]) for n in scorers} for g, texts in groups.items()
    }
    correlation = {n: max(0.0, report.scorers[n].correlation or 0.0) for n in scorers}
    total = sum(correlation.values())
    if not total:
        equal = {n: round(1 / len(scorers), 3) for n in scorers}
        return StyleCalibration(report, equal, means, "(no scorer agreed with the real texts: equal weights)")
    return StyleCalibration(report, {n: round(c / total, 3) for n, c in correlation.items()}, means)


def _mean(values: list[float | None]) -> float | None:
    known = [v for v in values if v is not None]
    return round(statistics.fmean(known), 3) if known else None
