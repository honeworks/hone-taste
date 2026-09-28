"""Write your own scorer and use it anywhere a built-in one fits.

What: `tt.Scorer` is a Protocol, not a base class: any callable object with `name`, `family`, `accepts`,
      `license` and `source_data` that returns a `tt.Score` is a scorer. It then works with `tt.combine`,
      `tt.agreement`, profiles and `tt.for_select` like the built-in ones.
How:  1. write a small dataclass with the five attributes and `__call__(input) -> tt.Score`,
      2. return `tt.Score(None, error=...)` for input you cannot score (never 0, never an exception),
      3. put details a person can act on in `details`, and a one-line `reason`,
      4. check it with `isinstance(scorer, tt.Scorer)` and combine it with a built-in scorer.
Why:  house rules ("verses of 6-10 words", "mentions the product name") are often the most useful signal
      and need no model. Pitfalls: `combine` and `for_select` turn an exception into `value=None`, but a
      direct call does not, so return errors instead of raising; your scorer's calls are recorded only
      as part of a built-in scorer that calls it (e.g. inside `tt.combine`).

Run: uv run python examples/custom_scorer.py
"""

from dataclasses import dataclass

import hone_taste as tt


@dataclass(frozen=True)
class LineLength:
    """Lyric lines should be singable: `low`-`high` words per line. Value = share of lines in range."""

    low: int = 4
    high: int = 10
    name: str = "line_length"
    family: str = "human_likeness"
    accepts: frozenset[str] = frozenset({"text"})
    license: str = "Apache-2.0"
    source_data: str = "house rule, no human ratings"

    def __call__(self, text: object, /) -> tt.Score:
        if not isinstance(text, str):
            return tt.Score(None, error=f"accepts text; got {type(text).__name__}")
        lines = [line.split() for line in text.splitlines() if line.strip()]
        if not lines:
            return tt.Score(None, error="no lines to score")
        too_long = [i for i, words in enumerate(lines, 1) if len(words) > self.high]
        too_short = [i for i, words in enumerate(lines, 1) if len(words) < self.low]
        in_range = len(lines) - len(too_long) - len(too_short)
        reason = f"{in_range}/{len(lines)} lines of {self.low}-{self.high} words"
        return tt.Score(
            in_range / len(lines), reason=reason, details={"too_long": too_long, "too_short": too_short}
        )


lines = LineLength()
assert isinstance(lines, tt.Scorer)

song = "Dad's boots by the door\nstill smell like diesel and rain after all these long and lonely years\nhome"
result = lines(song)
print(f"{lines.name}: value={result.value:.2f} reason={result.reason!r} details={dict(result.details)}")
assert result.value == 1 / 3 and result.details == {"too_long": [2], "too_short": [3]}
assert lines("").value is None and lines(3).error == "accepts text; got int"

# It combines with built-in scorers like any other; the combined call is recorded as one span.
slop = tt.slop_score(domain="lyrics")
lyric_check = tt.combine({"lines": 1, "slop": 1}, {"lines": lines, "slop": slop})
combined = lyric_check(song)
print(f"combined: value={combined.value:.2f} reason={combined.reason!r}")
slop_value = slop(song).value
assert result.value is not None and slop_value is not None
assert combined.value == (result.value + slop_value) / 2
