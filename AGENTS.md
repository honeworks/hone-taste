# AGENTS.md: hone-taste

Guidance for contributors who use AI coding tools (Claude Code, Codex, Cursor and others) in this
repository. People and tools follow the same rules: [`CONTRIBUTING.md`](CONTRIBUTING.md).

## What this is

`hone-taste` (import `hone_taste`): stand-in scorers for human judgment, meaning taste models,
human-likeness checks, audience panels and personal taste, all returning one `Score` shape.

Read before changing code:
- [`design/current.md`](design/current.md): the design, its rules and the acceptance cases;
- [`design/decisions.md`](design/decisions.md): smaller choices and why (some await owner review);
- [`design/changes/`](design/changes/): why the design is the way it is.

## Commands

```bash
uv sync --all-extras                        # install everything, dev tools included
scripts/check.sh                            # all quality gates; green = exit 0
uv run pytest                               # default suite (fast, offline, deterministic)
uv run pytest tests/e2e                     # acceptance cases only
scripts/gpu-lock.sh uv run pytest -m gpu    # real-model tests, under the machine-wide GPU lock
uv run ruff check . && uv run ruff format . # lint and format
uv run pyright                              # types (strict for src/)
```

## Layout

```text
src/hone_taste/       public API in __init__.py (explicit __all__)
  types.py            Score, Scorer, input kinds, FunctionScorer
  scorers/            slop, patterns, taste-model wrappers, binoculars
  audience.py  profile.py  fitting.py  agreement.py  combine.py  normalize.py  bridge.py
  registry.py         data/models.toml: licenses, whose taste, extras
  ports.py            the Protocols this package owns
  testing/            public fakes and contract checkers
  adapters/           optional integrations (one module per extra)
  _tracing.py  _records.py   trace context and span sinks
tests/unit|contract|integration|e2e|gpu   docs/   examples/   design/   scripts/
```

## Rules

1. Keep it simple: the simplest code that meets the acceptance cases; limits in `CONTRIBUTING.md`.
2. The core never imports an optional extra or another honeworks package; outside services come in
   through the ports, each with a fake and a contract checker.
3. "Could not score" is `None` with an `error`, never 0; typed exceptions for configuration errors.
4. Spans exactly as in `design/current.md` section 7; never record secrets.
5. Every acceptance case has a `tests/e2e/test_ac<N>_*` test; README, docs and examples are executed
   by tests. Never weaken a test to make it pass.
6. Real-model tests only through `scripts/gpu-lock.sh`; unload what you load.
7. A change to what the package promises needs a design change record (`CONTRIBUTING.md`); smaller
   choices go into `design/decisions.md`.
8. Conventional Commits; `scripts/check.sh` green before every commit. Do not push, tag or publish
   unless the maintainer asks.
