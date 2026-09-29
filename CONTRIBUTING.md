# Contributing to hone-taste

Thanks for helping. This page covers setup, the quality gates, the code and test conventions, and how to
propose a design change.

## Setup and the quality gates

```bash
uv sync --all-extras          # Python 3.11+; installs every extra and the dev tools (the `dev` group)
scripts/check.sh              # all quality gates; "green" means this exits 0
```

`scripts/check.sh` runs, in order: `uv sync --frozen`, `ruff check`, `ruff format --check`, `pyright`
(strict for `src/`), the default test suite with branch coverage of at least 90%, `uv build`, and a smoke
test that installs the wheel alone in a fresh virtual environment and runs the README quickstart. CI runs
the same checks on Python 3.11-3.13, but installs only the light extras (`cli`, `openai` and the
`audio-tests` group of CPU audio libraries), so a GitHub runner never downloads torch or CUDA:

```bash
uv sync --extra cli --extra openai --group audio-tests   # what CI installs; the default suite passes with it
```

Other useful commands:

```bash
uv run pytest                          # default suite: fast, offline, deterministic
uv run pytest tests/e2e                # acceptance cases only
uv run ruff check . && uv run ruff format .
uv run python examples/quickstart.py   # any example
```

## Keep it simple

Simple, easy to understand, maintainable code comes first after correctness. A newcomer should be able to
open any module and understand it in a few minutes.

- Build the simplest thing that meets the acceptance cases in
  [`design/current.md`](design/current.md#10-acceptance-cases), not the most general thing.
- No speculative generality: add an abstraction only when there are two real uses today, or when it is
  one of the named seams (a port, the model registry, a user-supplied function or backend).
- Plain Python first: functions and small dataclasses before classes; composition before inheritance;
  variants as a table of functions, not a class hierarchy; the standard library before dependencies.
- Flat and explicit: short call chains, no metaclasses, import hooks, monkey-patching or deep decorator
  stacks. One obvious way to do each thing.
- Names over comments; comments say *why*.
- Delete freely: dead code, unused parameters and single-caller helper layers go.

| Limit | Value |
|---|---|
| Cyclomatic complexity per function | 10 (ruff `C901` enforces it) |
| Function length | about 40 lines |
| Module length | about 300 lines |
| Parameters per function | 6 (group more into a small dataclass) |
| Own inheritance depth | 1 |
| Core dependencies | standard library and pydantic; anything else needs a recorded decision |

An exception is fine when it is clearly simpler, with a one-line comment saying why.

## Code conventions

- **Useful alone:** the core never imports an optional extra or another honeworks package. Heavy
  dependencies (torch, transformers ...) are imported only inside the backends that need them. A test
  checks the core imports without them.
- **Ports:** outside services come in through the small `typing.Protocol`s in `ports.py`, passed as
  arguments. Every port has a public fake in `hone_taste.testing` and a contract checker in
  `hone_taste.testing.contracts`.
- **Explicit failure:** "could not score" is `Score(None, error=...)`, never 0. One bad input never raises;
  configuration errors raise typed exceptions from `hone_taste.errors` with a message that says what to do.
- **Records:** every scored input writes one `hone.taste.score` span with the attributes in
  [`design/current.md`](design/current.md#7-records). Never record secrets.
- **Determinism:** explicit seeds; never use the built-in `hash()` for ids (use `hashlib`).
- **Style:** ruff (line length 110) and pyright strict for `src/`; public functions have docstrings with
  a short example (doctests run in the default suite); `pathlib.Path` for paths; UTC times; no prints in
  library code.
- **Honesty:** a new taste model gets a row in `src/hone_taste/data/models.toml` with its license, whose
  ratings it learned from and its extra; a model with unclear terms is gated behind `accept_license=True`,
  and the README table is updated (a test checks it matches the registry).

## Tests

| Suite | Folder | Runs by default | Uses |
|---|---|---|---|
| Unit | `tests/unit/` | yes | fakes only |
| Contract | `tests/contract/` | yes | the port checkers against fakes and fake transports |
| Integration | `tests/integration/` | yes | SQLite, files, subprocesses |
| Acceptance | `tests/e2e/` | yes | the public API and CLI, exactly as a user calls them |
| Real models | `tests/gpu/` | no | real local models, through the GPU lock (below) |

- Every acceptance case in [`design/current.md`](design/current.md#10-acceptance-cases) has a test
  `tests/e2e/test_ac<N>_<slug>.py` that uses only the public API and fails if the feature is removed.
- The README, every Python block in `docs/` and every file in `examples/` run as tests. A new public
  concept gets an example with a What / How / Why / Run docstring, listed in `examples/README.md`.
- Relative links in the README, `docs/`, `examples/README.md`, this file and `design/` are checked.
- Tests are deterministic and offline; never weaken a test to make it pass.

### Real-model tests and `scripts/gpu-lock.sh`

Real-model tests (`-m gpu`) load actual models (a local Ollama server, torch models) and are kept short.
Run them through `scripts/gpu-lock.sh`, which holds a machine-wide file lock
(`${HONE_GPU_LOCK:-/tmp/honeworks-gpu.lock}`) so test runs of several packages never share one GPU at the
same time:

```bash
scripts/gpu-lock.sh uv run pytest -m gpu -q -rs
```

Tests skip with a clear reason when a server, model or download is unavailable, and unload what they
loaded. Model names come from environment variables (`HONE_TEST_OLLAMA_URL`, default
`http://127.0.0.1:11434`; `HONE_TEST_TEXT_MODEL`, default `gemma3:12b`).

## Changing the design

Changes that alter what the package promises (public API, score semantics, ports, span attributes, file
formats, acceptance cases) are written down before they are built:

1. Write `design/changes/NNNN-<short-name>.md` with status `proposed` and the sections Status, Context,
   Problem, Options, Decision, Consequences, and Migration and compatibility.
2. The maintainer reviews it; the status becomes `accepted` (or `rejected`).
3. Implement it from [`design/current.md`](design/current.md) and the accepted change records, tests
   first.
4. Update `design/current.md`, set the record to `implemented in <version>`, and add a `CHANGELOG.md`
   entry that links to it.

Smaller implementation choices that need no change record go into
[`design/decisions.md`](design/decisions.md).

## Pull requests

- One branch per change, named `<type>/<short-name>` after the Conventional Commit types (`feat/ftp-storage`).
- Fill in [`.github/pull_request_template.md`](.github/pull_request_template.md): what, why (issue and
  change record), how it was tested, which docs changed, and the end of the `scripts/check.sh` output.
- claude[bot] reviews every pull request, with inline comments and suggested changes
  ([`.github/workflows/claude-review.yml`](.github/workflows/claude-review.yml)); on a pull request from a
  fork, the maintainer starts it with a `@claude review` comment. Answer each thread: agree and fix,
  disagree with a reason, or ask. A thread is resolved when it is fixed or decided.
- `main` accepts changes only through pull requests, with CI green and every review thread resolved.
- The code owners in [`.github/CODEOWNERS`](.github/CODEOWNERS) are asked to review automatically.
- The maintainer merges.

## Commits

- [Conventional Commits](https://www.conventionalcommits.org/) (`feat:`, `fix:`, `docs:`, `test:`,
  `refactor:`, `chore:`, `build:`, `ci:`), subject of at most 72 characters, a body that explains why, one
  logical change per commit, `scripts/check.sh` green before committing.
- When an AI tool wrote the change, end the message with a `Co-Authored-By:` line naming it.
- Never commit secrets, `.hone/`, model weights or large binaries.
- User-visible changes get a line in [`CHANGELOG.md`](CHANGELOG.md) (Keep a Changelog), linking to the
  design change record when there is one.

Contributors using AI coding tools will find a short brief for them in [`AGENTS.md`](AGENTS.md); Claude
Code users also get the whole workflow as skills, reviewer agents and hooks in [`.claude/`](.claude/).
