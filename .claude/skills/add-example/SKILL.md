---
name: add-example
description: Add a runnable, explained example to examples/ that the tests execute. Use when a change adds a concept users should see, or the user asks for an example.
---

# Add an example

1. One concept per file: `examples/<name>.py`, in the shape of `examples/agreement.py`:
   - a docstring with **What:**, **How:**, **Why:** and **Run:**, in that order;
   - offline: the package's fakes, no network, no GPU, no model downloads; prints what it shows and
     `assert`s the key facts;
2. List it in `examples/README.md`, in reading order.
3. The test finds it by itself; check that the README row links the file.
4. `uv run pytest tests/e2e/test_ac20_examples.py -q`.
