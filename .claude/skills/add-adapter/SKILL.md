---
name: add-adapter
description: Add an implementation of one of hone-taste's ports, or another optional integration, with its extra, contract test and docs. Use when adding support for a new service, library or data source.
---

# Add an adapter

The ports are in `src/hone_taste/ports.py` (`design/current.md` §6): `TextClient`, `DecisionClient`,
`GpuLease`, `RecordSink`. A new port, or a change to one, is a design change: `plan-change` first.

1. **Where.** Adapters go in `src/hone_taste/adapters/` (like `src/hone_taste/adapters/openai.py`), one
   module per extra.
2. **Optional dependency.** Add an extra in `pyproject.toml`; import the third-party library only inside
   the adapter module, and the module only when it is used. The core must still import without it.
3. **Contract.** Run the checker from `hone_taste.testing.contracts` (`check_text_client`,
   `check_decision_client`, `check_gpu_lease`, `check_record_sink`) against it in `tests/contract/`.
4. **Behaviour the port promises.** A client that can't answer leads to `None` with an `error`, never
   `0`, and never to a silent fallback.
5. **Tests with recorded HTTP**, never a live service, in the default suite; real calls only in
   `tests/gpu/` (`real-model-tests`).
6. `sync-docs`: the install line, `docs/adapters.md`, an example if it's a new kind of client.
