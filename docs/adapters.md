# Bring your own client, GPU lease or model

hone-taste talks to the outside through small ports (`typing.Protocol`); anything with the right
methods works, no subclassing or registration.

| Port | Methods | Used by | Shipped |
|---|---|---|---|
| `tt.DecisionClient` | `decide(state, questions, *, images=(), trace=None) -> {name: answer}` | `tt.audience` | `testing.FakeDecisionClient`, `tt.TextDecisionClient` |
| `tt.TextClient` | `complete(messages, *, schema=None, trace=None, **params) -> TextResult` | `tt.audience` (wrapped) | `adapters.openai.OpenAITextClient`, `testing.FakeTextClient` |
| `tt.GpuLease` | `lease(name, vram_gb, *, timeout_s=None, trace=None)` context manager | taste models (`gpu=`) | `tt.NullGpuLease` (default), `testing.FakeGpuLease` |
| `tt.RecordSink` | `emit(span)`, `flush()`, `close()` | recording | SQLite, JSONL, memory, null sinks |
| taste-model backend | `load()`, `predict(input) -> {name: raw}`, `unload()` | every taste model (`model=`) | `testing.FakeTasteModel` |

Example: [`bring_your_own_client.py`](../examples/bring_your_own_client.py) (your own `TextClient`, the
OpenAI adapter over an SDK-shaped fake).

Check your implementation with the same contract checkers the honeworks packages use:

```python
import contextlib

from hone_taste.testing import FakeDecisionClient, contracts


class MyLease:
    @contextlib.contextmanager
    def lease(self, name, vram_gb, *, timeout_s=None, trace=None):
        yield  # reserve / release your GPU here


contracts.check_gpu_lease(MyLease())
contracts.check_decision_client(FakeDecisionClient())
```

A taste-model backend can wrap any model you like; the scorer handles lazy loading, the lease and
`close()`:

```python
import hone_taste as tt


class LengthModel:  # stands in for a real model
    def load(self):
        print("loading")

    def predict(self, input):
        return {"CE": 7.0, "CU": 7.0, "PC": 3.0, "PQ": min(10.0, len(input) / 2)}

    def unload(self):
        print("unloading")


scorer = tt.audiobox(model=LengthModel())
print(scorer("a-long-file-name.wav").value)
scorer.close()
```
