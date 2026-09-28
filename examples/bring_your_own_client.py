"""Bring your own model client: a `TextClient` of your own, and the OpenAI-compatible adapter.

What: the audience panel talks to a model only through a port: a `DecisionClient` (`decide`) or a
      `TextClient` (`complete`), which `tt.audience` wraps in `tt.TextDecisionClient` for you. Any object
      with the right method fits; no subclassing or registration. `hone_taste.testing.contracts` checks
      that it behaves as the ports spec says. The shipped `OpenAITextClient` speaks to OpenAI, Ollama
      `/v1`, vLLM and other compatible servers; here it gets an SDK-shaped fake instead of the network.
How:  1. write a class with `complete(messages, *, schema=None, trace=None, **params)` returning
         `tt.TextResult` (parse JSON into `parsed` when a `schema` is given; set `error` when it is bad),
      2. run `contracts.check_text_client(client)`, then pass the client to `tt.audience(...)`,
      3. for an OpenAI-compatible server: `OpenAITextClient(model, base_url=..., api_key=...)`; tests and
         offline runs pass `client=` (any object with `chat.completions.create(...)`, e.g. `openai.OpenAI`),
      4. pass `generator_model=` to the panel to get a warning when judge and generator share a family.
Why:  the panel should never depend on one vendor SDK. The contract checkers are the same ones the other
      honeworks packages run, so a client that passes them works everywhere. Pitfalls: transport errors
      must raise (the panel records them per persona); a model reply that is not valid JSON is an `error`
      on the result, not an exception. A judge from the generator's own family tends to like its outputs.

Run: uv run python examples/bring_your_own_client.py
"""

import json
import warnings
from types import SimpleNamespace

import hone_taste as tt
from hone_taste.adapters.openai import OpenAITextClient
from hone_taste.testing import contracts


# 1. Your own TextClient. This one is a rule, not a model; a real one would call your SDK here.
class KeywordModel:
    model = "keyword-rules-1"

    def __init__(self):
        self.prompts = []

    def complete(self, messages, *, schema=None, trace=None, **params):
        prompt = messages[-1]["content"]
        self.prompts.append(prompt)
        if schema is None:  # plain text request
            return tt.TextResult(text="OK", model=self.model)
        # The panel's schema asks for {"<question name>": {"answer": <number>, "rationale": "..."}}.
        rating = 2 if "neon" in prompt.lower() else 4
        reply = {name: {"answer": rating, "rationale": "keyword rule"} for name in schema.get("required", [])}
        text = json.dumps(reply)
        return tt.TextResult(text=text, parsed=reply, model=self.model, finish_reason="stop")


mine = KeywordModel()
# 2. Check it against the port contract, then use it.
contracts.check_text_client(mine)  # raises AssertionError with the failing check otherwise
panel = tt.audience(["A blues fan"], "Would you keep listening?", client=mine)  # wrapped for you
result = panel("Neon lights in the rain")
print(f"own client: value={result.value} raw={result.details['personas'][0]['raw']}")
print("  the model saw:", mine.prompts[-1].splitlines()[:2], "...")
assert result.value == 0.25  # rating 2 on the 1-5 scale


# 3. The OpenAI-compatible adapter with an SDK-shaped fake instead of a server.
class FakeCompletions:
    """Answers like `openai.OpenAI().chat.completions`: a reasoning model that thinks, then fences JSON."""

    def __init__(self):
        self.requests = []

    def create(self, **request):
        self.requests.append(request)
        schema = request.get("response_format", {}).get("json_schema", {}).get("schema", {})
        names = schema.get("required", [])
        body = (
            json.dumps({name: {"answer": 5, "rationale": "fresh image"} for name in names}) if names else "OK"
        )
        content = f"<think>the lyric is concrete</think>\n```json\n{body}\n```"
        return SimpleNamespace(
            model=request["model"],
            choices=[SimpleNamespace(message=SimpleNamespace(content=content), finish_reason="stop")],
            usage=SimpleNamespace(prompt_tokens=120, completion_tokens=30),
        )


completions = FakeCompletions()
sdk = SimpleNamespace(chat=SimpleNamespace(completions=completions))
# Real use: OpenAITextClient("gemma3:12b", base_url="http://127.0.0.1:11434/v1", api_key="ollama")
client = OpenAITextClient("gemma3:12b", client=sdk, temperature=0)
contracts.check_text_client(client)

reply = client.complete([{"role": "user", "content": 'Return {"ok": true}.'}], schema={"required": ["ok"]})
print(f"adapter: parsed={reply.parsed} error={reply.error!r} usage={dict(reply.usage)}")
# reply.text keeps the raw answer; <think> blocks and code fences are stripped only before parsing.
assert "<think>" in reply.text
assert reply.error is None
assert reply.parsed == {"ok": {"answer": 5, "rationale": "fresh image"}}

adapter_panel = tt.audience(["A blues fan"], "Would you keep listening?", client=client)
scored = adapter_panel("Dad's boots by the door")
request = completions.requests[-1]
print(
    f"adapter panel: value={scored.value} request keys={sorted(request)} temperature={request['temperature']}"
)
assert scored.value == 1.0 and request["response_format"]["type"] == "json_schema"

# 4. Judge and generator from the same model family -> a warning, also kept in details.
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    same_family = tt.audience(["A blues fan"], "Keep listening?", client=client, generator_model="gemma3:4b")
print("warning:", caught[0].message)
assert any("same family" in str(w.message) for w in caught)
assert "same family" in same_family("Rain on the roof").details["warning"]
