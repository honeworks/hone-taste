"""AC-18 [real]: audience panel through the OpenAI adapter to Ollama's /v1 endpoint."""

import os

import pytest

import hone_taste as tt
from hone_taste.testing import contracts

OLLAMA_URL = os.environ.get("HONE_TEST_OLLAMA_URL", "http://127.0.0.1:11434")

pytestmark = [pytest.mark.gpu, pytest.mark.ollama]

CLICHED = (
    "Neon lights and whispered dreams, echoes of a broken heart,\n"
    "we dance beneath the endless sky, a tapestry torn apart."
)
FRESH = (
    "Dad's work boots by the door still smell like diesel and rain,\n"
    "Mom won't move them, says the floor forgets him if they're gone."
)


@pytest.mark.timeout(900)  # about 30 s per decision on the shared 8 GB GPU
def test_ac18_audience_panel_via_openai_adapter(ollama_model) -> None:
    pytest.importorskip("openai")
    from hone_taste.adapters.openai import OpenAITextClient

    model = ollama_model("HONE_TEST_TEXT_MODEL", "gemma3:12b")
    client = OpenAITextClient(model, base_url=f"{OLLAMA_URL}/v1", api_key="ollama")
    contracts.check_text_client(client)
    decisions = tt.TextDecisionClient(client, params={"temperature": 0, "seed": 7})
    contracts.check_decision_client(decisions)

    panel = tt.audience(
        [
            "A 25-year-old who streams blues-rock and skips songs within 20 seconds",
            "A 55-year-old live-blues fan who hates clichés",
        ],
        question="Would you keep listening past the first chorus of a song with these lyrics?",
        client=decisions,
    )
    cliched, fresh = panel(CLICHED), panel(FRESH)
    assert cliched.value is not None, cliched.error
    assert fresh.value is not None, fresh.error
    assert len(fresh.details["personas"]) == 2
    assert cliched.value <= fresh.value  # tolerant: the clichéd lyric never wins
