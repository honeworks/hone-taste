"""Public fakes for the ports and the contract checkers. No GPU, no network."""

from hone_taste.testing import contracts
from hone_taste.testing.fakes import (
    FakeDecisionClient,
    FakeEmbedder,
    FakeGpuLease,
    FakeTasteModel,
    FakeTextClient,
    ScriptedIO,
)

__all__ = [
    "FakeDecisionClient",
    "FakeEmbedder",
    "FakeGpuLease",
    "FakeTasteModel",
    "FakeTextClient",
    "ScriptedIO",
    "contracts",
]
