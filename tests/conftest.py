"""Shared pytest fixtures: GPU lock and real-model availability checks."""

from __future__ import annotations

import fcntl
import json
import os
import tempfile
import urllib.request
from collections.abc import Iterator
from pathlib import Path

import pytest
from hypothesis import settings

# The default suite must be deterministic: hypothesis explores a fixed set of examples.
settings.register_profile("deterministic", derandomize=True, max_examples=100)
settings.load_profile("deterministic")

# Scores are recorded to ${HONE_HOME}/taste/spans.db by default; keep test runs out of the repo.
os.environ["HONE_HOME"] = tempfile.mkdtemp(prefix="hone-taste-tests-")

OLLAMA_URL = os.environ.get("HONE_TEST_OLLAMA_URL", "http://127.0.0.1:11434")


@pytest.fixture(scope="session")
def gpu_lock() -> Iterator[None]:
    """Hold the machine-wide GPU lock for the session (no-op if scripts/gpu-lock.sh already holds it)."""
    if os.environ.get("HONE_GPU_LOCK_HELD") == "1":
        yield
        return
    path = Path(os.environ.get("HONE_GPU_LOCK", "/tmp/honeworks-gpu.lock"))  # noqa: S108 - shared machine-wide lock by design
    path.touch(exist_ok=True)
    with path.open("r+") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


def ollama_models() -> list[str]:
    try:
        with urllib.request.urlopen(f"{OLLAMA_URL}/api/tags", timeout=3) as r:  # noqa: S310
            return [m["name"] for m in json.load(r).get("models", [])]
    except OSError:
        return []


@pytest.fixture
def ollama_model(gpu_lock: None):
    """Factory: ollama_model("HONE_TEST_TEXT_MODEL", "gemma3:12b") -> name, or skip with a reason.

    Models used by a test are unloaded right after it, so the next test gets the whole GPU.
    """
    available = ollama_models()

    def _get(env: str, default: str) -> str:
        name = os.environ.get(env, default)
        if not available:
            pytest.skip(f"Ollama not reachable at {OLLAMA_URL}")
        if name not in available:
            pytest.skip(f"Ollama model {name!r} ({env}) not installed")
        return name

    loaded: list[str] = []

    def _use(env: str, default: str) -> str:
        name = _get(env, default)
        loaded.append(name)
        return name

    yield _use
    for name in set(loaded):  # free the shared GPU for the next test run
        try:
            req = urllib.request.Request(  # noqa: S310
                f"{OLLAMA_URL}/api/generate",
                data=json.dumps({"model": name, "keep_alive": 0}).encode(),
                headers={"Content-Type": "application/json"},
            )
            urllib.request.urlopen(req, timeout=30).read()  # noqa: S310
        except OSError:
            pass
