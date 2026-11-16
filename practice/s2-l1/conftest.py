"""Fixtures for S2.L1 practice: a fake embedding client, so tests run offline in milliseconds."""
import hashlib

import pytest


class FakeEmbedClient:
    """Stands in for ollama.Client(). Records every call; returns deterministic fake vectors."""

    def __init__(self, dim: int = 8):
        self.dim = dim
        self.calls: list[dict] = []

    def embed(self, model, input):
        texts = [input] if isinstance(input, str) else list(input)
        self.calls.append({"model": model, "input": texts})
        return {"model": model, "embeddings": [self.vector_for(model, t) for t in texts]}

    def vector_for(self, model: str, text: str) -> list[float]:
        digest = hashlib.sha256(f"{model}|{text}".encode()).digest()
        return [b / 255 for b in digest[: self.dim]]


@pytest.fixture
def client():
    return FakeEmbedClient()


def pytest_configure(config):
    config.addinivalue_line("markers", "live: calls a running Ollama (run with -m live)")


def pytest_collection_modifyitems(config, items):
    if "live" in (config.option.markexpr or ""):
        return
    skip = pytest.mark.skip(reason="live test: run with -m live")
    for item in items:
        if "live" in item.keywords:
            item.add_marker(skip)
