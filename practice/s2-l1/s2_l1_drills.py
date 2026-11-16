"""S2.L1 practice: embeddings in application code.

Fill in the bodies marked TODO. Run:  uv run pytest practice/s2-l1 -q
Live check against Ollama (optional): uv run pytest practice/s2-l1 -q -m live
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

MODEL = "nomic-embed-text"


# ---------------------------------------------------------------------------
# D1 · Call it  (K1, K4)
# ---------------------------------------------------------------------------
def embed(texts: list[str], client, model: str = MODEL) -> list[list[float]]:
    """Embed every text with ONE call: client.embed(model=model, input=texts).

    - Return one vector per text, in the same order as `texts`.
    - The response works like a dict: response["embeddings"] is a list of vectors.
    - An empty list returns [] without calling the client.
    """
    if not texts:
    	return []
    response = client.embed(model=model, input=texts)
    return response["embeddings"]




# ---------------------------------------------------------------------------
# D2 · Harden it  (K4)
# ---------------------------------------------------------------------------
class CachedEmbedder:
    """Embed texts, never paying twice for the same (model, text).

    - Cache key: sha256 of model + "\\n" + text.
    - Only texts missing from the cache are sent, in one batched call (use embed() from D1).
    - The cache lives in `cache_dir` and survives a restart (a new CachedEmbedder on the same folder).
    - How you store it on disk is up to you.
    """

    def __init__(self, client, cache_dir: Path, model: str = MODEL):
        raise NotImplementedError  # TODO

    def embed(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError  # TODO


# ---------------------------------------------------------------------------
# D3 · Break it  (K2, K3)
# ---------------------------------------------------------------------------
class ModelMismatchError(ValueError):
    """Two vectors from different models (or of different lengths) can't be compared."""


@dataclass(frozen=True)
class StoredVector:
    """A vector plus the model that made it. Store vectors like this, never bare."""
    model: str
    values: tuple[float, ...]

    @property
    def dim(self) -> int:
        return len(self.values)


def normalize(values) -> tuple[float, ...]:
    """Same direction, length 1. Raise ValueError for an all-zero vector."""
    raise NotImplementedError  # TODO


def cosine(a: StoredVector, b: StoredVector) -> float:
    """Cosine similarity: a·b / (|a| × |b|).

    Raise ModelMismatchError if a.model != b.model or a.dim != b.dim.
    """
    raise NotImplementedError  # TODO


def dot(a: StoredVector, b: StoredVector) -> float:
    """Dot product a·b, with the same model and dimension check as cosine()."""
    raise NotImplementedError  # TODO
