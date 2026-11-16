"""S2.L1 practice tests. Each test names the K item it checks."""
import math

import pytest

from s2_l1_drills import (
    MODEL, CachedEmbedder, ModelMismatchError, StoredVector, cosine, dot, embed, normalize,
)

TEXTS = ["Can I get a refund?", "Refund status for order 4417", "My charger arrived broken"]


# ---------------- D1 · Call it ----------------
def test_d1_one_call_for_many_texts(client):  # K4: batch
    vectors = embed(TEXTS, client)
    assert len(client.calls) == 1, "send all texts in one call, not one call per text"
    assert client.calls[0]["input"] == TEXTS and client.calls[0]["model"] == MODEL
    assert len(vectors) == len(TEXTS)


def test_d1_order_and_fixed_length(client):  # K1: one fixed-length vector per text, in order
    vectors = embed(TEXTS, client)
    assert vectors == [client.vector_for(MODEL, t) for t in TEXTS]
    assert {len(v) for v in vectors} == {client.dim}


def test_d1_empty_list_makes_no_call(client):  # K4: don't pay for nothing
    assert embed([], client) == []
    assert client.calls == []


@pytest.mark.live
def test_d1_live_ollama():  # K1 + K2: real model, 768 numbers, length 1
    ollama = pytest.importorskip("ollama")
    vectors = embed(["Can I get a refund?", "Where is my money back?"], ollama.Client())
    assert len(vectors) == 2 and len(vectors[0]) == 768
    assert math.isclose(math.sqrt(sum(x * x for x in vectors[0])), 1.0, rel_tol=1e-3)


# ---------------- D2 · Harden it ----------------
def test_d2_second_call_is_free(client, tmp_path):  # K4: cache
    e = CachedEmbedder(client, tmp_path)
    first = e.embed(TEXTS)
    second = e.embed(TEXTS)
    assert second == first
    assert len(client.calls) == 1, "the second call should be served from the cache"


def test_d2_only_missing_texts_are_sent(client, tmp_path):  # K4: cache + batch
    e = CachedEmbedder(client, tmp_path)
    e.embed(TEXTS[:2])
    out = e.embed(TEXTS)
    assert client.calls[-1]["input"] == [TEXTS[2]], "send only the text that isn't cached, in one call"
    assert out == [client.vector_for(MODEL, t) for t in TEXTS]


def test_d2_cache_survives_restart(client, tmp_path):  # K4: re-indexing is free for unchanged text
    CachedEmbedder(client, tmp_path).embed(TEXTS)
    CachedEmbedder(client, tmp_path).embed(TEXTS)
    assert len(client.calls) == 1


def test_d2_other_model_is_a_cache_miss(client, tmp_path):  # K3: the model is part of the key
    CachedEmbedder(client, tmp_path, model=MODEL).embed(TEXTS)
    CachedEmbedder(client, tmp_path, model="text-embedding-3-small").embed(TEXTS)
    assert len(client.calls) == 2
    assert client.calls[1]["model"] == "text-embedding-3-small"


# ---------------- D3 · Break it ----------------
def v(*xs, model=MODEL):
    return StoredVector(model, tuple(xs))


def test_d3_cosine_compares_direction():  # K2
    assert math.isclose(cosine(v(1, 0), v(1, 0)), 1.0)
    assert math.isclose(cosine(v(1, 0), v(0, 1)), 0.0, abs_tol=1e-12)
    assert math.isclose(cosine(v(1, 0), v(-1, 0)), -1.0)
    assert math.isclose(cosine(v(1, 2), v(3, 6)), 1.0), "length must not change cosine"


def test_d3_normalize():  # K2
    n = normalize((3, 4))
    assert math.isclose(math.hypot(*n), 1.0) and math.isclose(n[0], 0.6) and math.isclose(n[1], 0.8)
    with pytest.raises(ValueError):
        normalize((0, 0))


def test_d3_dot_equals_cosine_after_normalize():  # K2
    a, b = (0.9, 0.2, 0.4), (2.0, 1.5, -0.3)
    raw = cosine(v(*a), v(*b))
    assert math.isclose(dot(v(*normalize(a)), v(*normalize(b))), raw)


def test_d3_long_vector_wins_dot_not_cosine():  # K2: the Lab's long-FAQ case
    q = v(math.cos(math.radians(70)), math.sin(math.radians(70)))
    refund = v(0.9 * math.cos(math.radians(74)), 0.9 * math.sin(math.radians(74)))
    faq = v(2.2 * math.cos(math.radians(98)), 2.2 * math.sin(math.radians(98)))
    assert dot(q, faq) > dot(q, refund)        # raw dot: the long FAQ wins
    assert cosine(q, refund) > cosine(q, faq)  # cosine: the refund answer wins


def test_d3_mixed_models_raise():  # K3
    with pytest.raises(ModelMismatchError):
        cosine(v(1, 0), v(1, 0, model="text-embedding-3-small"))
    with pytest.raises(ModelMismatchError):
        dot(v(1, 0), v(1, 0, model="text-embedding-3-small"))


def test_d3_mixed_dimensions_raise():  # K3
    with pytest.raises(ModelMismatchError):
        cosine(v(1, 0), v(1, 0, 0))
