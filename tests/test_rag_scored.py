"""rag.retrieve:带分数的检索(search_scored)与可注入的向量函数;rag.index.embed 的超时参数。不调用 Ollama。"""
import numpy as np
import pytest

import rag.index as IDX
from rag.retrieve import Retriever, rrf, rrf_scores

CHUNKS = [
    {"id": "a", "source": "a.md", "text": "McNemar 检验只看不一致配对", "embed_text": "McNemar 检验 不一致 配对"},
    {"id": "b", "source": "b.md", "text": "BM25 是词频加权的检索", "embed_text": "BM25 词频 检索"},
    {"id": "c", "source": "c.md", "text": "RRF 按倒数排名融合", "embed_text": "RRF 倒数 排名 融合"},
]
EMB = np.eye(3, 4, dtype=np.float32)                  # 三个块各占一维


def fake_embed(texts, model="bge-m3", cpu=False):
    """查询里含「检验」→ 第 0 维,含「检索」→ 第 1 维,否则第 2 维。"""
    out = np.zeros((len(texts), 4), dtype=np.float32)
    for i, t in enumerate(texts):
        out[i, 0 if "检验" in t else 1 if "检索" in t else 2] = 1.0
    return out


def retriever():
    return Retriever(CHUNKS, EMB, {"embedding_model": "fake"}, embed_fn=fake_embed)


def test_dense_scores_use_injected_embed_fn():
    hits = retriever().search_scored("McNemar 检验", k=2, mode="dense")
    assert [(c["id"], s) for c, s in hits] == [("a", pytest.approx(1.0)), ("b", pytest.approx(0.0))]


def test_bm25_scores_are_positive_and_sorted():
    hits = retriever().search_scored("BM25 检索", k=3, mode="bm25")
    assert hits and hits[0][0]["id"] == "b"
    scores = [s for _, s in hits]
    assert all(s > 0 for s in scores) and scores == sorted(scores, reverse=True)


def test_hybrid_scores_are_rrf_scores():
    r = retriever()
    hits = r.search_scored("BM25 检索", k=3, mode="hybrid")
    from rag.retrieve import top_k
    expected = rrf_scores([top_k(r.dense_scores("BM25 检索"), 30), r.bm25_top("BM25 检索", 30)])
    assert [s for _, s in hits] == pytest.approx([expected[CHUNKS.index(c)] for c, _ in hits])
    assert hits[0][0]["id"] == "b"


@pytest.mark.parametrize("mode", ["dense", "bm25", "hybrid"])
def test_search_equals_search_scored_without_scores(mode):
    r = retriever()
    assert r.search("BM25 检索", 3, mode) == [c for c, _ in r.search_scored("BM25 检索", 3, mode)]


def test_rrf_order_matches_rrf_scores():
    lists = [[0, 1, 2], [2, 0, 3]]
    s = rrf_scores(lists)
    assert rrf(lists) == sorted(s, key=lambda i: (-s[i], i))
    assert s[0] == pytest.approx(1 / 61 + 1 / 62)


def test_embed_passes_timeout_to_http(monkeypatch):
    seen = {}

    def fake_post(path, body, timeout=600):
        seen["timeout"] = timeout
        return {"embeddings": [[1.0, 0.0]]}

    monkeypatch.setattr(IDX, "_post", fake_post)
    IDX.embed(["x"], timeout=7)
    assert seen["timeout"] == 7
    IDX.embed(["x"])
    assert seen["timeout"] == 600                       # 默认值不变(建索引用)
