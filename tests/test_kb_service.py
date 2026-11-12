"""kb_service:知识库检索 HTTP 服务。用假的向量函数,不调用 Ollama、不联网。"""
import json
import logging
import secrets
import threading
import urllib.error

import numpy as np
import pytest
from fastapi.testclient import TestClient

from kb_service.app import Settings, create_app, search_tool_schema
from kb_service.security import RateLimiter
from rag.retrieve import Retriever

KEY, OTHER = "test-key-1", "test-key-2"
CHUNKS = [
    {"id": "a", "source": "a.md", "text": "McNemar 检验只看不一致配对", "embed_text": "McNemar 检验 不一致 配对"},
    {"id": "b", "source": "b.md", "text": "BM25 是词频加权的检索", "embed_text": "BM25 词频 检索"},
    {"id": "c", "source": "c.md", "text": "RRF 按倒数排名融合", "embed_text": "RRF 倒数 排名 融合"},
]
EMB = np.eye(3, 4, dtype=np.float32)
SECRET_QUERY = "McNemar 检验 私密查询词"


class FakeClock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


def fake_embed(texts, model="bge-m3", cpu=False):
    out = np.zeros((len(texts), 4), dtype=np.float32)
    for i, t in enumerate(texts):
        out[i, 0 if "检验" in t else 1 if "检索" in t else 2] = 1.0
    return out


def make_client(rate=100, embed_fn=fake_embed, clock=None):
    retriever = Retriever(CHUNKS, EMB, {"embedding_model": "fake"}, embed_fn=embed_fn)
    settings = Settings(api_keys=frozenset({KEY, OTHER}), rate_per_min=rate, embed_timeout_s=5)
    return TestClient(create_app(retriever, settings, clock=clock or FakeClock()))


def post(client, path="/search", body=None, key=KEY, headers=None):
    headers = {**({"X-API-Key": key} if key else {}), **(headers or {})}
    return client.post(path, json={"query": "McNemar 检验", "k": 2, "mode": "dense"} if body is None else body,
                       headers=headers)


# ---------------------------------------------------------------- 正常请求
def test_health_needs_no_key():
    r = make_client().get("/health")
    assert r.status_code == 200 and r.json() == {"status": "ok", "chunks": 3}


def test_search_returns_ranked_hits_with_text_and_request_id():
    r = post(make_client())
    assert r.status_code == 200
    body = r.json()
    assert body["request_id"] == r.headers["X-Request-ID"]
    hits = body["results"]
    assert [h["chunk_id"] for h in hits] == ["a", "b"]
    assert hits[0] == {"rank": 1, "chunk_id": "a", "source": "a.md", "score": pytest.approx(1.0),
                       "text": "McNemar 检验只看不一致配对"}


def test_request_id_header_is_echoed():
    r = post(make_client(), headers={"X-API-Key": KEY, "X-Request-ID": "abc-123"}, key=None)
    assert r.headers["X-Request-ID"] == "abc-123" and r.json()["request_id"] == "abc-123"


def test_defaults_k5_hybrid():
    r = post(make_client(), body={"query": "BM25 检索"})
    assert r.status_code == 200 and r.json()["results"][0]["chunk_id"] == "b"


# ---------------------------------------------------------------- 参数校验
@pytest.mark.parametrize("body", [
    {},                                              # 缺 query
    {"query": ""}, {"query": "   "},                 # 空查询
    {"query": "x" * 501},                            # 过长
    {"query": "x", "k": 0}, {"query": "x", "k": 21},  # k 越界
    {"query": "x", "mode": "fuzzy"},                 # mode 不合法
    {"query": "x", "extra": 1},                      # 未知字段
])
def test_invalid_body_returns_422(body):
    assert post(make_client(), body=body).status_code == 422


# ---------------------------------------------------------------- 鉴权
@pytest.mark.parametrize("path", ["/search", "/search/stream"])
def test_missing_or_wrong_key_returns_401(path):
    c = make_client()
    assert post(c, path, key=None).status_code == 401
    assert post(c, path, key="wrong").status_code == 401
    assert post(c, path, key=None, body={}).status_code == 401     # 鉴权在参数校验之前


def test_invalid_keys_get_401_and_never_create_buckets():
    c = make_client()
    keys = {secrets.token_urlsafe(16) for _ in range(100)}
    assert len(keys) == 100 and not keys & {KEY, OTHER}
    assert all(post(c, key=k).status_code == 401 for k in keys)
    assert c.app.state.limiter.bucket_count == 0                    # 先鉴权后限流:无效 Key 不建桶
    assert post(c).status_code == 200 and c.app.state.limiter.bucket_count == 1


def test_settings_require_at_least_one_key():
    with pytest.raises(ValueError):
        Settings.from_env({"KB_SERVICE_API_KEYS": " , "})
    s = Settings.from_env({"KB_SERVICE_API_KEYS": "k1, k2", "KB_SERVICE_RATE_PER_MIN": "12"})
    assert s.api_keys == frozenset({"k1", "k2"}) and s.rate_per_min == 12


# ---------------------------------------------------------------- 限流
def test_rate_limit_returns_429_with_retry_after_and_recovers():
    clock = FakeClock()
    c = make_client(rate=2, clock=clock)
    assert [post(c).status_code for _ in range(3)] == [200, 200, 429]
    r = post(c)
    assert r.status_code == 429 and int(r.headers["Retry-After"]) >= 1
    assert post(c, key=OTHER).status_code == 200                   # 另一个 Key 不受影响
    clock.t += 30                                                  # 每分钟 2 个 → 30 秒补 1 个
    assert post(c).status_code == 200


def test_rate_30_per_min_burst_of_31_then_retry_at_10s():
    clock = FakeClock()
    c = make_client(rate=30, clock=clock)
    codes = [post(c) for _ in range(31)]                           # 时钟不动 = 0 秒内连发
    assert [r.status_code for r in codes[:30]] == [200] * 30
    retry_after = codes[30].headers["Retry-After"]
    print(f"\n第 31 个:{codes[30].status_code},Retry-After = {retry_after}")
    assert codes[30].status_code == 429 and retry_after == "2"     # 补 1 个需 60/30 = 2 秒
    clock.t += 10                                                  # 10 秒补 10 × 30/60 = 5 个
    r = post(c)
    print(f"第 10 秒再发:{r.status_code}")
    assert r.status_code == 200


def test_rate_limit_is_exact_under_20_concurrent_requests():
    c = make_client(rate=10)                                       # 假时钟不走:不会补充额度
    barrier, codes, lock = threading.Barrier(20), [], threading.Lock()

    def worker():
        barrier.wait()
        code = post(c).status_code
        with lock:
            codes.append(code)

    threads = [threading.Thread(target=worker) for _ in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sorted(codes) == [200] * 10 + [429] * 10


def test_rate_limiter_lock_under_contention():
    limiter = RateLimiter(rate_per_min=10, clock=FakeClock())
    barrier, results, lock = threading.Barrier(50), [], threading.Lock()

    def worker():
        barrier.wait()
        wait = limiter.try_acquire(KEY)
        with lock:
            results.append(wait)

    threads = [threading.Thread(target=worker) for _ in range(50)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sum(w == 0 for w in results) == 10 and all(w > 0 for w in results if w != 0)


# ---------------------------------------------------------------- 向量模型出错
@pytest.mark.parametrize("exc, code, phrase", [
    (ConnectionRefusedError("refused"), 503, "连不上"),
    (urllib.error.URLError(ConnectionRefusedError("refused")), 503, "连不上"),
    (TimeoutError("timed out"), 504, "超时"),
    (urllib.error.URLError(TimeoutError("timed out")), 504, "超时"),
])
def test_embed_errors_map_to_503_or_504(exc, code, phrase):
    def broken(texts, model="bge-m3", cpu=False):
        raise exc

    for path in ("/search", "/search/stream"):
        r = post(make_client(embed_fn=broken), path)
        assert r.status_code == code and phrase in r.json()["detail"] and "Ollama" in r.json()["detail"]


# ---------------------------------------------------------------- SSE
def parse_sse(text):
    events = []
    for block in text.split("\n\n"):
        if not block.strip():
            continue
        lines = block.split("\n")
        assert lines[0].startswith("event: ") and lines[1].startswith("data: ") and len(lines) == 2, block
        events.append((lines[0][len("event: "):], json.loads(lines[1][len("data: "):])))
    return events


def test_stream_sends_result_events_then_done():
    r = post(make_client(), "/search/stream", body={"query": "McNemar 检验", "k": 3, "mode": "dense"})
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream")
    assert r.text.endswith("\n\n")
    events = parse_sse(r.text)
    kinds = [e for e, _ in events]
    assert kinds == ["result"] * 3 + ["done"]
    assert [d["rank"] for _, d in events[:-1]] == [1, 2, 3] and events[0][1]["text"] == CHUNKS[0]["text"]
    assert events[-1][1] == {"request_id": r.headers["X-Request-ID"], "count": 3}


def test_stream_validates_before_streaming():
    assert post(make_client(), "/search/stream", body={"query": ""}).status_code == 422


# ---------------------------------------------------------------- 日志
def test_log_has_request_id_duration_count_but_no_query_or_text(caplog):
    caplog.set_level(logging.INFO, logger="kb_service")
    c = make_client()
    r = post(c, body={"query": SECRET_QUERY, "k": 2, "mode": "dense"})
    post(c, "/search/stream", body={"query": SECRET_QUERY, "k": 1, "mode": "dense"})
    recs = [json.loads(m) for m in caplog.messages]
    first = next(x for x in recs if x["request_id"] == r.headers["X-Request-ID"])
    assert first["path"] == "/search" and first["status"] == 200 and first["results"] == 2
    assert isinstance(first["duration_ms"], float) and first["duration_ms"] >= 0
    assert any(x["path"] == "/search/stream" and x["results"] == 1 for x in recs)
    assert SECRET_QUERY not in caplog.text and "私密" not in caplog.text
    assert all(ch["text"] not in caplog.text for ch in CHUNKS)


class ExplodingRetriever:
    chunks = CHUNKS

    def search_scored(self, query, k, mode):
        raise RuntimeError("boom")


@pytest.mark.parametrize("path", ["/search", "/search/stream"])
def test_unhandled_exception_is_still_logged_as_500(caplog, path):
    caplog.set_level(logging.INFO, logger="kb_service")
    settings = Settings(api_keys=frozenset({KEY}), rate_per_min=100, embed_timeout_s=5)
    c = TestClient(create_app(ExplodingRetriever(), settings, clock=FakeClock()), raise_server_exceptions=False)
    r = post(c, path, headers={"X-Request-ID": "boom-1"})
    assert r.status_code == 500
    recs = [json.loads(m) for m in caplog.messages if m.startswith("{")]
    line = next(x for x in recs if x.get("request_id") == "boom-1")
    assert line["path"] == path and line["status"] == 500 and line["results"] is None
    assert isinstance(line["duration_ms"], float)
    assert SECRET_QUERY not in caplog.text


# ---------------------------------------------------------------- 工具 schema
def _objects(schema, path="$"):
    if isinstance(schema, dict):
        if schema.get("type") == "object":
            yield path, schema
        for k, v in schema.items():
            yield from _objects(v, f"{path}.{k}")
    elif isinstance(schema, list):
        for i, v in enumerate(schema):
            yield from _objects(v, f"{path}[{i}]")


def test_tool_schema_has_no_bare_objects_and_matches_request_model():
    tool = search_tool_schema()
    assert tool["type"] == "function" and tool["function"]["name"] == "kb_search"
    params = tool["function"]["parameters"]
    objs = list(_objects(params))
    assert objs, "参数本身应是 object"
    for path, s in objs:                                          # E2 的教训:无属性 object 会被服务端压成 {}
        assert s.get("properties"), f"{path} 是没有 properties 的 object"
    assert set(params["properties"]) == {"query", "k", "mode"} and params["required"] == ["query"]
    assert params["properties"]["mode"]["enum"] == ["dense", "bm25", "hybrid"]
    assert "$ref" not in json.dumps(params) and "$defs" not in params
