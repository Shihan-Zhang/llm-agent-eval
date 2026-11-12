"""知识库检索 HTTP 服务(只读)。create_app 注入检索器、配置和时钟,测试用假的向量函数。

路由都写成 def 而不是 async def:查询向量化(Ollama HTTP 调用)和 numpy / BM25 打分都是阻塞调用,
def 路由由 FastAPI 放进线程池执行,不会卡住事件循环;写成 async def 会让一个慢请求挡住所有请求。

规矩 5:索引来自个人笔记,向量只用本地 Ollama;服务只给本机的本地模型调用(见 kb_service/README.md)。
"""
from __future__ import annotations

import json
import logging
import math
import re
import time
import urllib.error
import uuid
from dataclasses import dataclass
from typing import Callable, Iterator, Literal, Protocol

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator

from kb_service.security import RateLimiter, key_matches

log = logging.getLogger("kb_service")
MAX_QUERY_CHARS, MAX_K, DEFAULT_K = 500, 20, 5
DEFAULT_RATE_PER_MIN, DEFAULT_EMBED_TIMEOUT_S = 30, 30.0
REQUEST_ID = re.compile(r"^[A-Za-z0-9._-]{1,64}$")     # 调用方给的 X-Request-ID 只接受这种格式,防止日志注入


# ---------------------------------------------------------------- 配置
@dataclass(frozen=True)
class Settings:
    api_keys: frozenset[str]
    rate_per_min: int = DEFAULT_RATE_PER_MIN
    embed_timeout_s: float = DEFAULT_EMBED_TIMEOUT_S

    def __post_init__(self):
        if not self.api_keys:
            raise ValueError("没有配置 API Key:在 .env 里设置 KB_SERVICE_API_KEYS(逗号分隔)")
        if self.rate_per_min < 1 or self.embed_timeout_s <= 0:
            raise ValueError("KB_SERVICE_RATE_PER_MIN 必须 ≥ 1,KB_SERVICE_EMBED_TIMEOUT_S 必须 > 0")

    @classmethod
    def from_env(cls, env: dict[str, str]) -> "Settings":
        keys = frozenset(k.strip() for k in env.get("KB_SERVICE_API_KEYS", "").split(",") if k.strip())
        return cls(keys, int(env.get("KB_SERVICE_RATE_PER_MIN") or DEFAULT_RATE_PER_MIN),
                   float(env.get("KB_SERVICE_EMBED_TIMEOUT_S") or DEFAULT_EMBED_TIMEOUT_S))


class ScoredRetriever(Protocol):
    chunks: list[dict]

    def search_scored(self, query: str, k: int, mode: str) -> list[tuple[dict, float]]: ...


# ---------------------------------------------------------------- 请求与响应
class SearchRequest(BaseModel):
    """在个人知识库里检索,返回最相关的块。"""
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1, max_length=MAX_QUERY_CHARS, description="检索问题")
    k: int = Field(DEFAULT_K, ge=1, le=MAX_K, description="返回的块数")
    mode: Literal["dense", "bm25", "hybrid"] = Field("hybrid", description="dense 向量 / bm25 关键词 / hybrid 两者 RRF 融合")

    @field_validator("query")
    @classmethod
    def not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("query 不能只有空白")
        return v


class SearchHit(BaseModel):
    rank: int
    chunk_id: str
    source: str
    score: float
    text: str


class SearchResponse(BaseModel):
    request_id: str
    results: list[SearchHit]


def _strip_titles(schema: object) -> object:
    if isinstance(schema, dict):
        return {k: _strip_titles(v) for k, v in schema.items() if k != "title"}
    if isinstance(schema, list):
        return [_strip_titles(v) for v in schema]
    return schema


def search_tool_schema() -> dict:
    """/search 的请求模型 → OpenAI 格式的函数调用工具定义。每个 object 都带 properties(E2 里无属性 object 被服务端压成 {})。"""
    params = _strip_titles(SearchRequest.model_json_schema())
    description = params.pop("description", SearchRequest.__doc__)
    return {"type": "function", "function": {"name": "kb_search", "description": description, "parameters": params}}


# ---------------------------------------------------------------- 向量模型出错
def embed_error(exc: BaseException, timeout_s: float) -> HTTPException | None:
    """连不上 → 503,超时 → 504;其他异常不在这里处理(返回 None,按 500 走)。"""
    inner = exc.reason if isinstance(exc, urllib.error.URLError) else None
    if isinstance(exc, TimeoutError) or isinstance(inner, TimeoutError):
        return HTTPException(504, f"向量模型(Ollama)超时:{timeout_s:g} 秒内没有返回。可能正在加载 bge-m3 或显存不足,稍后重试。")
    if isinstance(exc, urllib.error.HTTPError):
        return HTTPException(503, f"向量模型(Ollama)返回 HTTP {exc.code}:确认已执行 ollama pull bge-m3。")
    if isinstance(exc, (ConnectionError, urllib.error.URLError)):
        return HTTPException(503, f"连不上向量模型(Ollama,{type(inner or exc).__name__}):确认 Ollama 已启动(默认 localhost:11434)。")
    return None


# ---------------------------------------------------------------- 应用
def create_app(retriever: ScoredRetriever, settings: Settings, clock: Callable[[], float] = time.monotonic) -> FastAPI:
    app = FastAPI(title="kb_service", description="个人知识库检索(只读,只给本机的本地模型调用)")
    limiter = app.state.limiter = RateLimiter(settings.rate_per_min, clock)

    @app.middleware("http")
    async def request_log(request: Request, call_next):
        given = request.headers.get("X-Request-ID", "")
        request.state.request_id = given if REQUEST_ID.match(given) else uuid.uuid4().hex
        request.state.result_count = None
        t0 = time.perf_counter()
        status = 500                                   # 路由抛出未处理异常时按 500 记,异常照常向外抛
        try:
            response = await call_next(request)
            status = response.status_code
            response.headers["X-Request-ID"] = request.state.request_id
        finally:
            log.info(json.dumps({"request_id": request.state.request_id, "path": request.url.path,
                                 "status": status,
                                 "duration_ms": round((time.perf_counter() - t0) * 1000, 2),
                                 "results": request.state.result_count}, ensure_ascii=False))
        return response

    def authorize(x_api_key: str | None = Header(None)) -> str:
        if not key_matches(x_api_key, settings.api_keys):
            raise HTTPException(401, "缺少或无效的 API Key(请求头 X-API-Key)", headers={"WWW-Authenticate": "ApiKey"})
        wait = limiter.try_acquire(x_api_key)
        if wait > 0:
            raise HTTPException(429, f"请求过多:每个 Key 每分钟 {settings.rate_per_min} 次",
                                headers={"Retry-After": str(math.ceil(wait))})
        return x_api_key

    def run_search(request: Request, body: SearchRequest) -> list[SearchHit]:
        try:
            scored = retriever.search_scored(body.query, body.k, body.mode)
        except Exception as exc:
            mapped = embed_error(exc, settings.embed_timeout_s)
            if mapped is None:
                raise
            log.warning(json.dumps({"request_id": request.state.request_id, "embed_error": type(exc).__name__,
                                    "status": mapped.status_code}))
            raise mapped from exc
        hits = [SearchHit(rank=i, chunk_id=c["id"], source=c["source"], score=s, text=c["text"])
                for i, (c, s) in enumerate(scored, start=1)]
        request.state.result_count = len(hits)
        return hits

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "chunks": len(retriever.chunks)}

    @app.post("/search", response_model=SearchResponse)
    def search(request: Request, body: SearchRequest, _key: str = Depends(authorize)) -> SearchResponse:
        return SearchResponse(request_id=request.state.request_id, results=run_search(request, body))

    @app.post("/search/stream")
    def search_stream(request: Request, body: SearchRequest, _key: str = Depends(authorize)) -> StreamingResponse:
        hits = run_search(request, body)            # 先算完:出错时照常返回 503 / 504,不会推到一半
        rid = request.state.request_id

        def events() -> Iterator[str]:
            for h in hits:
                yield sse("result", h.model_dump())
            yield sse("done", {"request_id": rid, "count": len(hits)})

        return StreamingResponse(events(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})

    return app


def sse(event: str, data: dict) -> str:
    """一个 SSE 事件:event 行 + data 行(单行 JSON)+ 空行。"""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"
