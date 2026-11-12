# kb_service:知识库检索 HTTP 服务

把 `rag/` 的检索(bge-m3 向量 + BM25 + RRF)做成只读的 HTTP 服务。FastAPI,只监听 `127.0.0.1`。

## 谁能调用(规矩 5)

索引来自个人笔记 `data/notes/`,返回结果里带**完整块正文**(`text`)。所以:

- **调用方只能是本机上的本地模型**(如 Ollama 的 qwen3i-8k 走的 Agent)。不要把结果转给任何云端 API,
  也不要让云端模型的 Agent 调这个服务;
- 查询向量只用本地 Ollama 的 bge-m3,服务本身不调用任何外部 API;
- 只监听 `127.0.0.1`,不提供对外开放的选项;
- 请求日志只记 request_id、路径、状态码、耗时、结果数,**不记查询和正文**。

## 启动

1. `.env` 里设置 `KB_SERVICE_API_KEYS`(见 `.env.example`,为空时拒绝启动);
2. 有建好的索引(`python -m rag.index`)且 Ollama 在运行;
3. 双击 `scripts\run_kb_service.bat`,请求日志写 `external\logs\kb_service.log`。

## 接口

| 接口 | 鉴权 | 说明 |
|---|---|---|
| `GET /health` | 否 | `{"status": "ok", "chunks": N}` |
| `POST /search` | `X-API-Key` | 请求 `{"query", "k", "mode"}`,返回 `{"request_id", "results": [{rank, chunk_id, source, score, text}]}` |
| `POST /search/stream` | `X-API-Key` | 参数同上,`text/event-stream`:每个结果一个 `event: result`,最后 `event: done`(`{request_id, count}`) |

参数:`query` 1–500 字、不能只有空白;`k` 1–20,默认 5;`mode` 为 `dense` / `bm25` / `hybrid`(默认),
不接受其他字段。分数口径随 mode:dense = 余弦相似度,bm25 = BM25 分数,hybrid = RRF 分数。
请求头 `X-Request-ID`(字母数字和 `._-`,≤ 64 字符)会被沿用,否则生成一个,都在响应头里返回。

| 状态码 | 含义 |
|---|---|
| 401 | 没带或带错 `X-API-Key`(两种情况不区分;鉴权在参数校验之前) |
| 422 | 参数不合法 |
| 429 | 超过该 Key 的限额,`Retry-After` 给出秒数 |
| 503 | 连不上 Ollama,或 Ollama 返回错误(如没拉取 bge-m3) |
| 504 | Ollama 在 `KB_SERVICE_EMBED_TIMEOUT_S` 秒内没有返回 |

函数调用工具定义:`kb_service.app.search_tool_schema()` 从请求模型导出,每个 object 都带 `properties`
(E2 里只写 `type: object` 的参数会被服务端压成 `{}`,测试保证这里没有这种参数)。

## 设计说明

- **路由用 `def` 而不是 `async def`**:查询向量化(对 Ollama 的 HTTP 调用)和 numpy / BM25 打分都是阻塞调用。
  `def` 路由由 FastAPI 放进线程池执行;写成 `async def` 会在事件循环里阻塞,一个慢请求就会挡住所有请求
  (包括 `/health`)。
- **限流**:每个 Key 一个令牌桶,容量 = 每分钟限额(允许的突发),按秒匀速补充;桶的读写在一把
  `threading.Lock` 里(线程池里的路由会并发调用)。测试:20 个线程同时请求、限额 10,恰好 10 个 200、10 个 429。
  状态只在内存里,单进程够用,重启清零。不复用 `eval/agent_error/ratelimit.py`:那是调用方的限速器
  (额度不够就 sleep 等待,同时管 token 和请求两种预算),服务端需要立刻拒绝并返回 `Retry-After`;
  它也不在包里,服务不应依赖实验脚本目录。
- **SSE 先算完再推送**:检索一次就出全部结果,出错时照常返回 503 / 504,不会推到一半断掉。
  手写事件格式(`event` 行 + 单行 JSON 的 `data` 行 + 空行),不另加依赖。
