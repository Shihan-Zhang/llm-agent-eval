# AGENTS.md

给编码 Agent(Claude Code、Codex、Cursor、Deep Agents 等)的项目说明。人类读者请看 README。

## 项目是什么

大模型与 Agent 评测平台:复现公开基准(ToolMaze、BFCL、RGB、CRITICTOOL),做有统计依据、能自查判分错误的评测。
不自建虚构测试集;外部基准的代码和数据放 `external/`,只引用、不进仓库。

## 目录

| 路径 | 内容 |
|---|---|
| `agent.py`、`providers/`、`router/` | Agent 循环、模型接入、路由 Agent 的三个工具 |
| `eval/agent_error/` | ToolMaze 接入:配置、运行时补丁 `toolmaze_run.py`(不改上游源码)、成本报告 |
| `eval/fc/` | BFCL 函数调用评测 |
| `rag/`、`eval/rag/` | 中文 RAG(bge-m3 + BM25 + RRF)与检索评测 |
| `kb_mcp/` | 知识库 MCP 服务器(在 mcp-server-qdrant 0.8.1 上扩展) |
| `eval/results/` | 定稿报告;`predictions.md` 是事先写下的预测 |
| `scripts/` | Windows 一键脚本(`.bat` 调 `.ps1`),日志写 `external/logs/` |

## 工作规矩(必须遵守)

1. **每块工作从 main 开新分支,做完走 PR**,不直接推 main。
2. **跑实验前先把预测写进 `eval/results/predictions.md`**,结果出来再填判定;预测由人写,Agent 不代写。
3. **几个配置在同一题全错,先查判分器和管线,再怀疑模型**;结论要有轨迹或数据为证。
4. **README 和报告里的每个数字都必须能由仓库里的数据或公式复现**;改数字时写明出处和计算。
5. **个人数据(笔记 `data/notes/`、岗位表 `data/jobs/`)只给本地模型用**,不发到任何 API。
6. **密钥只在 `.env`**(已 gitignore),不打印、不提交;新增配置项同步写进 `.env.example`。
7. 优先在成熟的开源项目上扩展(子类、注入、运行时补丁),不 fork、不重写;上游的坑写进对应 README。

## 环境

- 机器是 Windows + RTX 3060 6 GB;大文件、缓存、虚拟环境一律放 D 盘。
- 两个虚拟环境:项目 `venv`(`requirements.txt`);知识库 MCP 用 `external/venv-mcp`(`requirements-mcp.txt`,
  因为上游锁定 pydantic < 2.12)。基准用 `external/venv-bench`。
- Ollama 本地模型:`qwen3i-8k`(固定 num_ctx 8192)、`bge-m3`。OpenAI 兼容接口不能逐请求传 num_ctx。
- PowerShell 5.1:`.ps1` 存成 UTF-8 带 BOM;`2>&1` 配合 `$ErrorActionPreference = "Stop"` 会因 stderr 中断脚本;
  Python 读文件要指定 encoding(或设 `PYTHONUTF8=1`)。

## 测试

```bash
python -m pytest -q                                  # 项目环境
python -m pytest -q tests/test_kb_mcp.py             # MCP 环境(external/venv-mcp)
```

测试不调用模型、不联网。改判分器、解析、检索时,先加能复现问题的测试再修。
CI(`.github/workflows/tests.yml`)在每个 PR 上跑这两组。

## 提交

提交信息用中文,写清做了什么和为什么;实验结果的提交附上版本记录(仓库提交、模型、日期)。
