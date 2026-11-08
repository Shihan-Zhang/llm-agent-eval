# 知识库 MCP 服务器:建环境 → 单元测试 → 导入向量到本地 Qdrant → 等价性评测(MCP vs 直接检索)
# 用法:双击 scripts\run_kb_mcp.bat;日志 external\logs\kb_mcp.log
$ErrorActionPreference = "Stop"
# 注意:PS 5.1 下 2>&1 会把原生程序的 stderr 变成错误记录,配合 Stop 会中断脚本(fastmcp 的日志写在 stderr),所以只管道 stdout
. (Join-Path $PSScriptRoot "common.ps1")
Start-Transcript -Path (Join-Path $Logs "kb_mcp.log") -Force | Out-Null
$Venv = Join-Path $Ext "venv-mcp"
$PyM  = Join-Path $Venv "Scripts\python.exe"
$env:QDRANT_LOCAL_PATH = "D:\ZSH\cache\qdrant"
$env:COLLECTION_NAME   = "notes"
$env:PYTHONPATH        = $Root

Step "1. 版本记录"
Write-Host "repo: $(git -C $Root rev-parse --short HEAD) ($(git -C $Root rev-parse --abbrev-ref HEAD))   日期: $(Get-Date -Format s)"
ollama --version
ollama list | Select-String "bge-m3"
if (-not (Test-Path (Join-Path $Root "rag\store\emb.npy"))) { throw "rag\store 里没有向量:先运行 python -m rag.ingest 和 python -m rag.index" }
Get-Content (Join-Path $Root "rag\store\manifest.json")

Step "2. 独立虚拟环境(mcp-server-qdrant 锁定的 pydantic 版本与项目环境冲突)"
if (-not (Test-Path $PyM)) { python -m venv $Venv }
& $PyM -m pip install -q --upgrade pip
& $PyM -m pip install -q -r (Join-Path $Root "requirements-mcp.txt")
& $PyM -m pip show mcp-server-qdrant fastmcp mcp qdrant-client | Select-String "^(Name|Version)"

Step "3. 单元测试(不需要 Ollama)"
Push-Location $Root
& $PyM -m pytest -q tests\test_kb_mcp.py -p no:warnings | Out-Host

Step "4. 导入:复用 rag\store 的块和 bge-m3 向量,写入 $env:QDRANT_LOCAL_PATH"
& $PyM -m kb_mcp.ingest --recreate | Out-Host

Step "5. 等价性评测:通过 MCP(stdio 子进程)vs 直接调用 retrieve.py"
$t0 = Get-Date
& $PyM eval\rag\eval_mcp_retrieval.py | Out-Host
Write-Host ("耗时 {0:N1} 秒" -f ((Get-Date) - $t0).TotalSeconds)
Pop-Location

Step "完成"
Stop-Transcript | Out-Null
