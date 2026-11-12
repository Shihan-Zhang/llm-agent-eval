# 知识库检索服务(kb_service):只监听 127.0.0.1,向量只用本地 Ollama bge-m3(规矩 5)。
# 先在 .env 里设置 KB_SERVICE_API_KEYS;启动前要有 rag/store(python -m rag.index 建好)且 Ollama 在运行。
# 用法:双击 scripts\run_kb_service.bat;请求日志 external\logs\kb_service.log。Ctrl+C 停止。
$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "common.ps1")
if (-not $env:KB_SERVICE_API_KEYS) { throw ".env 里的 KB_SERVICE_API_KEYS 为空" }
$VenvPy = Join-Path $Root "venv\Scripts\python.exe"
Set-Location $Root
& $VenvPy -m kb_service
if ($LASTEXITCODE -ne 0) { throw "kb_service 退出(退出码 $LASTEXITCODE)" }
