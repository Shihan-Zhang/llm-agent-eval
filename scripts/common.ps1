# 各实验脚本共用:读取仓库根目录 .env、把缓存放 D 盘、Python 用 UTF-8
$Root = Split-Path -Parent $PSScriptRoot
$EnvFile = Join-Path $Root ".env"
if (Test-Path $EnvFile) {
  Get-Content $EnvFile -Encoding UTF8 | Where-Object { $_ -match '^\s*[A-Za-z_][A-Za-z0-9_]*\s*=' } | ForEach-Object {
    $k, $v = $_ -split '=', 2
    $k = $k.Trim(); $v = $v.Trim().Trim('"')
    if ($v) { Set-Item -Path "Env:$k" -Value $v }
  }
} else { Write-Warning "没有找到 .env:复制 .env.example 为 .env 并填写" }
$env:PIP_CACHE_DIR = "D:\ZSH\cache\pip"
$env:HF_HOME       = "D:\ZSH\cache\hf"
$env:PYTHONUTF8    = "1"      # ToolMaze 用 open(f, 'r') 读 JSON 不指定编码;中文 Windows 默认 GBK
$Ext  = Join-Path $Root "external"
$AE   = Join-Path $Ext "agent_error"
$TM   = Join-Path $AE "toolmaze"
$Py   = Join-Path $Ext "venv-bench\Scripts\python.exe"
$Logs = Join-Path $Ext "logs"
New-Item -ItemType Directory -Force $AE, $Logs | Out-Null
function Step($name) { Write-Host "`n===== $name =====" }
function VersionInfo {
  # 版本记录:两个仓库的提交、Ollama 版本、模型摘要、显卡状态,让结果能追溯到是哪次运行
  Write-Host "llm-agent-eval: $(git -C $Root rev-parse --short HEAD) ($(git -C $Root rev-parse --abbrev-ref HEAD))"
  Write-Host "ToolMaze:             $(git -C $TM rev-parse --short HEAD)"
  ollama --version
  ollama list | Select-String "qwen3i-8k"
  nvidia-smi --query-gpu=name,memory.total,memory.used,driver_version --format=csv,noheader
}
