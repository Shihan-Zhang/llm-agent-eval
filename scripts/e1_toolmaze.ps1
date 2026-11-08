# E1:按官方设置复现 ToolMaze C1 —— 100 题 × P0–P4 = 500 回合,本地 qwen3i-8k,串行
# 用法:双击 scripts\run_e1.bat。可中断后重跑:已有推理结果的回合会直接跳过(ToolMaze 自带续跑)。
$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "common.ps1")
Start-Transcript -Path (Join-Path $Logs "e1.log") -Append | Out-Null

Step "1. 版本记录"
VersionInfo
$Cfg = Join-Path $Root "eval\agent_error\configs\toolmaze_e1.yaml"

Step "2. 预热模型(避免首题计入约 15 s 的加载时间)"
$body = @{ model = "qwen3i-8k"; messages = @(@{ role = "user"; content = "hi" }); max_tokens = 1 } | ConvertTo-Json -Depth 4
Invoke-RestMethod -Method Post -Uri http://localhost:11434/v1/chat/completions -ContentType "application/json" -Body $body | Out-Null
ollama ps

Step "3. 运行 500 回合(串行)"
Push-Location $TM
$t0 = Get-Date
& $Py evaluation\scripts\run_eval.py --config $Cfg --no-parallel
Write-Host ("总耗时 {0:N1} 分钟" -f ((Get-Date) - $t0).TotalMinutes)
Pop-Location
nvidia-smi --query-gpu=memory.used --format=csv,noheader

Step "完成"
Stop-Transcript | Out-Null
