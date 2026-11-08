# E2 试跑:API(Qwen3.5-35B-A3B,温度 0,关闭思考)与本地 qwen3i-8k 各 5 题 × 基线/回执两组,
# 题目取冻结清单之外的 C3/C4 P3/P4;实测 token 数、预计总耗时与花费写进日志。不跑正式实验。
# 用法:双击 scripts\run_e2_pilot.bat;日志 external\logs\e2_pilot.log。可中断后重跑(已完成的配对跳过)。
$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "common.ps1")
Start-Transcript -Path (Join-Path $Logs "e2_pilot.log") -Append | Out-Null
if (-not $env:SILICONFLOW_API_KEY) { throw ".env 里的 SILICONFLOW_API_KEY 为空" }

Step "1. 版本记录"
VersionInfo
Write-Host "API 模型: Qwen/Qwen3.5-35B-A3B @ $env:SILICONFLOW_BASE_URL   日期: $(Get-Date -Format s)"

Step "2. 契约注册表(不存在则构建;规格与清单哈希见 eval\agent_error\README.md)"
$Reg = Join-Path $AE "results\e2\contracts\gold\fold_0.json"
if (-not (Test-Path $Reg)) { & $Py (Join-Path $Root "eval\agent_error\contracts.py") build --source gold }
$Run = Join-Path $Root "eval\agent_error\run_e2.py"

Step "3. API 试跑 5 题 × 2 组,并估算正式运行 ①(80 对)"
$t0 = Get-Date
& $Py $Run --config (Join-Path $Root "eval\agent_error\configs\e2_api.yaml") --run pilot --estimate main
Write-Host ("API 试跑耗时 {0:N1} 分钟" -f ((Get-Date) - $t0).TotalMinutes)

Step "4. 本地 qwen3i-8k 试跑 5 题 × 2 组:测输入 token 是否装得下 8k 上下文"
$body = @{ model = "qwen3i-8k"; messages = @(@{ role = "user"; content = "hi" }); max_tokens = 1 } | ConvertTo-Json -Depth 4
Invoke-RestMethod -Method Post -Uri http://localhost:11434/v1/chat/completions -ContentType "application/json" -Body $body | Out-Null
$t0 = Get-Date
& $Py $Run --config (Join-Path $Root "eval\agent_error\configs\e2_local.yaml") --run pilot --estimate main
Write-Host ("本地试跑耗时 {0:N1} 分钟" -f ((Get-Date) - $t0).TotalMinutes)

Step "完成(正式运行需另行确认)"
Stop-Transcript | Out-Null
