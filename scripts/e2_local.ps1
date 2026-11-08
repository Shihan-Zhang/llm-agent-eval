# E2 补充:本地 qwen3i-8k(num_ctx 8192,温度 0)在冻结清单 80 对(C3/C4 × P3/P4)上,基线 vs 完整回执。
# 试跑最大输入 4,900 token(预算 8,192 − 2,048 = 6,144);每个回合记录最大输入,超过预算的在汇总里标出。
# 预计约 65 分钟。可中断后重跑。用法:双击 scripts\run_e2_local.bat;日志 external\logs\e2_local.log。
$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "common.ps1")
Start-Transcript -Path (Join-Path $Logs "e2_local.log") -Append | Out-Null

Step "1. 版本记录"
VersionInfo

Step "2. 预热模型"
$body = @{ model = "qwen3i-8k"; messages = @(@{ role = "user"; content = "hi" }); max_tokens = 1 } | ConvertTo-Json -Depth 4
Invoke-RestMethod -Method Post -Uri http://localhost:11434/v1/chat/completions -ContentType "application/json" -Body $body | Out-Null
ollama ps

Step "3. 运行:80 对,基线 vs 完整回执(串行)"
$t0 = Get-Date
& $Py (Join-Path $Root "eval\agent_error\run_e2.py") --config (Join-Path $Root "eval\agent_error\configs\e2_local.yaml") --run main
if ($LASTEXITCODE -ne 0) { throw "run_e2 失败(退出码 $LASTEXITCODE);重新双击即可续跑" }
Write-Host ("本次耗时 {0:N1} 分钟" -f ((Get-Date) - $t0).TotalMinutes)
nvidia-smi --query-gpu=memory.used --format=csv,noheader

Step "4. 汇总"
& $Py (Join-Path $Root "eval\agent_error\summarize_e2.py") --config (Join-Path $Root "eval\agent_error\configs\e2_local.yaml") --run main
if ($LASTEXITCODE -ne 0) { throw "summarize_e2 失败(退出码 $LASTEXITCODE)" }

Step "完成"
Stop-Transcript | Out-Null
