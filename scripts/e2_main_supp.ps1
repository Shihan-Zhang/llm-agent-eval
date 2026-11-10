# E2 ① 新管线补充(configs/e2_api_v2.yaml 的 main_supp):补充清单 120 对(C3/C4 × P3/P4 各 30),基线 vs 完整回执,
# 与 main 的 80 对合起来 200 对。结果单独写 e2\api_v2\main_supp,main 的结果不动。
# 按 ① v2 main 实测外推(80 对 210.6 分钟、$2.03):约 5.3 小时、约 $3.05。可中断后重跑:已完成的配对跳过。
# 启动前在 eval\results\predictions.md 写好「① v2 补充」的预测行,否则拒绝启动。
# 用法:双击 scripts\run_e2_main_supp.bat;日志 external\logs\e2_v2_main_supp.log。
$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "common.ps1")
Start-Transcript -Path (Join-Path $Logs "e2_v2_main_supp.log") -Append | Out-Null
if (-not $env:SILICONFLOW_API_KEY) { throw ".env 里的 SILICONFLOW_API_KEY 为空" }
& $Py (Join-Path $Root "eval\agent_error\check_prediction.py") (Join-Path $Root "eval\results\predictions.md") "① v2 补充"
if ($LASTEXITCODE -ne 0) { throw "predictions.md 里没有 ① v2 补充 的预测,拒绝启动" }

Step "1. 版本记录"
VersionInfo
Write-Host "新管线 e2_api_v2.yaml main_supp;API 模型: Qwen/Qwen3.5-35B-A3B @ $env:SILICONFLOW_BASE_URL   日期: $(Get-Date -Format s)"

Step "2. 正式运行 ① 补充:120 对,基线 vs 完整回执"
$t0 = Get-Date
& $Py (Join-Path $Root "eval\agent_error\run_e2.py") --config (Join-Path $Root "eval\agent_error\configs\e2_api_v2.yaml") --run main_supp
if ($LASTEXITCODE -ne 0) { throw "run_e2 失败(退出码 $LASTEXITCODE);重新双击即可续跑" }
Write-Host ("本次耗时 {0:N1} 分钟" -f ((Get-Date) - $t0).TotalMinutes)

Step "3. 汇总(只含这 120 对)"
& $Py (Join-Path $Root "eval\agent_error\summarize_e2.py") --config (Join-Path $Root "eval\agent_error\configs\e2_api_v2.yaml") --run main_supp
if ($LASTEXITCODE -ne 0) { throw "summarize_e2 失败(退出码 $LASTEXITCODE)" }

Step "完成"
Stop-Transcript | Out-Null
