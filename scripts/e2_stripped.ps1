# E2 ②:API 在冻结清单 80 对(C3/C4 × P3/P4)上,完整回执 vs 去掉备用工具列表的回执——增益是否来自备用工具列表。
# 两组都在本运行目录里重新跑(external\agent_error\results\e2\api\stripped),不复用 ① 的完整回执组;两组按配对随机交错(固定种子)。
# 按 ① 实测折算约 4.4 小时、约 $2.6(80 对)。可中断后重跑:已完成的配对跳过,API 出错的配对整对重跑。
# 启动前先在 eval\results\predictions.md 写好预测。用法:双击 scripts\run_e2_stripped.bat;日志 external\logs\e2_stripped.log。
$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "common.ps1")
Start-Transcript -Path (Join-Path $Logs "e2_stripped.log") -Append | Out-Null
if (-not $env:SILICONFLOW_API_KEY) { throw ".env 里的 SILICONFLOW_API_KEY 为空" }

Step "1. 版本记录"
VersionInfo
Write-Host "API 模型: Qwen/Qwen3.5-35B-A3B @ $env:SILICONFLOW_BASE_URL   日期: $(Get-Date -Format s)"

Step "2. 正式运行 ②:80 对,完整回执 vs 去掉备用工具列表"
$t0 = Get-Date
& $Py (Join-Path $Root "eval\agent_error\run_e2.py") --config (Join-Path $Root "eval\agent_error\configs\e2_api.yaml") --run stripped
if ($LASTEXITCODE -ne 0) { throw "run_e2 失败(退出码 $LASTEXITCODE);重新双击即可续跑" }
Write-Host ("本次耗时 {0:N1} 分钟" -f ((Get-Date) - $t0).TotalMinutes)

Step "3. 汇总"
& $Py (Join-Path $Root "eval\agent_error\summarize_e2.py") --config (Join-Path $Root "eval\agent_error\configs\e2_api.yaml") --run stripped
if ($LASTEXITCODE -ne 0) { throw "summarize_e2 失败(退出码 $LASTEXITCODE)" }

Step "完成"
Stop-Transcript | Out-Null