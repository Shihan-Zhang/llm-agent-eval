# E2 ③ 新管线(configs/e2_api_v2.yaml):API 在冻结清单同 80 道题的 P0(无故障)上,基线 vs 完整回执。
# 两组都在 external\agent_error\results\e2\api_v2\p0 里重新跑;两组按配对随机交错(固定种子)。
# 按旧管线 ③ 实测约 4.4 小时、约 $2.6(80 对)。可中断后重跑。在 ① 之后跑。
# 用法:双击 scripts\run_e2_p0.bat;日志 external\logs\e2_v2_p0.log。
$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "common.ps1")
Start-Transcript -Path (Join-Path $Logs "e2_v2_p0.log") -Append | Out-Null
if (-not $env:SILICONFLOW_API_KEY) { throw ".env 里的 SILICONFLOW_API_KEY 为空" }

Step "1. 版本记录"
VersionInfo
Write-Host "新管线 e2_api_v2.yaml;API 模型: Qwen/Qwen3.5-35B-A3B @ $env:SILICONFLOW_BASE_URL   日期: $(Get-Date -Format s)"

Step "2. 正式运行 ③:80 对 P0,基线 vs 完整回执"
$t0 = Get-Date
& $Py (Join-Path $Root "eval\agent_error\run_e2.py") --config (Join-Path $Root "eval\agent_error\configs\e2_api_v2.yaml") --run p0
if ($LASTEXITCODE -ne 0) { throw "run_e2 失败(退出码 $LASTEXITCODE);重新双击即可续跑" }
Write-Host ("本次耗时 {0:N1} 分钟" -f ((Get-Date) - $t0).TotalMinutes)

Step "3. 汇总"
& $Py (Join-Path $Root "eval\agent_error\summarize_e2.py") --config (Join-Path $Root "eval\agent_error\configs\e2_api_v2.yaml") --run p0
if ($LASTEXITCODE -ne 0) { throw "summarize_e2 失败(退出码 $LASTEXITCODE)" }

Step "完成"
Stop-Transcript | Out-Null