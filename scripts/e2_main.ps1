# E2 ① 新管线(configs/e2_api_v2.yaml:无属性 object 参数加 additionalProperties: true,结果写 e2\api_v2):
# API(Qwen3.5-35B-A3B,温度 0,关闭思考)在冻结清单 80 对(C3/C4 × P3/P4)上,基线 vs 完整回执。
# 按旧管线 ① 实测约 4.9 小时、约 $2.83 以内。可中断后重跑:已完成的配对跳过,API 出错的配对整对重跑。
# 启动前先在 eval\results\predictions.md 写好新管线的预测。按 ① → ③ → ② 的顺序跑。
# 用法:双击 scripts\run_e2_main.bat;日志 external\logs\e2_v2_main.log。
$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "common.ps1")
Start-Transcript -Path (Join-Path $Logs "e2_v2_main.log") -Append | Out-Null
if (-not $env:SILICONFLOW_API_KEY) { throw ".env 里的 SILICONFLOW_API_KEY 为空" }
& $Py (Join-Path $Root "eval\agent_error\check_prediction.py") (Join-Path $Root "eval\results\predictions.md") "① v2"
if ($LASTEXITCODE -ne 0) { throw "predictions.md 里没有 ① v2 的预测,拒绝启动" }

Step "1. 版本记录"
VersionInfo
Write-Host "新管线 e2_api_v2.yaml;API 模型: Qwen/Qwen3.5-35B-A3B @ $env:SILICONFLOW_BASE_URL   日期: $(Get-Date -Format s)"

Step "2. 正式运行 ①:80 对,基线 vs 完整回执"
$t0 = Get-Date
& $Py (Join-Path $Root "eval\agent_error\run_e2.py") --config (Join-Path $Root "eval\agent_error\configs\e2_api_v2.yaml") --run main
if ($LASTEXITCODE -ne 0) { throw "run_e2 失败(退出码 $LASTEXITCODE);重新双击即可续跑" }
Write-Host ("本次耗时 {0:N1} 分钟" -f ((Get-Date) - $t0).TotalMinutes)

Step "3. 汇总"
& $Py (Join-Path $Root "eval\agent_error\summarize_e2.py") --config (Join-Path $Root "eval\agent_error\configs\e2_api_v2.yaml") --run main
if ($LASTEXITCODE -ne 0) { throw "summarize_e2 失败(退出码 $LASTEXITCODE)" }

Step "完成"
Stop-Transcript | Out-Null
