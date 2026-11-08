# E1-API 试跑:Qwen3.5-35B-A3B(硅基流动)按论文设置,C1–C4 各取 task_001 的 5 个模式 = 20 回合
# 目的:实测花费、思考长度、接口错误与超时,再决定全量 2,000 回合。用法:双击 scripts\run_e1api_pilot.bat
$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "common.ps1")
Start-Transcript -Path (Join-Path $Logs "e1api_pilot.log") -Append | Out-Null
if (-not $env:SILICONFLOW_API_KEY) { throw ".env 里的 SILICONFLOW_API_KEY 为空" }

Step "1. 版本记录"
Write-Host "llm-agent-eval: $(git -C $Root rev-parse --short HEAD)   ToolMaze: $(git -C $TM rev-parse --short HEAD)"
Write-Host "模型: Qwen/Qwen3.5-35B-A3B @ $env:SILICONFLOW_BASE_URL   日期: $(Get-Date -Format s)"

Step "2. 试跑 20 回合(C1–C4 各 5 个,并发 4)"
$Cfg = Join-Path $Root "eval\agent_error\configs\toolmaze_api_pilot.yaml"
$Runner = Join-Path $Root "eval\agent_error\toolmaze_run.py"
$t0 = Get-Date
foreach ($c in "c1","c2","c3","c4") {
  Write-Host "--- $c ---"
  & $Py $Runner --config $Cfg --task-category $c --limit 5
}
Write-Host ("总耗时 {0:N1} 分钟" -f ((Get-Date) - $t0).TotalMinutes)

Step "3. 用量与花费"
& $Py (Join-Path $Root "eval\agent_error\cost_report.py") (Join-Path $AE "results\e1api_pilot\calls.jsonl") --episodes 20 --scale 2000

Step "完成"
Stop-Transcript | Out-Null
