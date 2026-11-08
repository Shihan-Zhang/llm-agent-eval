@echo off
rem E2 local run (qwen3i-8k): 80 frozen pairs, baseline vs full receipt. Log: external\logs\e2_local.log (resumable)
cd /d %~dp0..
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\e2_local.ps1
echo.
echo Done. Log: external\logs\e2_local.log
pause
