@echo off
rem E2 v2 pipeline (configs\e2_api_v2.yaml). main run (API): 80 frozen pairs, baseline vs full receipt. Log: external\logs\e2_v2_main.log (resumable: rerun to continue)
cd /d %~dp0..
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\e2_main.ps1
echo.
echo Done. Log: external\logs\e2_v2_main.log
pause
