@echo off
rem E2 run 3 (API): same 80 frozen tasks at P0, baseline vs full receipt. Log: external\logs\e2_p0.log (resumable: rerun to continue)
rem Write predictions in eval\results\predictions.md before starting.
cd /d %~dp0..
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\e2_p0.ps1
echo.
echo Done. Log: external\logs\e2_p0.log
pause
