@echo off
rem E2 run 2 (API): 80 frozen pairs, full receipt vs receipt without recovery tools. Log: external\logs\e2_stripped.log (resumable: rerun to continue)
rem Write predictions in eval\results\predictions.md before starting.
cd /d %~dp0..
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\e2_stripped.ps1
echo.
echo Done. Log: external\logs\e2_stripped.log
pause
