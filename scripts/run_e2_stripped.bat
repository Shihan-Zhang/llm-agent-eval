@echo off
rem E2 v2 pipeline (configs\e2_api_v2.yaml). run 2 (API): 200 frozen pairs (80 main + 120 ext), full receipt vs receipt without recovery tools. Log: external\logs\e2_v2_stripped.log (resumable: rerun to continue)
rem Write predictions in eval\results\predictions.md before starting.
cd /d %~dp0..
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\e2_stripped.ps1
echo.
echo Done. Log: external\logs\e2_v2_stripped.log
pause
