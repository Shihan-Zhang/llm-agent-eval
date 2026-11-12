@echo off
rem E2 main supplement (API, new pipeline v2): 120 extra pairs, baseline vs full receipt; with main = 200 pairs. Log: external\logs\e2_v2_main_supp.log (resumable: rerun to continue)
cd /d %~dp0..
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\e2_main_supp.ps1
echo.
echo Done. Log: external\logs\e2_v2_main_supp.log
pause
