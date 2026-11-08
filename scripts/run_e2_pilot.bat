@echo off
rem E2 pilot: API + local 4B, 5 tasks x 2 arms each. Log: external\logs\e2_pilot.log (resumable)
cd /d %~dp0..
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\e2_pilot.ps1
echo.
echo Done. Log: external\logs\e2_pilot.log
pause
