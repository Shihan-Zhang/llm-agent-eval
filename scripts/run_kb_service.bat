@echo off
rem Knowledge-base search service (read-only, 127.0.0.1 only, local Ollama embeddings). Request log: external\logs\kb_service.log
cd /d %~dp0..
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\kb_service.ps1
echo.
pause
