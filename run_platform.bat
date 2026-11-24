@echo off
title VoltEcho AI Platform Launcher
echo ========================================================
echo        VoltEcho: Acoustic Cyber-Physical AI Platform
echo ========================================================
echo.
echo Activating Python virtual environment...
cd /d "%~dp0"
if exist ".venv\Scripts\activate.bat" (
    call .venv\Scripts\activate.bat
) else (
    echo Warning: .venv not found, using global python...
)

echo Starting FastAPI backend and Web UI on http://127.0.0.1:8000 ...
echo Press Ctrl+C in this window to stop the server.
echo.

start "" "http://127.0.0.1:8000"
python -m uvicorn web.backend.main:app --host 127.0.0.1 --port 8000
pause
