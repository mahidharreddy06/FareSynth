@echo off
title FareSynth AI - Web Application
echo =======================================================================
echo          Starting FareSynth AI Web Application & Backend...
echo =======================================================================
echo.
cd /d "%~dp0"

echo Launching browser at http://127.0.0.1:8000 ...
start "" "http://127.0.0.1:8000"

echo.
echo Starting FastAPI Uvicorn Server...
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload

pause
