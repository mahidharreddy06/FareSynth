# FareSynth AI PowerShell Launcher
Set-Location -Path $PSScriptRoot
Write-Host "=======================================================================" -ForegroundColor Cyan
Write-Host "         Starting FareSynth AI Web Application & Backend...           " -ForegroundColor Green
Write-Host "=======================================================================" -ForegroundColor Cyan

Start-Process "http://127.0.0.1:8000"
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
