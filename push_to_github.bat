@echo off
title Push FareSynth AI to GitHub
echo =======================================================================
echo          Pushing FareSynth AI to github.com/mahidharreddy06/FareSynth
echo =======================================================================
echo.
cd /d "%~dp0"

git push -u origin main

echo.
echo =======================================================================
echo Done! Check your repo at: https://github.com/mahidharreddy06/FareSynth
echo =======================================================================
pause
