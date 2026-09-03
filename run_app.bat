@echo off
title HealthLens AI Diagnostic Suite
cd /d "%~dp0"
echo =======================================================
echo Starting HealthLens AI Clinical Suite...
echo URL: http://localhost:8000
echo =======================================================
start "" http://localhost:8000
python main.py
pause
