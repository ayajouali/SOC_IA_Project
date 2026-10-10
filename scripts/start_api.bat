@echo off
echo ========================================
echo   SOC-IA API - Demarrage
echo ========================================
cd /d "%~dp0\.."
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000