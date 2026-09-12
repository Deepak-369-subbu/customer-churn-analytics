@echo off
title Stop Customer Churn Intelligence Platform
echo ==========================================================
echo   Stopping Customer Churn Platform (Port 8501)...
echo ==========================================================
echo.
python -c "import os, subprocess; out = subprocess.check_output('netstat -ano', shell=True).decode(); [subprocess.call(f'taskkill /F /PID {line.split()[-1]}', shell=True) for line in out.splitlines() if ':8501 ' in line and 'LISTENING' in line]; print('Server stopped successfully!')"
echo.
echo Dashboard server has been completely shut down.
echo.
pause
