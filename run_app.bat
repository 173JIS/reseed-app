@echo off
chcp 65001 >nul
title ReSeed App

SET "APP_DIR=%~dp0"
SET "APP_FILE=%APP_DIR%app.py"
SET "REQ_FILE=%APP_DIR%requirements.txt"
SET "PORT=8501"
SET "PYTHON="

echo.
echo  ==========================================
echo   ReSeed - Drone SeedBall Decision System
echo  ==========================================
echo.

:: 1. conda/anaconda Python (priority)
CALL :TryPython "%USERPROFILE%\miniconda3\python.exe"
CALL :TryPython "%USERPROFILE%\anaconda3\python.exe"
CALL :TryPython "C:\miniconda3\python.exe"
CALL :TryPython "C:\anaconda3\python.exe"
CALL :TryPython "C:\ProgramData\miniconda3\python.exe"
CALL :TryPython "C:\ProgramData\anaconda3\python.exe"

:: 2. Fallback: system PATH python (skip Windows Store stub)
IF NOT DEFINED PYTHON (
    FOR /F "tokens=* delims=" %%i IN ('WHERE python.exe 2^>nul') DO (
        IF NOT DEFINED PYTHON (
            echo %%i | findstr /i "WindowsApps" >nul 2>&1
            IF ERRORLEVEL 1 SET "PYTHON=%%i"
        )
    )
)

IF NOT DEFINED PYTHON (
    echo [ERROR] Python not found.
    echo         Install miniconda3, anaconda3, or Python 3.9+.
    pause
    exit /b 1
)

echo [Python]  %PYTHON%

:: 3. Install packages if streamlit is missing
"%PYTHON%" -c "import streamlit" >nul 2>&1
IF ERRORLEVEL 1 (
    echo [Install] Installing packages from requirements.txt ...
    "%PYTHON%" -m pip install -r "%REQ_FILE%"
    IF ERRORLEVEL 1 (
        echo [ERROR] pip install failed. Check internet connection.
        pause
        exit /b 1
    )
)

:: 4. If already running, just open browser
netstat -ano | findstr ":%PORT% " | findstr "LISTENING" >nul 2>&1
IF NOT ERRORLEVEL 1 (
    echo [Info] App already running on port %PORT%.
    start "" "http://localhost:%PORT%"
    pause
    exit /b 0
)

:: 5. Launch app
echo [Start]   http://localhost:%PORT%
echo           Press Ctrl+C or close this window to stop.
echo.
"%PYTHON%" -m streamlit run "%APP_FILE%" --server.port %PORT% --browser.gatherUsageStats false
GOTO :EOF

:TryPython
IF NOT DEFINED PYTHON IF EXIST %1 SET "PYTHON=%~1"
EXIT /B 0
