@echo off
setlocal

echo ============================================
echo  SIRD Simulator - Environment Setup
echo ============================================

REM Always run from the folder this script lives in (project root)
cd /d "%~dp0"

REM Clean up an incomplete .venv left over from a previous failed attempt
if exist ".venv" if not exist ".venv\Scripts\python.exe" (
    echo Removing incomplete .venv from a previous attempt ...
    rmdir /s /q ".venv"
)

if not exist ".venv\Scripts\python.exe" (
    echo.
    echo Checking Python version on PATH ...
    python -c "import sys; sys.exit(0 if sys.version_info >= (3, 12) else 1)"
    if errorlevel 1 (
        echo.
        echo ERROR: Python 3.12 or higher is required. Detected:
        python --version
        echo Activate a Python 3.12+ environment first ^(e.g. "conda activate base"^),
        echo then re-run this script from that same window.
        pause
        exit /b 1
    )

    echo Creating virtual environment in .venv ...
    python -m venv .venv
)

if not exist ".venv\Scripts\python.exe" (
    echo.
    echo ERROR: Failed to create the virtual environment.
    echo Make sure Python 3.12+ is active, then try again.
    pause
    exit /b 1
)

echo.
echo Installing the project and all dependencies ...
".venv\Scripts\python.exe" -m pip install --upgrade pip
".venv\Scripts\python.exe" -m pip install -e ".[dev]"

if errorlevel 1 (
    echo.
    echo ERROR: Installation failed. Check the messages above.
    pause
    exit /b 1
)

echo.
echo ============================================
echo  Setup complete!
echo.
echo  Run the app:   .venv\Scripts\streamlit run main.py
echo  Run tests:     .venv\Scripts\pytest --cov=sird_sim --cov-report=term-missing
echo  Type check:    .venv\Scripts\mypy sird_sim
echo ============================================
pause
