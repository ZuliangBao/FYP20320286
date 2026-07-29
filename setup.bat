@echo off
setlocal

echo ============================================
echo  SIRD Simulator - Environment Setup
echo ============================================

REM Always run from the folder this script lives in (project root)
cd /d "%~dp0"

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
python --version

echo.
echo Installing the project into the current Python environment ...
echo (Packages already installed here are skipped, so this is fast
echo  on a machine that already has numpy/matplotlib/streamlit etc.)
python -m pip install -e ".[dev]"

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
echo  Run the app:   streamlit run main.py
echo  Run tests:     pytest --cov=sird_sim --cov-report=term-missing
echo  Type check:    mypy sird_sim
echo ============================================
pause
