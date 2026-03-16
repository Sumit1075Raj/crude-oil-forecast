@echo off
echo ============================================================
echo   CrudeEdge - Environment Setup
echo ============================================================
echo.

:: Check Python 3.11
py -3.11 --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python 3.11 not found. Please install it from:
    echo https://www.python.org/downloads/release/python-3119/
    pause
    exit /b 1
)

echo [1/4] Python 3.11 found.

:: Create virtual environment
echo [2/4] Creating virtual environment...
py -3.11 -m venv venv

:: Activate and upgrade pip
echo [3/4] Installing dependencies (this may take 3-5 minutes)...
call venv\Scripts\activate.bat
python -m pip install --upgrade pip --quiet
pip install -r requirements.txt

:: Done
echo.
echo [4/4] Done! Environment is ready.
echo.
echo ============================================================
echo   To activate later:   venv\Scripts\activate
echo   To run pipeline:     python run_pipeline.py
echo   To run dashboard:    streamlit run dashboard\app.py
echo ============================================================
echo.
pause
