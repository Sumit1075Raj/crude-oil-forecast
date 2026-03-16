@echo off
:: Change to the project folder (wherever this .bat file lives)
cd /d "%~dp0"
call venv\Scripts\activate.bat
streamlit run dashboard\app.py
pause
