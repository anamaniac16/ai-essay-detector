@echo off
echo ============================================================
echo   AI Essay Detector — 1-Command Launcher
echo ============================================================
if not exist venv (
    echo Creating Python virtual environment...
    python -m venv venv
)
echo Activating virtual environment...
call venv\Scripts\activate.bat

echo Installing dependencies...
pip install -r requirements.txt

echo Launching Streamlit interface...
streamlit run app.py
