#!/usr/bin/env bash
echo "============================================================"
echo "  AI Essay Detector — 1-Command Launcher"
echo "============================================================"
if [ ! -d "venv" ]; then
    echo "Creating Python virtual environment..."
    python3 -m venv venv
fi
echo "Activating virtual environment..."
source venv/bin/activate

echo "Installing dependencies..."
pip install -r requirements.txt

echo "Launching Streamlit interface..."
streamlit run app.py
