@echo off
echo Installing Python dependencies...
pip install -r requirements.txt

echo Starting Fire Incident ML Backend...
python app.py

pause