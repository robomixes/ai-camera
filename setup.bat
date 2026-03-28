@echo off
REM AI Camera - Setup Script (Windows)

echo === AI Camera Setup ===

REM Create virtual environment
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
)

REM Activate
call venv\Scripts\activate

REM Install dependencies
echo Installing dependencies...
pip install --upgrade pip
pip install -r requirements.txt

REM Create runtime directories
echo Creating directories...
if not exist "logs" mkdir logs
if not exist "event_images" mkdir event_images
if not exist "roi_events" mkdir roi_events
if not exist "plate_images" mkdir plate_images
if not exist "output_images" mkdir output_images
if not exist "people_search_queue\ready" mkdir people_search_queue\ready

echo.
echo === Setup Complete ===
echo.
echo To start the dashboard:
echo   venv\Scripts\activate
echo   python run_web.py
echo.
echo Then open http://localhost:8080
echo Default login: admin / admin
echo.
