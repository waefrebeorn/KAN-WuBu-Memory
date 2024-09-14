@echo off
setlocal

:: Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Python is not installed. Please install Python 3.8 or later from https://www.python.org/downloads/
    exit /b 1
)

:: Create a virtual environment
python -m venv venv
if %errorlevel% neq 0 (
    echo Failed to create virtual environment.
    exit /b 1
)

:: Activate the virtual environment
call venv\Scripts\activate

:: Upgrade pip
python -m pip install --upgrade pip

:: Install requirements
pip install -r requirements.txt

echo Setup completed successfully. You can now run the script using run.bat.
pause