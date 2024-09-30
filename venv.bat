@echo off
setlocal

:: Check if virtual environment exists, if not, create it
if not exist "venv\Scripts\activate" (
    echo Creating virtual environment...
    python -m venv venv
)

:: Activate the virtual environment
call venv\Scripts\activate

:: Inform the user that the environment is active and provide a command prompt
echo Virtual environment activated. Type your commands below.

:: Open command prompt for user to type commands
cmd /K
