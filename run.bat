@echo off
setlocal

:: Activate the virtual environment
call venv\Scripts\activate

:: Run the GUI script
python kan_gui.py

pause