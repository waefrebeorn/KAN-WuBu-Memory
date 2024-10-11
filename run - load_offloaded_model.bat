@echo off
setlocal

:: Activate the virtual environment
call venv\Scripts\activate

:: Run the GUI script
python load_offloaded_model.py

pause