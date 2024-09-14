@echo off
setlocal

:: Activate the virtual environment
call venv\Scripts\activate

:: Run the main script
python kan_emotional_character_phi3.py

pause