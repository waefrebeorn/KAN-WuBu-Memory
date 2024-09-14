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

:: Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

:: Install other requirements
pip install -r requirements.txt

:: Install Hugging Face transformers from source (for latest LLaMA support)
pip install git+https://github.com/huggingface/transformers

echo Setup completed successfully. You can now run the script using run.bat.
pause