@echo off
setlocal

echo Starting setup for KAN Emotional Character with LLaMA 3.1 8B Instruct...

:: Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Python is not installed. Please install Python 3.8 or later from https://www.python.org/downloads/
    exit /b 1
)

:: Create a virtual environment if it doesn't exist
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
    if %errorlevel% neq 0 (
        echo Failed to create virtual environment.
        exit /b 1
    )
)

:: Activate the virtual environment
call venv\Scripts\activate
if %errorlevel% neq 0 (
    echo Failed to activate virtual environment.
    exit /b 1
)

:: Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip

:: Install requirements
echo Installing requirements...
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo Failed to install requirements.
    exit /b 1
)

:: Install PyTorch with CUDA support
echo Installing PyTorch with CUDA support...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
if %errorlevel% neq 0 (
    echo Failed to install PyTorch.
    exit /b 1
)

:: Install Hugging Face transformers
echo Installing latest Hugging Face transformers...
pip install git+https://github.com/huggingface/transformers
if %errorlevel% neq 0 (
    echo Failed to install Hugging Face transformers.
    exit /b 1
)

echo Environment setup complete.

echo.
echo IMPORTANT: Manual Model Download Required
echo ==========================================
echo 1. Visit: https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct
echo 2. Accept the license agreement if you haven't already.
echo 3. Download the following files:
echo    - config.json
echo    - model-00001-of-00004.safetensors
echo    - model-00002-of-00004.safetensors
echo    - model-00003-of-00004.safetensors
echo    - model-00004-of-00004.safetensors
echo    - tokenizer.json
echo    - tokenizer_config.json
echo 4. Place these files in the 'models\Meta-Llama-3.1-8B-Instruct' directory.
echo.
echo Press any key when you have completed this step...
pause > nul

:: Check if the model directory exists
if not exist "models\Meta-Llama-3.1-8B-Instruct" (
    echo Creating model directory...
    mkdir "models\Meta-Llama-3.1-8B-Instruct"
)

:: Run the main script
echo Starting KAN Emotional Character application...
python kan_gui.py

if %errorlevel% neq 0 (
    echo An error occurred while running the application.
    exit /b 1
)

echo Application closed.

:: Keep the window open
pause