@echo off
setlocal enabledelayedexpansion

echo Starting setup for KAN-WuBu-Memory with LLaMA 3.2 1B Model...

:: Define project-specific paths
set "PROJECT_DIR=%~dp0"
set "MODEL_DIR=%PROJECT_DIR%models\Llama_32_1B"

:: Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Python is not installed. Please install Python 3.8 or later from https://www.python.org/downloads/
    exit /b 1
)

:: Create the necessary folder structure
if not exist "%MODEL_DIR%" (
    echo Creating LLaMA 3.2 1B model directory...
    mkdir "%MODEL_DIR%"
    if %errorlevel% neq 0 (
        echo Failed to create the LLaMA 3.2 model directory.
        exit /b 1
    )
)

echo Directory structure created successfully: %MODEL_DIR%

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

:: Install PyTorch with CUDA support
echo Installing PyTorch with CUDA support...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

:: Install other requirements
echo Installing other requirements...
pip install -r requirements.txt

:: Install Hugging Face transformers
echo Installing latest Hugging Face transformers...
pip install git+https://github.com/huggingface/transformers

:: Install Accelerate
echo Installing Accelerate...
pip install accelerate>=0.26.0

:: Verify CUDA installation
echo Verifying CUDA installation...
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('CUDA version:', torch.version.cuda if torch.cuda.is_available() else 'N/A')"

:: Additional CUDA diagnostics
echo.
echo Running CUDA diagnostics...
python -c "import torch; print('CUDA device count:', torch.cuda.device_count()); print('CUDA device name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"

:: Check NVIDIA driver
echo.
echo Checking NVIDIA driver...
nvidia-smi

echo Environment setup complete.

echo.
echo IMPORTANT: Manual Model Download Required
echo ==========================================
echo You have two options to get the LLaMA models:
echo 1. **Directly from Meta:**
echo    - Visit the LLaMA download form at [https://www.llama.com/llama-downloads]
echo    - Fill in your details, select the models you want, and accept the licenses.
echo    - Check your email for download instructions and a pre-signed URL to download the model files:
echo        - checklist.chk
echo        - consolidated.00.pth
echo        - params.json
echo        - tokenizer.model
echo    - Place these files in the following directory:
echo        %MODEL_DIR%
echo.
echo 2. **From Hugging Face:**
echo    - Use the following command to download directly:
echo        huggingface-cli login
echo        huggingface-cli download meta-llama/Llama-3.2-1B-Instruct --include "checklist.chk,consolidated.00.pth,params.json,tokenizer.model" --local-dir "%MODEL_DIR%"

echo.
echo Setup completed successfully. You can now run the main script using run.bat.
pause
