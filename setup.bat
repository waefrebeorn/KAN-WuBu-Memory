@echo off
setlocal enabledelayedexpansion

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

:: Install PyTorch with CUDA support
echo Installing PyTorch with CUDA support...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

:: Install other requirements
echo Installing other requirements...
pip install -r requirements.txt

:: Install Hugging Face transformers from source
echo Installing latest Hugging Face transformers...
pip install git+https://github.com/huggingface/transformers

:: Install Accelerate (corrected)
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
echo    - generation_config.json
echo    - special_tokens_map.json
echo 4. Place these files in the 'models\Meta-Llama-3.1-8B-Instruct' directory.
echo.

echo Setup completed successfully. You can now run the script using run.bat.
pause