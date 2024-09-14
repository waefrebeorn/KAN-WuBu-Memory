import sys
print(f"Python version: {sys.version}")

print("Importing torch...")
import torch
print(f"Torch version: {torch.__version__}")

print("\nGPU Information:")
print(f"CUDA is available: {torch.cuda.is_available()}")
print(f"CUDA device count: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f"CUDA device {i} name: {torch.cuda.get_device_name(i)}")

# Attempt to set the default CUDA device to the NVIDIA GPU
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        if "NVIDIA" in torch.cuda.get_device_name(i):
            torch.cuda.set_device(i)
            print(f"Set default CUDA device to: {torch.cuda.get_device_name(i)}")
            break
    print(f"Current CUDA device: {torch.cuda.current_device()}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

print("\nImporting transformers...")
try:
    import transformers
    print(f"Transformers version: {transformers.__version__}")
except ImportError as e:
    print(f"Error importing transformers: {e}")

print("\nInitializing AutoTokenizer...")
try:
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-medium-128k-instruct")
    print("AutoTokenizer initialized successfully.")
except Exception as e:
    print(f"Error initializing AutoTokenizer: {e}")

print("\nInitializing AutoModelForCausalLM...")
try:
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained("microsoft/Phi-3-medium-128k-instruct").to(device)
    print(f"AutoModelForCausalLM initialized successfully on {device}.")
except Exception as e:
    print(f"Error initializing AutoModelForCausalLM: {e}")

print("\nTest complete.")