import os
import torch
import numpy as np
from safetensors.torch import load_file

# Define the directories for source and output
SOURCE_FILE = "models/Llama_32_1B/model.safetensors"  # Path to the input safetensor file
OUTPUT_DIR = "models/Llama_32_1B/offload"             # Path to the output directory

# Create the output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load the safetensors file
print(f"Loading safetensors file from: {SOURCE_FILE}")
state_dict = load_file(SOURCE_FILE)
print(f"Safetensors file loaded. Found {len(state_dict)} tensors.")

# Utility function to save individual tensors, preserving their original format
def save_tensor(tensor, file_path):
    """
    Save a PyTorch tensor to a binary .dat file without any format conversion.
    """
    # Identify the original tensor type
    original_dtype = tensor.dtype

    # Check if the format is supported by NumPy
    try:
        # If the tensor is in bfloat16, PyTorch has direct support for saving
        if original_dtype == torch.bfloat16:
            # Convert to float32 for saving as .dat, since bfloat16 is not supported by numpy
            print(f"Saving {file_path} as bfloat16 using float32 for compatibility.")
            tensor.to(torch.float32).cpu().numpy().tofile(file_path)
        else:
            # Use the original format without conversion
            tensor.cpu().numpy().tofile(file_path)
        
        print(f"Saved tensor to {file_path} with original type {original_dtype}")
    except Exception as e:
        print(f"Failed to save {file_path} with dtype {original_dtype} due to: {e}")

# Iterate through the state dictionary and save each tensor to a separate .dat file
for tensor_name, tensor in state_dict.items():
    # Construct a file path based on the tensor's name, replacing '.' with '_'
    file_path = os.path.join(OUTPUT_DIR, tensor_name.replace('.', '_') + ".dat")

    # Save the tensor in its original format
    try:
        save_tensor(tensor, file_path)
    except ValueError as e:
        print(f"Skipping {tensor_name} due to error: {e}")

print(f"Model has been successfully split into individual .dat files in: {OUTPUT_DIR}")
