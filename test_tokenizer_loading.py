import torch
from transformers import AutoTokenizer

def test_tokenizer_loading(model_path):
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            use_fast=True,
            trust_remote_code=True  # Enable custom tokenizer code execution
        )
        # Optionally, set a padding token if not already set
        if tokenizer.pad_token is None:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        print("Tokenizer loaded successfully.")
        print(f"Tokenizer type: {type(tokenizer)}")
    except Exception as e:
        print(f"Error loading tokenizer: {e}")

if __name__ == "__main__":
    model_path = "C:\\Projects\\KAN-WuBu-Memory\\models\\Llama_32_1B"
    test_tokenizer_loading(model_path)
