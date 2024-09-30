import torch
from transformers import pipeline

def test_model_loading(model_path):
    try:
        # Use device 0 (GPU) if available, else CPU
        device = 0 if torch.cuda.is_available() else -1

        # Define the prompt as a list of messages
        prompt = [
            {"role": "system", "content": "You are a helpful assistant, that responds as a pirate."},
            {"role": "user", "content": "What's Deep Learning?"},
        ]

        # Initialize the pipeline with explicit task, model, and tokenizer
        generator = pipeline(
            task="text-generation",
            model=model_path,
            tokenizer=model_path,  # Explicitly specify the tokenizer path
            device=device,
            torch_dtype=torch.float16  # Use torch.bfloat16 if supported
        )

        # Generate the response
        generation = generator(
            prompt,
            do_sample=False,
            temperature=1.0,
            top_p=1,
            max_new_tokens=50
        )

        print(f"Generation: {generation[0]['generated_text']}")
    except Exception as e:
        print(f"Error during generation: {e}")

if __name__ == "__main__":
    # Use a raw string to prevent backslash interpretation
    model_path = r"C:\Projects\KAN-WuBu-Memory\models\Llama_32_1B"
    test_model_loading(model_path)
