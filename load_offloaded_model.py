import os
import torch
import json
import numpy as np
import re
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import LlamaForCausalLM, AutoTokenizer, LlamaConfig

# Define paths to the directories and files
SOURCE_DIR = "models/Llama_32_1B/"
WEIGHTS_DIR = os.path.join(SOURCE_DIR, "offload")
MODEL_JSON_PATH = os.path.join(SOURCE_DIR, "config.json")

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set device to GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type != "cuda":
    logger.error("CUDA-enabled GPU not found. Please ensure a compatible GPU is available.")
    raise SystemExit("CUDA-enabled GPU not found.")

# Load the configuration from the JSON file
def load_configuration(model_json_path):
    with open(model_json_path, "r") as f:
        config_data = json.load(f)
    config = LlamaConfig(**config_data)
    return config

# Use AutoTokenizer instead of LlamaTokenizer to resolve class conflicts
def load_tokenizer(source_dir):
    return AutoTokenizer.from_pretrained(source_dir)

# Load the model configuration
logger.info(f"Loading model configuration from: {MODEL_JSON_PATH}")
config = load_configuration(MODEL_JSON_PATH)

# Initialize an empty model based on the configuration and move it to GPU
model = LlamaForCausalLM(config).to(device)
logger.info("Initialized empty LLaMA model on GPU.")

# Load the offloaded weights from the `.dat` files directly to GPU
def load_dat_file(file_path, dtype):
    with open(file_path, 'rb') as f:
        tensor_data = np.fromfile(f, dtype=dtype)
    loaded_tensor = torch.from_numpy(tensor_data).to(device)

    # If dtype was mapped to float32 for bfloat16 compatibility, convert back
    if dtype == np.float32 and "bfloat16" in file_path:
        loaded_tensor = loaded_tensor.to(torch.bfloat16)
    return loaded_tensor

def load_offloaded_weights(model, weights_dir):
    for name, param in model.named_parameters():
        file_name = name.replace('.', '_') + ".dat"
        file_path = os.path.join(weights_dir, file_name)

        if os.path.exists(file_path):
            dtype_map = {
                torch.float16: np.float16,
                torch.float32: np.float32,
                torch.int64: np.int64,
                torch.int32: np.int32,
                torch.bfloat16: np.float32,
            }
            expected_dtype = dtype_map.get(param.dtype, np.float32)
            logger.info(f"Loading {file_name} into {name} with expected type {expected_dtype}")
            loaded_tensor = load_dat_file(file_path, expected_dtype).view_as(param)

            if param.dtype == torch.bfloat16:
                loaded_tensor = loaded_tensor.to(torch.bfloat16)

            with torch.no_grad():
                param.data.copy_(loaded_tensor)
        else:
            logger.warning(f"Warning: {file_name} not found in offloaded directory.")

# Load the weights into the model
load_offloaded_weights(model, WEIGHTS_DIR)

# Ensure the model is in evaluation mode and on GPU
model.eval()
model.to(device)
logger.info("Model weights loaded successfully and moved to GPU.")

# Load tokenizer
logger.info(f"Loading tokenizer from directory: {SOURCE_DIR}")
tokenizer = load_tokenizer(SOURCE_DIR)

# Implement the ResponseQualityManager with metrics and corrective strategies
class ResponseQualityManager:
    def __init__(self, kan_model, tokenizer):
        self.kan_model = kan_model
        self.tokenizer = tokenizer
        self.tfidf_vectorizer = TfidfVectorizer()

    def evaluate_response(self, user_input, response):
        relevance_score = self.calculate_relevance(user_input, response)
        structure_valid = self.has_proper_structure(response)
        is_garbled = self.detect_garbled_output(response)
        return relevance_score > 0.3 and structure_valid and not is_garbled

    def calculate_relevance(self, user_input, response):
        user_tokens = set(self.tokenizer.tokenize(user_input))
        response_tokens = set(self.tokenizer.tokenize(response))
        overlap = len(user_tokens.intersection(response_tokens))
        overlap_score = overlap / max(len(user_tokens), 1)

        combined_texts = [user_input, response]
        tfidf_matrix = self.tfidf_vectorizer.fit_transform(combined_texts)
        cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]

        return 0.5 * overlap_score + 0.5 * cosine_sim

    def detect_garbled_output(self, response):
        if re.search(r'[^\x00-\x7F]+', response):
            return True
        if len(response.split()) < 3:
            return True
        if response.count('.') / max(len(response.split()), 1) > 0.5:
            return True
        return False

    def has_proper_structure(self, response):
        sentences = re.split(r'(?<=[.!?])\s+', response.strip())
        return len(sentences) > 0 and sentences[0][0].isupper() and sentences[-1][-1] in '.!?'

# Quality Manager instance for response evaluation
quality_manager = ResponseQualityManager(model, tokenizer)

# Updated generation logic to handle context better and avoid repetitive responses
def generate_response(input_text, model, tokenizer, max_new_tokens=150, pad_token_id=128001, history=[], context_limit=512):
    # Clean the history to avoid redundant prompts
    history = [line for line in history if line.strip()]  # Remove empty lines

    # Create a simplified context prompt from the last few exchanges
    prompt = f"{' '.join(history[-3:])}\nUser: {input_text}\n" if history else f"User: {input_text}\n"

    # Prepare inputs for the model
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=context_limit).to(device)

    # Generate the response
    with torch.no_grad():
        outputs = model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=max_new_tokens,  # Control new tokens
            do_sample=True,
            temperature=0.7,
            top_k=50,
            top_p=0.9,
            repetition_penalty=1.2,
            pad_token_id=pad_token_id,
            early_stopping=True
        )

    # Decode the response and format it properly
    response = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

    # Ensure clean history management and context length control
    cleaned_response = response.split("User:")[-1].strip()  # Remove any overlap
    cleaned_response = re.sub(r'\s+', ' ', cleaned_response)  # Clean excess whitespace

    # Append the cleaned response to history
    history.append(f"User: {input_text}\nModel: {cleaned_response}")

    # Trim history to prevent excessive accumulation
    if len(history) > 6:
        history = history[-6:]

    return cleaned_response, history

# Updated user input loop to handle context better
def user_input_loop(model, tokenizer):
    print("\n--- LLaMA Instruct Model Interactive Query ---")
    print("Type 'exit' to quit.")
    history = []  # Initialize a history buffer to keep track of conversation
    while True:
        try:
            user_input = input("\nEnter your query: ")
        except (EOFError, KeyboardInterrupt):
            print("\nExiting...")
            break

        if user_input.lower() == 'exit':
            print("Exiting...")
            break

        if not user_input.strip():
            print("Please enter a valid query.")
            continue

        response, history = generate_response(user_input, model, tokenizer, history=history)

        # Evaluate response quality
        if not quality_manager.evaluate_response(user_input, response):
            logger.warning("Generated response failed quality checks. Regenerating...")
            response, history = generate_response(user_input, model, tokenizer, history=history)

        print(f"Model Response: {response}")

# Start the interactive query loop with the refined response generation
logger.info("Model loaded successfully. You can now query the model.")
user_input_loop(model, tokenizer)
