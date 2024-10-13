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
logging.info(f"Loading model configuration from: {MODEL_JSON_PATH}")
config = load_configuration(MODEL_JSON_PATH)

# Initialize an empty model based on the configuration
model = LlamaForCausalLM(config)
logging.info("Initialized empty LLaMA model.")

# Load the offloaded weights from the `.dat` files
def load_dat_file(file_path, dtype):
    with open(file_path, 'rb') as f:
        tensor_data = np.fromfile(f, dtype=dtype)
    loaded_tensor = torch.tensor(tensor_data)
    
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
            logging.info(f"Loading {file_name} into {name} with expected type {expected_dtype}")
            loaded_tensor = load_dat_file(file_path, expected_dtype).view_as(param)

            if param.dtype == torch.bfloat16:
                loaded_tensor = loaded_tensor.to(torch.bfloat16)

            param.data.copy_(loaded_tensor.to("cuda"))
        else:
            logging.warning(f"Warning: {file_name} not found in offloaded directory.")

# Load the weights into the model
load_offloaded_weights(model, WEIGHTS_DIR)

# Move the model to GPU for inference
model.to('cuda')
model.eval()

# Use AutoTokenizer to handle any tokenizer class discrepancies
logging.info(f"Loading tokenizer from directory: {SOURCE_DIR}")
tokenizer = load_tokenizer(SOURCE_DIR)

# ResponseQualityManager implementation for response evaluation
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
        if response.count('.') / len(response.split()) > 0.5:
            return True
        return False

    def has_proper_structure(self, response):
        sentences = re.split(r'(?<=[.!?])\s+', response.strip())
        return len(sentences) > 0 and sentences[0][0].isupper() and sentences[-1][-1] in '.!?'

# Helper function to calculate entropy
def calculate_entropy(probs):
    log_probs = torch.log(probs + 1e-10)  # Add small epsilon to avoid log(0)
    entropy = -torch.sum(probs * log_probs, dim=-1)
    return entropy

# Updated generation logic to handle context better, avoid repetitive responses, and include entropy-based refinement
def generate_response(input_text, model, tokenizer, max_new_tokens=50, pad_token_id=128001, history=[], context_limit=512, entropy_threshold=0.7):
    # Clean the history to avoid redundant prompts
    history = [line for line in history if line.strip()]  # Remove empty lines
    
    # Create a simplified context prompt from the last few exchanges
    prompt = f"{' '.join(history[-3:])}\nUser: {input_text}\n" if history else f"User: {input_text}\n"
    
    # Prepare inputs for the model
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=context_limit).to("cuda")

    # Initialize response and keep track of tokens for refinement
    refined_response = ""
    refined_token_ids = []
    full_response = ""

    # Iteratively generate responses
    for iteration in range(10):  # Number of iterations can be adjusted as needed
        logging.info(f"Iteration {iteration + 1}: Generating tokens...")

        # Generate a chunk of response
        with torch.no_grad():
            outputs = model.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=max_new_tokens,  # Control new tokens per iteration
                do_sample=True,
                temperature=0.7,
                top_k=50,
                top_p=0.9,
                repetition_penalty=1.2,
                pad_token_id=pad_token_id,
                early_stopping=True
            )

        # Decode and refine based on entropy
        token_ids = outputs[0].tolist()
        refined_token_ids.extend(token_ids)

        # Convert logits to probabilities and calculate entropy
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1)
        entropies = calculate_entropy(probs)

        # If entropy is manageable, continue refining; otherwise, break
        if entropies.mean() < entropy_threshold:
            refined_response = tokenizer.decode(refined_token_ids, skip_special_tokens=True).strip()
            full_response = f"{full_response} {refined_response}".strip()
        else:
            logging.warning("High entropy detected, refining response before continuing.")
            break

    # Append final cleaned response to history
    history.append(f"User: {input_text}\nModel: {full_response}")
    
    # Trim history to avoid excessive accumulation
    if len(history) > 6:
        history = history[-6:]

    return full_response, history

# Updated user input loop to handle context better
def user_input_loop(model, tokenizer):
    print("\n--- LLaMA Instruct Model Interactive Query ---")
    print("Type 'exit' to quit.")
    history = []  # Initialize a history buffer to keep track of conversation
    while True:
        user_input = input("\nEnter your query: ")
        if user_input.lower() == 'exit':
            print("Exiting...")
            break
        response, history = generate_response(user_input, model, tokenizer, history=history)
        print(f"Model Response: {response}")

# Start the interactive query loop with the refined response generation
logging.info("Model loaded successfully. You can now query the model.")
user_input_loop(model, tokenizer)
