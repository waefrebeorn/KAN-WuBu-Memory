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

# Entropy calculation function
def calculate_entropy(probs):
    log_probs = torch.log(probs + 1e-10)  # Avoid log(0)
    entropy = -torch.sum(probs * log_probs, dim=-1)
    return entropy

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

# Load the tokenizer
logging.info(f"Loading tokenizer from directory: {SOURCE_DIR}")
tokenizer = load_tokenizer(SOURCE_DIR)

# Adaptive layer squeezing based on entropy
def adaptive_layer_squeeze(model, inputs, entropy_threshold=0.5):
    hidden_states, _ = model(inputs["input_ids"], output_hidden_states=True).hidden_states
    
    for i, hidden_state in enumerate(hidden_states):
        # Calculate entropy for each layer
        layer_entropy = calculate_entropy(torch.softmax(hidden_state, dim=-1))
        
        # Squeeze (prune) layers based on entropy
        if layer_entropy.mean() < entropy_threshold:
            model.model.layers[i].forward = lambda x: x  # Skip this layer
    
    return model(inputs["input_ids"])

# ResponseQualityManager with entropy-based evaluation
class ResponseQualityManager:
    def __init__(self, kan_model, tokenizer):
        self.kan_model = kan_model
        self.tokenizer = tokenizer
        self.tfidf_vectorizer = TfidfVectorizer()

    def evaluate_response(self, user_input, response, entropy):
        relevance_score = self.calculate_relevance(user_input, response)
        structure_valid = self.has_proper_structure(response)
        is_garbled = self.detect_garbled_output(response)
        
        # Use entropy to refine response evaluation
        if entropy.mean() > 0.7:
            logging.info("High entropy detected, refining response.")
            response = self.refine_response(response)
        
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

    def refine_response(self, response):
        # Implement response refinement logic (could include rerunning the model)
        return response  # Placeholder

# Response generation with entropy-driven refinement
def generate_response(input_text, model, tokenizer, max_new_tokens=150, pad_token_id=128001, history=[], context_limit=512, entropy_threshold=0.7):
    # Clean the history to avoid redundant prompts
    history = [line for line in history if line.strip()]
    
    # Create a simplified context prompt from the last few exchanges
    prompt = f"{' '.join(history[-3:])}\nUser: {input_text}\n" if history else f"User: {input_text}\n"
    
    # Prepare inputs for the model
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=context_limit).to("cuda")
    
    # Adaptive layer squeezing
    model_output = adaptive_layer_squeeze(model, inputs)
    
    # Calculate entropy for the output logits
    probs = torch.softmax(model_output.logits, dim=-1)
    entropy = calculate_entropy(probs)

    # Generate the response
    response = tokenizer.decode(model_output.logits.argmax(dim=-1), skip_special_tokens=True).strip()
    
    # Evaluate and refine response based on entropy
    response_quality_manager = ResponseQualityManager(model, tokenizer)
    refined_response = response_quality_manager.evaluate_response(input_text, response, entropy)

    # Append the refined response to history
    history.append(f"User: {input_text}\nModel: {refined_response}")
    
    # Trim history to prevent excessive accumulation
    if len(history) > 6:
        history = history[-6:]

    return refined_response, history

# User input loop with refined response generation
def user_input_loop(model, tokenizer):
    print("\n--- LLaMA Instruct Model Interactive Query ---")
    print("Type 'exit' to quit.")
    history = []
    while True:
        user_input = input("\nEnter your query: ")
        if user_input.lower() == 'exit':
            print("Exiting...")
            break
        response, history = generate_response(user_input, model, tokenizer, history=history)
        print(f"Model Response: {response}")

# Start the interactive query loop
logging.info("Model loaded successfully. You can now query the model.")
user_input_loop(model, tokenizer)
