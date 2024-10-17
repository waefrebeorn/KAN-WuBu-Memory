import os
import torch
import torch.nn as nn
import json
import numpy as np
import re
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import LlamaForCausalLM, AutoTokenizer, LlamaConfig

import copy
from torch.utils.checkpoint import checkpoint

# Enable CUDA launch blocking for accurate error reporting
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# Define paths to the directories and files
SOURCE_DIR = r"C:\Projects\KAN-WuBu-Memory\models\Llama_32_1B"
WEIGHTS_DIR = os.path.join(SOURCE_DIR, "offload")
MODEL_JSON_PATH = os.path.join(SOURCE_DIR, "config.json")
TOKENIZER_CONFIG_PATH = os.path.join(SOURCE_DIR, "tokenizer_config.json")

# Initialize logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(message)s')

# Check CUDA availability
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available. Please ensure that a compatible GPU is installed and CUDA is properly configured.")

device = torch.device('cuda')

# Load the model configuration from the JSON file
def load_configuration(MODEL_JSON_PATH):
    with open(MODEL_JSON_PATH, "r") as f:
        config_data = json.load(f)
    config = LlamaConfig(**config_data)
    logging.info("Model configuration loaded successfully.")
    return config

# Function to update tokenizer's vocab_size in tokenizer_config.json
def update_tokenizer_vocab_size(tokenizer_config_path, correct_vocab_size):
    with open(tokenizer_config_path, "r") as f:
        tokenizer_config = json.load(f)
    
    if tokenizer_config.get("vocab_size", None) != correct_vocab_size:
        logging.info(f"Updating tokenizer vocab_size from {tokenizer_config.get('vocab_size')} to {correct_vocab_size}")
        tokenizer_config["vocab_size"] = correct_vocab_size
        with open(tokenizer_config_path, "w") as f:
            json.dump(tokenizer_config, f, indent=2)
        logging.info("Tokenizer config.json updated successfully.")
    else:
        logging.info(f"Tokenizer vocab_size is already set to {correct_vocab_size}.")

# Load and possibly update tokenizer configuration
def prepare_tokenizer_config(tokenizer_config_path, correct_vocab_size):
    update_tokenizer_vocab_size(tokenizer_config_path, correct_vocab_size)


def load_tokenizer_with_model_config(MODEL_JSON_PATH):
    # Load the correct model configuration
    with open(MODEL_JSON_PATH, "r") as f:
        model_config = json.load(f)

    # Initialize tokenizer by loading the config directly from MODEL_JSON_PATH
    tokenizer = AutoTokenizer.from_pretrained(MODEL_JSON_PATH)
    logging.info("Tokenizer loaded successfully using the model's config.json.")

    # Check if vocab size matches the model config (since we can't set vocab_size directly)
    if tokenizer.vocab_size != model_config['vocab_size']:
        logging.error(f"Tokenizer vocab_size ({tokenizer.vocab_size}) does not match model vocab_size ({model_config['vocab_size']}).")
        raise ValueError("Tokenizer vocab_size does not match model config vocab_size.")

    # Ensure special tokens are correctly set based on the model config
    tokenizer.bos_token_id = model_config["bos_token_id"]
    tokenizer.eos_token_ids = model_config["eos_token_id"]
    logging.info(f"BOS token ID set to: {tokenizer.bos_token_id}")
    logging.info(f"EOS token IDs set to: {tokenizer.eos_token_ids}")

    return tokenizer



# SharedLayer class remains unchanged
class SharedLayer(nn.Module):
    def __init__(self, hidden_size):
        super(SharedLayer, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.attention = nn.MultiheadAttention(hidden_size, num_heads=8)

    def forward(self, x):
        x = self.mlp(x)
        x, _ = self.attention(x, x, x)
        return x

# LoRA class remains unchanged
class LoRA(nn.Module):
    def __init__(self, hidden_size, rank=8):
        super(LoRA, self).__init__()
        self.rank = rank
        self.lora_A = nn.Linear(hidden_size, rank, bias=False)
        self.lora_B = nn.Linear(rank, hidden_size, bias=False)

    def forward(self, x):
        return x + self.lora_B(self.lora_A(x))

# Optimized module for multiple stacked LLaMA layers using shared weights
class OptimizedStackedLlamaModule(nn.Module):
    def __init__(self, config, num_layers=6):
        super(OptimizedStackedLlamaModule, self).__init__()
        self.num_layers = num_layers
        # Create a list of deep-copied models to ensure each layer has its own instance
        self.models = nn.ModuleList([copy.deepcopy(LlamaForCausalLM(config).to(device)) for _ in range(self.num_layers)])
        logging.info(f"Initialized {self.num_layers} LlamaForCausalLM instances for shared layers.")

    def forward_pass(self, input_ids, attention_mask, layer_num):
        if layer_num >= self.num_layers:
            raise IndexError(f"layer_num {layer_num} is out of range for {self.num_layers} layers.")
        outputs = self.models[layer_num](input_ids=input_ids, attention_mask=attention_mask)
        return outputs.logits

    def forward(self, input_ids, attention_mask=None):
        x = input_ids
        for layer_num in range(self.num_layers):
            if not torch.is_tensor(x):
                raise ValueError("Input to shared_model must be a tensor.")
            if x.dtype != torch.long:
                logging.warning(f"Converting input tensor from {x.dtype} to torch.long")
                x = x.long()
            x = self.forward_pass(x, attention_mask, layer_num)
        return x

# Optimized Stacked LLaMA Network with shared components and LoRA
class OptimizedStackedLlamaNetwork(nn.Module):
    def __init__(self, config, num_stacks=3):
        super(OptimizedStackedLlamaNetwork, self).__init__()
        self.shared_model = OptimizedStackedLlamaModule(config)
        self.num_stacks = num_stacks
        self.linears = nn.ModuleList([nn.Linear(config.hidden_size, config.hidden_size).to(device) for _ in range(num_stacks)])
        self.shared_layer = SharedLayer(config.hidden_size).to(device)
        self.lora_adapters = nn.ModuleList([LoRA(config.hidden_size).to(device) for _ in range(num_stacks)])
        logging.info(f"Initialized OptimizedStackedLlamaNetwork with {self.num_stacks} stacks.")

    def forward(self, input_ids, attention_mask=None):
        x = input_ids
        for i in range(self.num_stacks):
            x = self.shared_model(x, attention_mask)
            x = self.linears[i](x)
            x = self.shared_layer(x)
            x = self.lora_adapters[i](x)
        return x

# Function to load tensors from .dat files directly to GPU
def load_dat_file(file_path, dtype):
    with open(file_path, 'rb') as f:
        tensor_data = np.fromfile(f, dtype=dtype)
    loaded_tensor = torch.tensor(tensor_data, device=device)

    # If dtype was mapped to float32 for bfloat16 compatibility, convert back
    if dtype == np.float32 and "bfloat16" in file_path:
        loaded_tensor = loaded_tensor.to(torch.bfloat16)
    return loaded_tensor

# Optimized weight loading function to load weights once for the shared model
def load_offloaded_weights(model, weights_dir):
    logging.info("Loading weights for the shared LLaMA model.")
    # Load weights for the first model instance
    first_model = model.shared_model.models[0]
    for name, param in first_model.named_parameters():
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

            # Load tensor directly to GPU and copy it to the model's parameters
            loaded_tensor = load_dat_file(file_path, expected_dtype).view_as(param)
            if param.dtype == torch.bfloat16:
                loaded_tensor = loaded_tensor.to(torch.bfloat16)

            param.data.copy_(loaded_tensor)
            logging.info(f"Loaded weights for parameter: {name}")
        else:
            logging.warning(f"Warning: {file_name} not found in offloaded directory.")

    # Share the loaded weights across all other model instances
    for i, model_copy in enumerate(model.shared_model.models[1:], start=1):
        model_copy.load_state_dict(first_model.state_dict())
        logging.info(f"Shared weights loaded for model layer {i}")

# ResponseQualityManager class remains unchanged
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

# Function to generate responses with optimized memory usage
def generate_response(input_text, model, tokenizer, config, max_new_tokens=150, history=[], context_limit=512):
    history = [line for line in history if line.strip()]  # Clean the history
    prompt = f"{' '.join(history[-3:])}\nUser: {input_text}\n" if history else f"User: {input_text}\n"

    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=context_limit)

    # Move tensors to device, ensuring input_ids are long type
    input_ids = inputs["input_ids"].to(device).long()  # Ensure long type
    attention_mask = inputs["attention_mask"].to(device)

    # Check input_ids are within vocab_size
    if (input_ids >= config.vocab_size).any():
        invalid_ids = input_ids[input_ids >= config.vocab_size]
        raise ValueError(f"Out-of-bounds input_ids found: {invalid_ids}")

    # Truncate input if necessary
    max_length = config.max_position_embeddings
    if input_ids.shape[1] > max_length:
        input_ids = input_ids[:, :max_length]
        attention_mask = attention_mask[:, :max_length]

    with torch.no_grad():
        # Call the model with the proper input format
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        output_ids = torch.argmax(outputs, dim=-1)

    response = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
    cleaned_response = re.sub(r'\s+', ' ', response.split("User:")[-1].strip())
    history.append(f"User: {input_text}\nModel: {cleaned_response}")

    if len(history) > 6:
        history = history[-6:]

    return cleaned_response, history

# Interactive query loop with refined response generation
def user_input_loop(model, tokenizer, config):
    print("\n--- LLaMA Instruct Model Interactive Query ---")
    print("Type 'exit' to quit.")
    history = []  # Initialize a history buffer to keep track of conversation
    while True:
        try:
            user_input = input("\nEnter your query: ")
            if user_input.lower() == 'exit':
                print("Exiting...")
                break
            response, history = generate_response(user_input, model, tokenizer, config, history=history)
            print(f"Model Response: {response}")
        except Exception as e:
            logging.error(f"An error occurred: {e}")
            torch.cuda.empty_cache()
            raise e  # Reraise the exception to see full crash details

# Main execution flow
if __name__ == "__main__":
    try:
        # Load and prepare tokenizer configuration
        config = load_configuration(MODEL_JSON_PATH)
        prepare_tokenizer_config(TOKENIZER_CONFIG_PATH, config.vocab_size)

        # Load tokenizer
        logging.info("Loading tokenizer...")
        tokenizer = load_tokenizer_with_model_config(MODEL_JSON_PATH)

        # Initialize the optimized model
        logging.info("Initializing the optimized Stacked LLaMA Network.")
        model = OptimizedStackedLlamaNetwork(config, num_stacks=3).to(device)

        # Load weights and move to GPU
        logging.info("Loading offloaded weights into the model.")
        load_offloaded_weights(model, WEIGHTS_DIR)
        model.to(device)
        model.eval()

        # Log tokenizer details
        logging.info(f"Tokenizer length (len(tokenizer)): {len(tokenizer)}")
        logging.info(f"Tokenizer vocab_size: {tokenizer.vocab_size}")

        # Initialize ResponseQualityManager
        quality_manager = ResponseQualityManager(model, tokenizer)

        # Start the interactive query loop
        logging.info("Optimized model loaded successfully. You can now query the model.")
        user_input_loop(model, tokenizer, config)
    except Exception as main_e:
        logging.error(f"Failed to initialize the model: {main_e}")
        raise main_e
