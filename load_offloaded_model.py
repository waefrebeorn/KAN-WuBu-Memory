import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import numpy as np
import re
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import LlamaForCausalLM, AutoTokenizer, LlamaConfig
from torch.utils.checkpoint import checkpoint

# Enable CUDA launch blocking for accurate error reporting
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# Define paths to the directories and files
SOURCE_DIR = "models/Llama_32_1B/"
WEIGHTS_DIR = os.path.join(SOURCE_DIR, "offload")
MODEL_JSON_PATH = os.path.join(SOURCE_DIR, "config.json")

# Initialize logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(message)s')

# Check CUDA availability
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available. Please ensure that a compatible GPU is installed and CUDA is properly configured.")

device = torch.device('cuda')

# Load the configuration from the JSON file
def load_configuration(model_json_path):
    with open(model_json_path, "r") as f:
        config_data = json.load(f)
    config = LlamaConfig(**config_data)
    return config

# Load configuration before the tokenizer
config = load_configuration(MODEL_JSON_PATH)

# Load tokenizer with proper handling of the pad token
def load_tokenizer(source_dir):
    tokenizer = AutoTokenizer.from_pretrained(source_dir)

    # Use the predefined finetune padding token ID
    predefined_pad_token_id = 128004  # <|finetune_right_pad_id|>

    if tokenizer.pad_token_id is None:
        # Set the pad_token_id to the predefined padding token
        tokenizer.pad_token_id = predefined_pad_token_id
        tokenizer.pad_token = tokenizer.convert_ids_to_tokens(predefined_pad_token_id)
        logging.info(f"Set pad_token_id to predefined finetune padding ID {predefined_pad_token_id}.")
    else:
        logging.info(f"Tokenizer already has a pad token with ID {tokenizer.pad_token_id}.")

    # Ensure that the pad_token_id is valid
    if tokenizer.pad_token_id >= tokenizer.vocab_size:
        raise ValueError(f"pad_token_id {tokenizer.pad_token_id} is out of bounds for the tokenizer's vocabulary size {tokenizer.vocab_size}.")

    pad_token = tokenizer.convert_ids_to_tokens(tokenizer.pad_token_id)
    if pad_token is None:
        raise ValueError(f"pad_token_id {tokenizer.pad_token_id} does not correspond to any token in the tokenizer.")

    tokenizer.pad_token = pad_token
    tokenizer.pad_token_id = tokenizer.pad_token_id

    return tokenizer


# Load the tokenizer
logging.info(f"Loading tokenizer from directory: {SOURCE_DIR}")
tokenizer = load_tokenizer(SOURCE_DIR)

# Log tokenizer details
logging.info(f"Tokenizer length (len(tokenizer)): {len(tokenizer)}")
logging.info(f"Tokenizer vocab_size: {tokenizer.vocab_size}")

# Ensure that the tokenizer's vocab_size matches the model's config.vocab_size
if tokenizer.vocab_size != config.vocab_size:
    logging.error(f"Tokenizer vocabulary size ({tokenizer.vocab_size}) does not match the model's config vocab_size ({config.vocab_size}). Please ensure they are aligned.")
    raise ValueError(f"Tokenizer vocabulary size ({tokenizer.vocab_size}) does not match the model's config vocab_size ({config.vocab_size}). Please ensure they are aligned.")

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
        self.shared_model = LlamaForCausalLM(config).to(device)
        self.num_layers = num_layers

    def forward_pass(self, input_ids, attention_mask):
        return self.shared_model(input_ids=input_ids, attention_mask=attention_mask).logits

    def forward(self, input_ids, attention_mask=None):
        def forward_pass_fn(x):
            return self.forward_pass(x, attention_mask)

        x = input_ids
        for _ in range(self.num_layers):
            # Ensure x is LongTensor before passing to the model
            if not torch.is_tensor(x):
                raise ValueError("Input to shared_model must be a tensor.")
            if x.dtype != torch.long:
                logging.warning(f"Converting input tensor from {x.dtype} to torch.long")
                x = x.long()
            # Use use_reentrant=False to handle future PyTorch changes
            x = checkpoint(forward_pass_fn, x, use_reentrant=False)
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
    logging.info("Loading weights for the shared LLaMA model")
    for name, param in model.shared_model.shared_model.named_parameters():
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
        else:
            logging.warning(f"Warning: {file_name} not found in offloaded directory.")

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
def generate_response(input_text, model, tokenizer, max_new_tokens=150, history=[], context_limit=512):
    history = [line for line in history if line.strip()]  # Clean the history
    prompt = f"{' '.join(history[-3:])}\nUser: {input_text}\n" if history else f"User: {input_text}\n"

    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=context_limit)

    # Move tensors to device, preserving data types
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    # Ensure that input IDs are torch.long
    if input_ids.dtype != torch.long:
        logging.warning(f"Converting input tensor from {input_ids.dtype} to torch.long")
        input_ids = input_ids.long()

    # Check input_ids are within vocab_size
    if (input_ids >= config.vocab_size).any():
        invalid_ids = input_ids[input_ids >= config.vocab_size]
        raise ValueError(f"Out-of-bounds input_ids found: {invalid_ids}")

    with torch.no_grad():
        # Call the model with the proper input format
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        output_ids = torch.argmax(outputs.logits, dim=-1)

    response = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
    cleaned_response = re.sub(r'\s+', ' ', response.split("User:")[-1].strip())
    history.append(f"User: {input_text}\nModel: {cleaned_response}")

    if len(history) > 6:
        history = history[-6:]

    return cleaned_response, history

# Interactive query loop with refined response generation
def user_input_loop(model, tokenizer):
    print("\n--- LLaMA Instruct Model Interactive Query ---")
    print("Type 'exit' to quit.")
    history = []  # Initialize a history buffer to keep track of conversation
    while True:
        try:
            user_input = input("\nEnter your query: ")
            if user_input.lower() == 'exit':
                print("Exiting...")
                break
            response, history = generate_response(user_input, model, tokenizer, history=history)
            print(f"Model Response: {response}")
        except Exception as e:
            logging.error(f"An error occurred: {e}")
            torch.cuda.empty_cache()
            raise e  # Reraise the exception to see full crash details

# Main execution flow
if __name__ == "__main__":
    try:
        # Initialize the optimized model
        logging.info("Initializing the optimized Stacked LLaMA Network.")
        model = OptimizedStackedLlamaNetwork(config, num_stacks=3).to(device)

        # Load weights and move to GPU
        logging.info("Loading offloaded weights into the model.")
        load_offloaded_weights(model, WEIGHTS_DIR)
        model.to(device)
        model.eval()

        # Load tokenizer
        logging.info(f"Loading tokenizer from directory: {SOURCE_DIR}")
        tokenizer = load_tokenizer(SOURCE_DIR)

        # Log tokenizer details
        logging.info(f"Tokenizer length (len(tokenizer)): {len(tokenizer)}")
        logging.info(f"Tokenizer vocab_size: {tokenizer.vocab_size}")

        # Ensure that the tokenizer's vocab_size matches the model's config.vocab_size
        if tokenizer.vocab_size != config.vocab_size:
            raise ValueError(f"Tokenizer vocabulary size ({tokenizer.vocab_size}) does not match the model's config vocab_size ({config.vocab_size}). Please ensure they are aligned.")

        # Resize token embeddings in case new tokens were added
        model.shared_model.shared_model.resize_token_embeddings(tokenizer.vocab_size)

        # Initialize ResponseQualityManager
        quality_manager = ResponseQualityManager(model, tokenizer)

        # Start the interactive query loop
        logging.info("Optimized model loaded successfully. You can now query the model.")
        user_input_loop(model, tokenizer)
    except Exception as main_e:
        logging.error(f"Failed to initialize the model: {main_e}")
        raise main_e
