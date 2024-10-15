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

# Custom module for multiple stacked LLaMA layers (equivalent to 6x Mamba2 in NVIDIA presentation)
class StackedLlamaModule(nn.Module):
    def __init__(self, config, num_layers=6):
        super(StackedLlamaModule, self).__init__()
        self.layers = nn.ModuleList([LlamaForCausalLM(config) for _ in range(num_layers)])  # Mimicking 6x Mamba2

    def forward(self, input_ids, attention_mask=None):
        x = input_ids
        for layer in self.layers:
            outputs = layer(input_ids=x, attention_mask=attention_mask)
            x = outputs.logits
        return x

# Define shared components (e.g., Shared1 and Shared2) used in the modular structure
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

# Define Low-Rank Adaptation (LoRA) for efficient fine-tuning
class LoRA(nn.Module):
    def __init__(self, hidden_size, rank=8):
        super(LoRA, self).__init__()
        self.rank = rank
        self.lora_A = nn.Linear(hidden_size, rank, bias=False)
        self.lora_B = nn.Linear(rank, hidden_size, bias=False)

    def forward(self, x):
        return x + self.lora_B(self.lora_A(x))

# Complete Stacked LLaMA model with shared components, stacking, and LoRA
class StackedLlamaNetwork(nn.Module):
    def __init__(self, config, shared1, shared2, num_stacks=3):
        super(StackedLlamaNetwork, self).__init__()
        self.blocks = nn.ModuleList()
        
        for i in range(num_stacks):
            specialization = "early" if i == 0 else "mid" if i == 1 else "late"
            self.blocks.append(
                nn.ModuleDict({
                    "transformer_block": StackedLlamaModule(config),  # Equivalent to 6x Mamba2
                    "linear": nn.Linear(config.hidden_size, config.hidden_size),
                    "shared": shared1 if i % 2 == 0 else shared2,  # Alternating shared layers
                    "lora_adapter": LoRA(config.hidden_size)  # Optional LoRA for fine-tuning
                })
            )

    def forward(self, input_ids, attention_mask=None):
        x = input_ids
        intermediate_outputs = []

        for block in self.blocks:
            x = block["transformer_block"](x, attention_mask)
            x = block["linear"](x)
            x = block["shared"](x)
            x = block["lora_adapter"](x)
            intermediate_outputs.append(x)

        # Concatenation of intermediate outputs (mimicking 'cat' operation in the image)
        x = torch.cat(intermediate_outputs, dim=-1)
        
        return x

# Load the offloaded weights from the `.dat` files
def load_dat_file(file_path, dtype):
    with open(file_path, 'rb') as f:
        tensor_data = np.fromfile(f, dtype=dtype)
    loaded_tensor = torch.tensor(tensor_data)
    
    # If dtype was mapped to float32 for bfloat16 compatibility, convert back
    if dtype == np.float32 and "bfloat16" in file_path:
        loaded_tensor = loaded_tensor.to(torch.bfloat16)
    return loaded_tensor

def load_offloaded_weights(stacked_model, weights_dir):
    for i, llama_model in enumerate(stacked_model.blocks):
        logging.info(f"Loading weights for LLaMA stack {i + 1}")
        for name, param in llama_model["transformer_block"].layers.named_parameters():
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
shared1 = SharedLayer(config.hidden_size)
shared2 = SharedLayer(config.hidden_size)
num_stacks = 3  # Number of stacked LLaMA instances
model = StackedLlamaNetwork(config, shared1, shared2, num_stacks=num_stacks)
load_offloaded_weights(model, WEIGHTS_DIR)

# Move the model to GPU for inference
model.to('cuda')
model.eval()

# Load the tokenizer for LLaMA
logging.info(f"Loading tokenizer from directory: {SOURCE_DIR}")
tokenizer = load_tokenizer(SOURCE_DIR)

# ResponseQualityManager class for evaluating and improving responses
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

# Quality Manager instance for response evaluation
quality_manager = ResponseQualityManager(model, tokenizer)

# Updated generation logic to handle context better and avoid repetitive responses
def generate_response(input_text, model, tokenizer, max_new_tokens=150, pad_token_id=128001, history=[], context_limit=512):
    history = [line for line in history if line.strip()]  # Clean the history
    prompt = f"{' '.join(history[-3:])}\nUser: {input_text}\n" if history else f"User: {input_text}\n"
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=context_limit).to("cuda")
    
    with torch.no_grad():
        outputs = model(inputs["input_ids"], attention_mask=inputs["attention_mask"])
        output_ids = torch.argmax(outputs, dim=-1)

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
        user_input = input("\nEnter your query: ")
        if user_input.lower() == 'exit':
            print("Exiting...")
            break
        response, history = generate_response(user_input, model, tokenizer, history=history)
        print(f"Model Response: {response}")

# Start the interactive query loop
logging.info("Model loaded successfully. You can now query the model.")
user_input_loop(model, tokenizer)
