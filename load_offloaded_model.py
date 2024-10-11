import os
import torch
import torch.nn as nn
import numpy as np
import re
import logging
from transformers import LlamaForCausalLM, AutoTokenizer, LlamaConfig
from typing import Tuple
import json

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

# Rotary embedding application with frequency scaling
def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    d_q, d_k = xq.shape[-1] // 2, xk.shape[-1] // 2
    xq_ = torch.complex(xq[..., :d_q], xq[..., d_q:])
    xk_ = torch.complex(xk[..., :d_k], xk[..., d_k:])
    
    # Apply the rotary embedding frequencies to the queries and keys
    xq_out = xq_ * freqs_cis.unsqueeze(0).unsqueeze(0)
    xk_out = xk_ * freqs_cis.unsqueeze(0).unsqueeze(0)

    return torch.view_as_real(xq_out).flatten(2), torch.view_as_real(xk_out).flatten(2)

# Generating scaled rotary frequencies for LLaMA 3.2
def get_rotary_frequencies(hidden_size, max_position_embeddings=128000):
    inv_freq = 1.0 / (10000 ** (torch.arange(0, hidden_size, 2).float() / hidden_size))
    
    # Scaling frequencies based on maximum position embeddings
    position_ids = torch.arange(0, max_position_embeddings, device=inv_freq.device).float()
    freqs = torch.einsum("i,j->ij", position_ids, inv_freq)
    
    # Convert to complex numbers for cosine and sine components
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # Convert into complex
    return freqs_cis

# Custom Attention Layer that applies rotary embeddings and processes attention
class CustomAttentionLayer(nn.Module):
    def __init__(self, hidden_size, num_heads, layer_index, weights_dir):
        super(CustomAttentionLayer, self).__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.weights_dir = weights_dir
        self.layer_index = layer_index
        self.query_weight = self.load_weight(f"model_layers_{layer_index}_self_attn_q_proj_weight.dat")
        self.key_weight = self.load_weight(f"model_layers_{layer_index}_self_attn_k_proj_weight.dat")
        self.value_weight = self.load_weight(f"model_layers_{layer_index}_self_attn_v_proj_weight.dat")
        self.output_weight = self.load_weight(f"model_layers_{layer_index}_self_attn_o_proj_weight.dat")
        self.scale = 1 / (hidden_size // num_heads) ** 0.5

    def load_weight(self, file_name):
        file_path = os.path.join(self.weights_dir, file_name)
        if os.path.exists(file_path):
            tensor_data = np.fromfile(file_path, dtype=np.float32)
            return torch.tensor(tensor_data).view(-1, self.hidden_size).to("cuda")
        else:
            raise FileNotFoundError(f"Weight file {file_name} not found.")

    def forward(self, hidden_states, freqs_cis, past_key_value=None, position_ids=None):
        q = torch.matmul(hidden_states, self.query_weight.T)
        k = torch.matmul(hidden_states, self.key_weight.T)
        v = torch.matmul(hidden_states, self.value_weight.T)

        # Apply rotary embeddings using position_ids
        q_rot, k_rot = apply_rotary_emb(q, k, freqs_cis[position_ids])

        # Handle past_key_values
        if past_key_value is not None:
            past_k, past_v = past_key_value
            k_rot = torch.cat([past_k, k_rot], dim=-2)
            v = torch.cat([past_v, v], dim=-2)

        attention_scores = torch.matmul(q_rot, k_rot.transpose(-1, -2)) * self.scale
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        output = torch.matmul(attention_probs, v)

        output = torch.matmul(output, self.output_weight.T)
        past = (k_rot, v)
        return output, past

# Modify the model's transformer layer to use the custom attention layer
class CustomTransformerLayer(nn.Module):
    def __init__(self, hidden_size, num_heads, layer_index, weights_dir):
        super(CustomTransformerLayer, self).__init__()
        self.attention = CustomAttentionLayer(hidden_size, num_heads, layer_index, weights_dir)
        self.layernorm = nn.LayerNorm(hidden_size)

    def forward(self, hidden_states, freqs_cis, past_key_value=None, position_ids=None, cache_position=None):
        if cache_position is not None:
            hidden_states = hidden_states[:, cache_position:]

        attention_output, past = self.attention(hidden_states, freqs_cis, past_key_value, position_ids)
        return self.layernorm(attention_output), past

# CustomLlamaModel that integrates custom transformer layers and rotary embeddings
class CustomLlamaModel(LlamaForCausalLM):
    def __init__(self, config, weights_dir):
        super(CustomLlamaModel, self).__init__(config)
        self.weights_dir = weights_dir
        self.num_hidden_layers = config.num_hidden_layers
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads

        # Replace the model's transformer layers with custom transformer layers
        self.transformer_layers = nn.ModuleList(
            [CustomTransformerLayer(self.hidden_size, self.num_attention_heads, layer_index, weights_dir)
             for layer_index in range(self.num_hidden_layers)]
        )
        
        # Generate rotary frequencies
        self.freqs_cis = get_rotary_frequencies(self.hidden_size)

    def forward(self, input_ids=None, attention_mask=None, inputs_embeds=None, position_ids=None, past_key_values=None, cache_position=None, use_cache=False, return_dict=True):
        # Get the embedding layer from the parent class (LlamaForCausalLM)
        if inputs_embeds is not None:
            hidden_states = inputs_embeds
        else:
            hidden_states = self.get_input_embeddings()(input_ids)

        # Ensure cache_position is handled properly and is an integer
        if position_ids is not None:
            if cache_position is not None:
                if isinstance(cache_position, torch.Tensor):
                    if cache_position.numel() > 1:
                        logging.warning("cache_position tensor has more than one element; using the first element.")
                        cache_position = cache_position[0].item()  # Use the first element
                    else:
                        cache_position = cache_position.item()  # Convert single-element tensor to Python integer
                position_ids = position_ids[:, cache_position:]
        else:
            position_ids = torch.arange(hidden_states.shape[1], device=hidden_states.device).unsqueeze(0)

        if past_key_values is None:
            past_key_values = [None] * len(self.transformer_layers)

        next_past_key_values = []
        for i, layer in enumerate(self.transformer_layers):
            hidden_states, past = layer(hidden_states, self.freqs_cis, past_key_values[i], position_ids, cache_position)
            next_past_key_values.append(past)

        logits = self.lm_head(hidden_states)

        if return_dict:
            return {"logits": logits, "past_key_values": next_past_key_values if use_cache else None}
        else:
            return logits, next_past_key_values if use_cache else None





# Updated generation logic to ensure inputs are on the correct device
def generate_response(input_text, model, tokenizer, max_new_tokens=150, pad_token_id=128001, history=[], context_limit=512):
    # Clean the history to avoid redundant prompts
    history = [line for line in history if line.strip()]  # Remove empty lines
    
    # Create a simplified context prompt from the last few exchanges
    prompt = f"{' '.join(history[-3:])}\nUser: {input_text}\n" if history else f"User: {input_text}\n"
    
    # Prepare inputs for the model
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=context_limit)
    
    # Move inputs to the same device as the model
    device = next(model.parameters()).device
    inputs = {key: value.to(device) for key, value in inputs.items()}
    
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

# Interactive input loop to query the model
def user_input_loop(custom_model, tokenizer):
    print("\n--- Custom LLaMA 3.2 Instruct Model ---")
    print("Type 'exit' to quit.")
    history = []  # Initialize a history buffer to keep track of conversation
    while True:
        user_input = input("\nEnter your query: ")
        if user_input.lower() == 'exit':
            print("Exiting...")
            break
        response, history = generate_response(user_input, custom_model, tokenizer, history=history)
        print(f"Model Response: {response}")

# Initialize the custom model and tokenizer
config = load_configuration(MODEL_JSON_PATH)
tokenizer = load_tokenizer(SOURCE_DIR)
custom_model = CustomLlamaModel(config, WEIGHTS_DIR)

# Start the user input loop
user_input_loop(custom_model, tokenizer)
