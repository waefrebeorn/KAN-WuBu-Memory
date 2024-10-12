import os
import torch
import torch.nn as nn
import torch.nn.functional as F  
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
    
    d = xq.shape[-1]
    
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    
    batch_size, num_heads, seq_len, _ = xq_.shape

    if freqs_cis.dim() == 3:
        freqs_cis = freqs_cis.squeeze(0)
    
    freqs_cis = freqs_cis[:seq_len, :d//2]
    
    freqs_cis = freqs_cis.to(xq_.device)
    freqs_cis = freqs_cis.unsqueeze(0).unsqueeze(0)
    freqs_cis = freqs_cis.expand(batch_size, num_heads, seq_len, d//2)
    
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    
    return xq_out.type_as(xq), xk_out.type_as(xk)







# Generating scaled rotary frequencies for LLaMA 3.2
def get_rotary_frequencies(config):
    hidden_size = config.hidden_size
    max_position_embeddings = config.max_position_embeddings
    base = config.rope_theta
    scaling_factor = config.rope_scaling['factor']
    
    inv_freq = 1.0 / (base ** (torch.arange(0, hidden_size, 2).float() / hidden_size))
    t = torch.arange(max_position_embeddings, device=inv_freq.device)
    freqs = torch.outer(t, inv_freq) * scaling_factor
    return torch.polar(torch.ones_like(freqs), freqs)



# Custom Attention Layer that applies rotary embeddings and processes attention
class CustomAttentionLayer(nn.Module):
    def __init__(self, config, layer_index, weights_dir):
        super(CustomAttentionLayer, self).__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.weights_dir = weights_dir
        self.layer_index = layer_index

        # Create nn.Linear layers
        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

        # Load weights into the layers
        self.load_weights()

        self.scale = 1 / (self.head_dim ** 0.5)

    def load_weights(self):
        self.q_proj.weight.data = self.load_weight(f"model_layers_{self.layer_index}_self_attn_q_proj_weight.dat", (self.hidden_size, self.hidden_size))
        self.k_proj.weight.data = self.load_weight(f"model_layers_{self.layer_index}_self_attn_k_proj_weight.dat", (self.num_key_value_heads * self.head_dim, self.hidden_size))
        self.v_proj.weight.data = self.load_weight(f"model_layers_{self.layer_index}_self_attn_v_proj_weight.dat", (self.num_key_value_heads * self.head_dim, self.hidden_size))
        self.o_proj.weight.data = self.load_weight(f"model_layers_{self.layer_index}_self_attn_o_proj_weight.dat", (self.hidden_size, self.hidden_size))

    def load_weight(self, file_name, shape):
        file_path = os.path.join(self.weights_dir, file_name)
        if os.path.exists(file_path):
            tensor_data = np.fromfile(file_path, dtype=np.float32)
            return torch.tensor(tensor_data).view(*shape).to("cuda")
        else:
            raise FileNotFoundError(f"Weight file {file_name} not found.")

    def forward(self, hidden_states, freqs_cis, past_key_value=None, position_ids=None):
        # Ensure hidden_states are on the same device as model parameters (GPU)
        device = self.q_proj.weight.device
        hidden_states = hidden_states.to(device)
    
        batch_size, seq_length, _ = hidden_states.shape
    
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
    
        q = q.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_length, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_length, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    
        # Repeat k and v for multi-query attention
        k = k.repeat_interleave(self.num_heads // self.num_key_value_heads, dim=1)
        v = v.repeat_interleave(self.num_heads // self.num_key_value_heads, dim=1)
    
        q_rot, k_rot = apply_rotary_emb(q, k, freqs_cis)
    
        if past_key_value is not None:
            past_k, past_v = past_key_value
            if past_k is not None and past_v is not None:
                k_rot = torch.cat([past_k, k_rot], dim=2)
                v = torch.cat([past_v, v], dim=2)
    
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            q_rot, k_rot, v, attn_mask=None, dropout_p=0.0, is_causal=True
        )
    
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(batch_size, seq_length, self.hidden_size)
        attn_output = self.o_proj(attn_output)
    
        return attn_output, (k_rot, v)
    
        
        
class CustomTransformerLayer(nn.Module):
    def __init__(self, config, layer_index, weights_dir):
        super(CustomTransformerLayer, self).__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.layer_index = layer_index
        self.weights_dir = weights_dir

        # Attention
        self.attention = CustomAttentionLayer(config, layer_index, weights_dir)
        
        # Layer norms
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # Feed-forward network
        self.mlp = MLP(
            gate_proj=self.load_weight(f"model_layers_{layer_index}_mlp_gate_proj_weight.dat", (self.intermediate_size, self.hidden_size)),
            up_proj=self.load_weight(f"model_layers_{layer_index}_mlp_up_proj_weight.dat", (self.intermediate_size, self.hidden_size)),
            down_proj=self.load_weight(f"model_layers_{layer_index}_mlp_down_proj_weight.dat", (self.hidden_size, self.intermediate_size)),
            act_fn=F.silu
        )

    def load_weight(self, file_name, shape):
        file_path = os.path.join(self.weights_dir, file_name)
        if os.path.exists(file_path):
            tensor_data = np.fromfile(file_path, dtype=np.float32)
            return torch.tensor(tensor_data).view(*shape).to("cuda")
        else:
            raise FileNotFoundError(f"Weight file {file_name} not found.")

    def forward(self, hidden_states, freqs_cis, past_key_value=None, position_ids=None, use_cache=False):
        # Ensure hidden_states are on the same device as model parameters (GPU)
        device = self.attention.q_proj.weight.device
        hidden_states = hidden_states.to(device)
    
        # Pre-attention norm
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
    
        # Attention
        attention_output, new_past = self.attention(hidden_states, freqs_cis, past_key_value, position_ids)
    
        # Ensure residual and attention_output are on the same device
        attention_output = attention_output.to(residual.device)
    
        # Residual connection
        hidden_states = residual + attention_output
    
        # Pre-FFN norm
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
    
        # Feed-forward network
        hidden_states = self.mlp(hidden_states)
    
        # Residual connection
        hidden_states = residual + hidden_states
    
        if use_cache:
            return hidden_states, new_past
        else:
            return hidden_states, None
    

class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, hidden_states):
        # Ensure hidden_states and self.weight are on the same device
        device = self.weight.device
        hidden_states = hidden_states.to(device)

        # Compute the variance and apply normalization
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        return self.weight * hidden_states


class MLP(nn.Module):
    def __init__(self, gate_proj, up_proj, down_proj, act_fn):
        super().__init__()
        self.gate_proj = nn.Parameter(gate_proj)  # We keep these as nn.Parameters (tensors)
        self.up_proj = nn.Parameter(up_proj)
        self.down_proj = nn.Parameter(down_proj)
        self.act_fn = act_fn

    def forward(self, x):
        # Ensure input x is on the same device as the model parameters
        device = self.gate_proj.device
        x = x.to(device)

        # Perform the MLP computation
        gate_out = torch.matmul(x, self.gate_proj.T)  # Matrix multiplication with gate_proj
        up_out = torch.matmul(x, self.up_proj.T)      # Matrix multiplication with up_proj
        activated_out = self.act_fn(gate_out)         # Apply activation function

        # Perform element-wise multiplication and apply down_proj
        output = torch.matmul(activated_out * up_out, self.down_proj.T)

        return output




# CustomLlamaModel that integrates custom transformer layers and rotary embeddings
class CustomLlamaModel(LlamaForCausalLM):
    def __init__(self, config, weights_dir):
        super(CustomLlamaModel, self).__init__(config)
        self.weights_dir = weights_dir
        self.config = config

        self.transformer_layers = nn.ModuleList(
            [CustomTransformerLayer(config, layer_index, weights_dir)
             for layer_index in range(config.num_hidden_layers)]
        )
        
        self.freqs_cis = get_rotary_frequencies(config)

    def forward(self, input_ids=None, attention_mask=None, inputs_embeds=None, position_ids=None, past_key_values=None, use_cache=False, cache_position=None, return_dict=True):
        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)
    
        batch_size, seq_length = input_ids.shape if input_ids is not None else inputs_embeds.shape[:2]
    
        if position_ids is None:
            if cache_position is not None:
                position_ids = torch.arange(cache_position, cache_position + seq_length, dtype=torch.long, device=inputs_embeds.device)
                position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
            else:
                position_ids = torch.arange(seq_length, dtype=torch.long, device=inputs_embeds.device).unsqueeze(0).expand(batch_size, -1)
    
        # Initialize past_key_values if not provided
        if past_key_values is None:
            past_key_values = [None] * self.num_hidden_layers
    
        hidden_states = inputs_embeds
        presents = [] if use_cache else None
    
        for i, layer in enumerate(self.transformer_layers):
            # Ensure that past_key_values are properly handled for each layer
            layer_past = past_key_values[i] if past_key_values is not None and len(past_key_values) > i else None
            hidden_states, past = layer(hidden_states, self.freqs_cis, layer_past, position_ids, use_cache)
    
            if use_cache:
                presents.append(past)
    
        # Ensure hidden_states are on the same device as lm_head
        hidden_states = hidden_states.to(self.lm_head.weight.device)
        
        # Apply the lm_head (final linear layer)
        logits = self.lm_head(hidden_states)
    
        # Return output according to the return_dict flag
        if return_dict:
            return {"logits": logits, "past_key_values": presents if use_cache else None}
        else:
            return logits, presents if use_cache else logits









# Updated generation logic to ensure inputs are on the correct device
def generate_response(input_text, model, tokenizer, max_new_tokens=150, pad_token_id=128001, history=[], context_limit=512):
    prompt = f"{' '.join(history[-3:])}\nUser: {input_text}\n" if history else f"User: {input_text}\n"
    
    # Tokenize the input prompt
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=context_limit)

    # Move inputs to the correct device (GPU)
    device = next(model.parameters()).device
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        # Use the model's generate method with proper handling for past_key_values
        outputs = model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_k=50,
            top_p=0.9,
            repetition_penalty=1.2,
            pad_token_id=pad_token_id,
            use_cache=True  # Ensure cache is enabled for the model to handle past_key_values
        )

    # Decode the generated output
    response = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

    # Clean up the response to remove duplicate User tags or extraneous whitespace
    cleaned_response = response.split("User:")[-1].strip()
    cleaned_response = re.sub(r'\s+', ' ', cleaned_response)

    # Append this conversation turn to the history
    history.append(f"User: {input_text}\nModel: {cleaned_response}")

    # Trim the history to the last 6 conversation turns
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
