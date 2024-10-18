import os
import torch
import torch.nn as nn
import json
import numpy as np
import re
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import LlamaForCausalLM, AutoTokenizer, LlamaConfig, PreTrainedTokenizerFast

import copy
from torch.utils.checkpoint import checkpoint

# Enable CUDA launch blocking for accurate error reporting
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# Define paths to the directories and files
SOURCE_DIR = r"C:\Projects\KAN-WuBu-Memory\models\Llama_32_1B"
WEIGHTS_DIR = os.path.join(SOURCE_DIR, "offload")
MODEL_JSON_PATH = os.path.join(SOURCE_DIR, "config.json")
TOKENIZER_CONFIG_PATH = os.path.join(SOURCE_DIR, "tokenizer_config.json")
SPECIAL_TOKENS_MAP_PATH = os.path.join(SOURCE_DIR, "special_tokens_map.json")
GENERATION_CONFIG_PATH = os.path.join(SOURCE_DIR, "generation_config.json")
TOKENIZER_JSON_PATH = os.path.join(SOURCE_DIR, "tokenizer.json")

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

# Function to load special tokens from special_tokens_map.json
def load_special_tokens(special_tokens_map_path):
    with open(special_tokens_map_path, "r", encoding='utf-8') as f:
        special_tokens_map = json.load(f)
    
    # Extract special tokens
    special_tokens = {
        'bos_token': special_tokens_map.get("bos_token", "<|begin_of_text|>"),
        'eos_token': special_tokens_map.get("eos_token", "<|end_of_text|>"),
        'pad_token': special_tokens_map.get("pad_token", "<|finetune_right_pad_id|>"),
    }
    
    return special_tokens

# Function to load tokenizer and add special tokens
def load_tokenizer_with_special_tokens(source_dir, tokenizer_json_path, special_tokens_map_path, model_config):
    # Load tokenizer using from_pretrained
    tokenizer = PreTrainedTokenizerFast.from_pretrained(
        source_dir,
        tokenizer_file=tokenizer_json_path,
        bos_token="<|begin_of_text|>",
        eos_token="<|end_of_text|>",
        pad_token="<|finetune_right_pad_id|>",
    )
    
    # Load special tokens
    special_tokens = load_special_tokens(special_tokens_map_path)
    
    # Define additional special tokens
    additional_special_tokens = [
        "<|reserved_special_token_0|>",
        "<|reserved_special_token_1|>",
        "<|reserved_special_token_2|>",
        "<|reserved_special_token_3|>",
        "<|reserved_special_token_4|>",
        "<|reserved_special_token_5|>",
        "<|reserved_special_token_6|>",
        "<|reserved_special_token_7|>",
        "<|reserved_special_token_8|>",
        "<|reserved_special_token_9|>",
        "<|reserved_special_token_10|>",
        "<|reserved_special_token_11|>",
        "<|reserved_special_token_12|>",
        "<|reserved_special_token_13|>",
        "<|reserved_special_token_14|>",
        "<|reserved_special_token_15|>",
        "<|reserved_special_token_16|>",
        "<|reserved_special_token_17|>",
        "<|reserved_special_token_18|>",
        "<|reserved_special_token_19|>",
        "<|reserved_special_token_20|>",
        "<|reserved_special_token_21|>",
        "<|reserved_special_token_22|>",
        "<|reserved_special_token_23|>",
        "<|reserved_special_token_24|>",
        "<|reserved_special_token_25|>",
        "<|reserved_special_token_26|>",
        "<|reserved_special_token_27|>",
        "<|reserved_special_token_28|>",
        "<|reserved_special_token_29|>",
        "<|reserved_special_token_30|>",
        "<|reserved_special_token_31|>",
        "<|reserved_special_token_32|>",
        "<|reserved_special_token_33|>",
        "<|reserved_special_token_34|>",
        "<|reserved_special_token_35|>",
        "<|reserved_special_token_36|>",
        "<|reserved_special_token_37|>",
        "<|reserved_special_token_38|>",
        "<|reserved_special_token_39|>",
        "<|reserved_special_token_40|>",
        "<|reserved_special_token_41|>",
        "<|reserved_special_token_42|>",
        "<|reserved_special_token_43|>",
        "<|reserved_special_token_44|>",
        "<|reserved_special_token_45|>",
        "<|reserved_special_token_46|>",
        "<|reserved_special_token_47|>",
        "<|reserved_special_token_48|>",
        "<|reserved_special_token_49|>",
        "<|reserved_special_token_50|>",
        "<|reserved_special_token_51|>",
        "<|reserved_special_token_52|>",
        "<|reserved_special_token_53|>",
        "<|reserved_special_token_54|>",
        "<|reserved_special_token_55|>",
        "<|reserved_special_token_56|>",
        "<|reserved_special_token_57|>",
        "<|reserved_special_token_58|>",
        "<|reserved_special_token_59|>",
        "<|reserved_special_token_60|>",
        "<|reserved_special_token_61|>",
        "<|reserved_special_token_62|>",
        "<|reserved_special_token_63|>",
        "<|reserved_special_token_64|>",
        "<|reserved_special_token_65|>",
        "<|reserved_special_token_66|>",
        "<|reserved_special_token_67|>",
        "<|reserved_special_token_68|>",
        "<|reserved_special_token_69|>",
        "<|reserved_special_token_70|>",
        "<|reserved_special_token_71|>",
        "<|reserved_special_token_72|>",
        "<|reserved_special_token_73|>",
        "<|reserved_special_token_74|>",
        "<|reserved_special_token_75|>",
        "<|reserved_special_token_76|>",
        "<|reserved_special_token_77|>",
        "<|reserved_special_token_78|>",
        "<|reserved_special_token_79|>",
        "<|reserved_special_token_80|>",
        "<|reserved_special_token_81|>",
        "<|reserved_special_token_82|>",
        "<|reserved_special_token_83|>",
        "<|reserved_special_token_84|>",
        "<|reserved_special_token_85|>",
        "<|reserved_special_token_86|>",
        "<|reserved_special_token_87|>",
        "<|reserved_special_token_88|>",
        "<|reserved_special_token_89|>",
        "<|reserved_special_token_90|>",
        "<|reserved_special_token_91|>",
        "<|reserved_special_token_92|>",
        "<|reserved_special_token_93|>",
        "<|reserved_special_token_94|>",
        "<|reserved_special_token_95|>",
        "<|reserved_special_token_96|>",
        "<|reserved_special_token_97|>",
        "<|reserved_special_token_98|>",
        "<|reserved_special_token_99|>",
        "<|reserved_special_token_100|>",
        "<|reserved_special_token_101|>",
        "<|reserved_special_token_102|>",
        "<|reserved_special_token_103|>",
        "<|reserved_special_token_104|>",
        "<|reserved_special_token_105|>",
        "<|reserved_special_token_106|>",
        "<|reserved_special_token_107|>",
        "<|reserved_special_token_108|>",
        "<|reserved_special_token_109|>",
        "<|reserved_special_token_110|>",
        "<|reserved_special_token_111|>",
        "<|reserved_special_token_112|>",
        "<|reserved_special_token_113|>",
        "<|reserved_special_token_114|>",
        "<|reserved_special_token_115|>",
        "<|reserved_special_token_116|>",
        "<|reserved_special_token_117|>",
        "<|reserved_special_token_118|>",
        "<|reserved_special_token_119|>",
        "<|reserved_special_token_120|>",
        "<|reserved_special_token_121|>",
        "<|reserved_special_token_122|>",
        "<|reserved_special_token_123|>",
        "<|reserved_special_token_124|>",
        "<|reserved_special_token_125|>",
        "<|reserved_special_token_126|>",
        "<|reserved_special_token_127|>",
        "<|reserved_special_token_128|>",
        "<|reserved_special_token_129|>",
        "<|reserved_special_token_130|>",
        "<|reserved_special_token_131|>",
        "<|reserved_special_token_132|>",
        "<|reserved_special_token_133|>",
        "<|reserved_special_token_134|>",
        "<|reserved_special_token_135|>",
        "<|reserved_special_token_136|>",
        "<|reserved_special_token_137|>",
        "<|reserved_special_token_138|>",
        "<|reserved_special_token_139|>",
        "<|reserved_special_token_140|>",
        "<|reserved_special_token_141|>",
        "<|reserved_special_token_142|>",
        "<|reserved_special_token_143|>",
        "<|reserved_special_token_144|>",
        "<|reserved_special_token_145|>",
        "<|reserved_special_token_146|>",
        "<|reserved_special_token_147|>",
        "<|reserved_special_token_148|>",
        "<|reserved_special_token_149|>",
        "<|reserved_special_token_150|>",
        "<|reserved_special_token_151|>",
        "<|reserved_special_token_152|>",
        "<|reserved_special_token_153|>",
        "<|reserved_special_token_154|>",
        "<|reserved_special_token_155|>",
        "<|reserved_special_token_156|>",
        "<|reserved_special_token_157|>",
        "<|reserved_special_token_158|>",
        "<|reserved_special_token_159|>",
        "<|reserved_special_token_160|>",
        "<|reserved_special_token_161|>",
        "<|reserved_special_token_162|>",
        "<|reserved_special_token_163|>",
        "<|reserved_special_token_164|>",
        "<|reserved_special_token_165|>",
        "<|reserved_special_token_166|>",
        "<|reserved_special_token_167|>",
        "<|reserved_special_token_168|>",
        "<|reserved_special_token_169|>",
        "<|reserved_special_token_170|>",
        "<|reserved_special_token_171|>",
        "<|reserved_special_token_172|>",
        "<|reserved_special_token_173|>",
        "<|reserved_special_token_174|>",
        "<|reserved_special_token_175|>",
        "<|reserved_special_token_176|>",
        "<|reserved_special_token_177|>",
        "<|reserved_special_token_178|>",
        "<|reserved_special_token_179|>",
        "<|reserved_special_token_180|>",
        "<|reserved_special_token_181|>",
        "<|reserved_special_token_182|>",
        "<|reserved_special_token_183|>",
        "<|reserved_special_token_184|>",
        "<|reserved_special_token_185|>",
        "<|reserved_special_token_186|>",
        "<|reserved_special_token_187|>",
        "<|reserved_special_token_188|>",
        "<|reserved_special_token_189|>",
        "<|reserved_special_token_190|>",
        "<|reserved_special_token_191|>",
        "<|reserved_special_token_192|>",
        "<|reserved_special_token_193|>",
        "<|reserved_special_token_194|>",
        "<|reserved_special_token_195|>",
        "<|reserved_special_token_196|>",
        "<|reserved_special_token_197|>",
        "<|reserved_special_token_198|>",
        "<|reserved_special_token_199|>",
        "<|reserved_special_token_200|>",
        "<|reserved_special_token_201|>",
        "<|reserved_special_token_202|>",
        "<|reserved_special_token_203|>",
        "<|reserved_special_token_204|>",
        "<|reserved_special_token_205|>",
        "<|reserved_special_token_206|>",
        "<|reserved_special_token_207|>",
        "<|reserved_special_token_208|>",
        "<|reserved_special_token_209|>",
        "<|reserved_special_token_210|>",
        "<|reserved_special_token_211|>",
        "<|reserved_special_token_212|>",
        "<|reserved_special_token_213|>",
        "<|reserved_special_token_214|>",
        "<|reserved_special_token_215|>",
        "<|reserved_special_token_216|>",
        "<|reserved_special_token_217|>",
        "<|reserved_special_token_218|>",
        "<|reserved_special_token_219|>",
        "<|reserved_special_token_220|>",
        "<|reserved_special_token_221|>",
        "<|reserved_special_token_222|>",
        "<|reserved_special_token_223|>",
        "<|reserved_special_token_224|>",
        "<|reserved_special_token_225|>",
        "<|reserved_special_token_226|>",
        "<|reserved_special_token_227|>",
        "<|reserved_special_token_228|>",
        "<|reserved_special_token_229|>",
        "<|reserved_special_token_230|>",
        "<|reserved_special_token_231|>",
        "<|reserved_special_token_232|>",
        "<|reserved_special_token_233|>",
        "<|reserved_special_token_234|>",
        "<|reserved_special_token_235|>",
        "<|reserved_special_token_236|>",
        "<|reserved_special_token_237|>",
        "<|reserved_special_token_238|>",
        "<|reserved_special_token_239|>",
        "<|reserved_special_token_240|>",
        "<|reserved_special_token_241|>",
        "<|reserved_special_token_242|>",
        "<|reserved_special_token_243|>",
        "<|reserved_special_token_244|>",
        "<|reserved_special_token_245|>",
        "<|reserved_special_token_246|>",
        "<|reserved_special_token_247|>",
    ]
    
    # Add additional special tokens to the tokenizer
    tokenizer.add_special_tokens({'additional_special_tokens': additional_special_tokens})
    logging.info(f"Added {len(additional_special_tokens)} additional special tokens.")
    
    # Update vocab_size
    tokenizer.vocab_size = 128256
    logging.info(f"Tokenizer vocab_size set to: {tokenizer.vocab_size}")
    
    # Verify special tokens
    logging.info(f"BOS token ID: {tokenizer.bos_token_id}")
    logging.info(f"EOS token ID: {tokenizer.eos_token_id}")
    logging.info(f"PAD token ID: {tokenizer.pad_token_id}")
    
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
        # Load and prepare model configuration
        config = load_configuration(MODEL_JSON_PATH)
        prepare_tokenizer_config(TOKENIZER_CONFIG_PATH, config.vocab_size)

        # Load tokenizer and add special tokens
        logging.info("Loading and adding special tokens to tokenizer...")
        tokenizer = load_tokenizer_with_special_tokens(TOKENIZER_JSON_PATH, SPECIAL_TOKENS_MAP_PATH, config)

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
