import os
import torch
import json
import logging
import re
import numpy as np
from math import log2
from transformers import (
    LlamaForCausalLM,
    AutoTokenizer,
    LlamaConfig,
)
import torch.nn.functional as F
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA

# --------------------------- Configuration --------------------------- #

SOURCE_DIR = "models/Llama_32_1B/"
WEIGHTS_DIR = os.path.join(SOURCE_DIR, "offload")
MODEL_JSON_PATH = os.path.join(SOURCE_DIR, "config.json")
MAX_CONTEXT_LENGTH = 2048

LOG_FORMAT = "%(asctime)s:%(levelname)s:%(name)s: %(message)s"
LOG_LEVEL = logging.INFO

# --------------------------- Logging Setup --------------------------- #

logging.basicConfig(
    level=LOG_LEVEL,
    format=LOG_FORMAT,
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# --------------------------- Device Configuration --------------------------- #

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type != "cuda":
    logger.error("CUDA-enabled GPU not found. Please ensure a compatible GPU is available.")
    raise SystemExit("CUDA-enabled GPU not found.")

logger.info(f"Using device: {device}")

# --------------------------- Token Definitions --------------------------- #

SPECIAL_TOKEN_MAP = {
    128000: "<|begin_of_text|>",
    128001: "<|end_of_text|>",
    128002: "<|reserved_special_token_0|>",
    128003: "<|reserved_special_token_1|>",
    128004: "<|finetune_right_pad_id|>",
    128005: "<|reserved_special_token_2|>",
    128006: "<|start_header_id|>",
    128007: "<|end_header_id|>",
    128008: "<|eom_id|>",
    128009: "<|eot_id|>",
    128010: "<|python_tag|>",
    128011: "<|analytical_start|>",
    128012: "<|analytical_end|>",
    128013: "<|creative_start|>",
    128014: "<|creative_end|>",
    128015: "<|factual_start|>",
    128016: "<|factual_end|>",
}

# --------------------------- User Configuration --------------------------- #

class UserConfig:
    """Interface to set up and customize model configurations and preferences for a macroprocessor-like model."""
    def __init__(self, config_file=None):
        self.config = {
            "max_length": 2048,
            "initial_weights": {
                "entropy": 1.0,
                "varentropy": 0.5,
                "kl_div": 0.3,
                "perplexity": 0.2
            },
            "visualization_frequency": 5,
            "logging_level": "INFO",
            "interactive_visuals": True,
            "precision": "float16",  # Use float16 or bfloat16 for inference
            "kv_cache_enabled": True  # Enable smart KV-caching
        }
        if config_file:
            self.load_config(config_file)

    def load_config(self, file_path):
        """Load configuration from a JSON file."""
        with open(file_path, 'r') as f:
            self.config = json.load(f)
        logger.info(f"Configuration loaded from {file_path}")

    def save_config(self, file_path):
        """Save the current configuration to a JSON file."""
        with open(file_path, 'w') as f:
            json.dump(self.config, f, indent=4)
        logger.info(f"Configuration saved to {file_path}")

    def update_config(self, key, value):
        """Update a specific configuration setting."""
        self.config[key] = value
        logger.info(f"Configuration updated: {key} = {value}")

    def get_config(self):
        """Return the current configuration."""
        return self.config

# --------------------------- Helper Functions for Loss and Metrics --------------------------- #

def calculate_entropy(logits):
    """Calculate entropy from logits."""
    probs = F.softmax(logits, dim=-1)
    entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)
    return entropy

def calculate_varentropy(entropy):
    """Calculate varentropy from entropy values."""
    return torch.var(entropy)

def calculate_kl_divergence(logits, target_distribution=None):
    """Calculate KL divergence between logits and a target distribution."""
    probs = F.softmax(logits, dim=-1)
    if target_distribution is None:
        target_distribution = torch.ones_like(probs) / probs.size(-1)  # Uniform distribution
    kl_div = torch.sum(probs * (torch.log(probs + 1e-10) - torch.log(target_distribution + 1e-10)), dim=-1)
    return kl_div

def calculate_perplexity(logits):
    """Calculate perplexity from logits."""
    probs = F.softmax(logits, dim=-1)
    perplexity = torch.exp(-torch.sum(probs * torch.log(probs + 1e-10), dim=-1))
    return perplexity

# --------------------------- Adaptive Weighting System for Losses --------------------------- #

class AdaptiveWeightingSystem:
    """
    Dynamically adjusts the weighting of multiple losses based on previous
    token selections and their associated loss patterns.
    """
    def __init__(self, initial_weights=None):
        if initial_weights is None:
            self.weights = {'entropy': 1.0, 'varentropy': 0.5, 'kl_div': 0.3, 'perplexity': 0.2}
        else:
            self.weights = initial_weights
        self.history = {'entropy': [], 'varentropy': [], 'kl_div': [], 'perplexity': []}

    def adjust_weights(self):
        """Adjust the loss weights based on historical data to favor good token selections."""
        for loss_type in self.weights:
            if len(self.history[loss_type]) > 1:
                # If recent loss values have decreased, increase the weight for that metric
                if self.history[loss_type][-1] < self.history[loss_type][-2]:
                    self.weights[loss_type] *= 1.1  # Boost the weight
                    logger.debug(f"Boosting weight for {loss_type}: {self.weights[loss_type]:.4f}")
                else:
                    self.weights[loss_type] *= 0.9  # Reduce the weight
                    logger.debug(f"Reducing weight for {loss_type}: {self.weights[loss_type]:.4f}")
        # Normalize the weights to keep them in reasonable ranges
        total_weight = sum(self.weights.values())
        for key in self.weights:
            self.weights[key] /= total_weight
        logger.debug(f"Normalized weights: {self.weights}")

    def log_losses(self, entropy, varentropy, kl_div, perplexity):
        """Store the latest loss values to help adjust future weights."""
        self.history['entropy'].append(entropy.item())
        self.history['varentropy'].append(varentropy.item())
        self.history['kl_div'].append(kl_div.item())
        self.history['perplexity'].append(perplexity.item())
        logger.debug(f"Logged losses - Entropy: {entropy.item():.4f}, Varentropy: {varentropy.item():.4f}, KL Div: {kl_div.item():.4f}, Perplexity: {perplexity.item():.4f}")

    def get_weights(self):
        return self.weights

# --------------------------- Token Selection with Adaptive Weighting and Memory Efficiency --------------------------- #

def select_token_with_weights(logits, vertex_movements, loss_weighting_system, precision="float16"):
    """Efficient token selection using entropy, varentropy, and configurable loss weights with precision support."""
    if precision == "float16":
        logits = logits.half()  # Switch to float16 for inference speedup
        logger.debug("Switched logits to float16 precision.")
    elif precision == "bfloat16":
        logits = logits.bfloat16()  # Alternatively, use bfloat16
        logger.debug("Switched logits to bfloat16 precision.")

    # Compute multiple losses
    entropy = calculate_entropy(logits)
    varentropy = calculate_varentropy(entropy)
    kl_div = calculate_kl_divergence(logits)
    perplexity = calculate_perplexity(logits)

    # Log losses to adjust weights dynamically
    loss_weighting_system.log_losses(entropy, varentropy, kl_div, perplexity)

    # Adjust weights based on historical performance
    loss_weighting_system.adjust_weights()
    weights = loss_weighting_system.get_weights()

    # Adjust logits by the weighted sum of losses and vertex movements
    adjusted_logits = logits - (
        weights['entropy'] * entropy +
        weights['varentropy'] * varentropy +
        weights['kl_div'] * kl_div +
        weights['perplexity'] * perplexity
    ).unsqueeze(-1)

    # Apply vertex movement strategy
    adjusted_logits += vertex_movements

    # Sample from adjusted probabilities
    probs = F.softmax(adjusted_logits, dim=-1)
    selected_token = torch.multinomial(probs, 1)
    
    logger.debug(f"Selected token ID: {selected_token.item()}")

    return selected_token, adjusted_logits

# --------------------------- Interactive 3D Visualization with Matplotlib --------------------------- #

def plot_interactive_3d_space(hidden_states, entropies, time_steps):
    """Optimized 3D projection with token-level interactivity using Matplotlib."""
    pca = PCA(n_components=3)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    plt.ion()  # Interactive mode on

    for hs, entropy, time_step in zip(hidden_states, entropies, time_steps):
        if hs is None or hs.numel() == 0:
            continue
        projected_hs = pca.fit_transform(hs.detach().cpu().numpy())
        entropy_norm = (entropy.detach().cpu().numpy() - np.min(entropy.detach().cpu().numpy())) / \
                       (np.max(entropy.detach().cpu().numpy()) - np.min(entropy.detach().cpu().numpy()) + 1e-10)
        ax.scatter(projected_hs[:, 0], projected_hs[:, 1], projected_hs[:, 2],
                   c=entropy_norm, cmap='viridis', marker='o', s=5, alpha=0.6, label=f"Step {time_step}")
        plt.draw()
        plt.pause(0.001)  # Small pause to update the plot

    ax.set_title("Token-wise 3D Space Travel with Layer Progression")
    ax.set_xlabel('PCA 1')
    ax.set_ylabel('PCA 2')
    ax.set_zlabel('PCA 3')
    plt.ioff()
    plt.show()
    logger.info("Displayed interactive 3D visualization.")

# --------------------------- Context Management --------------------------- #

class AdvancedContextManager:
    def __init__(self, model, tokenizer, max_history=10, summary_threshold=5):
        self.model = model
        self.tokenizer = tokenizer
        self.conversation_history = []
        self.max_history = max_history
        self.summary_threshold = summary_threshold
        self.tfidf_vectorizer = TfidfVectorizer()
        self.persona_snippets = {
            "formal": "You are a formal and professional AI assistant.",
            "casual": "You are a friendly and casual AI assistant.",
            "academic": "You are an academic AI assistant with a focus on scientific accuracy.",
            "creative": "You are a creative and imaginative AI assistant."
        }

    def update_context(self, user_input, model_output):
        self.conversation_history.append((user_input, model_output))
        logger.debug(f"Updated conversation history with user input: {user_input}")
        if len(self.conversation_history) > self.max_history:
            self.summarize_older_context()

    def summarize_older_context(self):
        older_context = self.conversation_history[:-self.summary_threshold]
        summary_prompt = "Summarize the following conversation concisely, capturing key points and context:\n"
        for user, ai in older_context:
            summary_prompt += f"User: {user}\nAI: {ai}\n"
        
        summary_input = self.tokenizer(summary_prompt, return_tensors="pt", truncation=True, max_length=1024).to(self.model.device)
        with torch.no_grad():
            summary_output = self.model.generate(summary_input.input_ids, max_length=200, num_return_sequences=1, temperature=0.7)
        summary = self.tokenizer.decode(summary_output[0], skip_special_tokens=True)
        
        self.conversation_history = [("SUMMARY", summary)] + self.conversation_history[-self.summary_threshold:]
        logger.info("Summarized older context.")

    def get_relevant_context(self, current_input, top_k=3):
        if not self.conversation_history:
            return ""

        context_texts = [f"{user} {ai}" for user, ai in self.conversation_history]
        tfidf_matrix = self.tfidf_vectorizer.fit_transform(context_texts + [current_input])
        cosine_similarities = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1]).flatten()
        most_relevant_indices = cosine_similarities.argsort()[-top_k:][::-1]
        
        relevant_context = ""
        for idx in most_relevant_indices:
            user, ai = self.conversation_history[idx]
            relevant_context += f"User: {user}\nAI: {ai}\n\n"
        
        logger.debug("Retrieved relevant context based on TF-IDF similarities.")
        return relevant_context.strip()

    def select_persona_context(self, user_input):
        if any(word in user_input.lower() for word in ["academic", "scientific", "research"]):
            return self.persona_snippets["academic"]
        elif any(word in user_input.lower() for word in ["creative", "imagine", "story"]):
            return self.persona_snippets["creative"]
        elif any(word in user_input.lower() for word in ["formal", "professional", "business"]):
            return self.persona_snippets["formal"]
        else:
            return self.persona_snippets["casual"]

    def get_dynamic_prompt(self, user_input):
        relevant_context = self.get_relevant_context(user_input)
        persona_context = self.select_persona_context(user_input)
        dynamic_prompt = f"{persona_context}\n\nRelevant conversation history:\n{relevant_context}\n\nCurrent user input: {user_input}\n\nAI:"
        logger.debug("Constructed dynamic prompt for the model.")
        return dynamic_prompt

# --------------------------- Response Quality Management --------------------------- #

class ImprovedResponseQualityManager:
    LOW_ENTROPY_THRESHOLD = 1.5
    HIGH_ENTROPY_THRESHOLD = 25.0
    WINDOW_SIZE = 50
    EOT_TOKENS = ['�', '\ufffd']

    def __init__(self, tokenizer, model):
        self.tokenizer = tokenizer
        self.model = model
        self.embedding_cache = {}
        logger.info("Initialized ImprovedResponseQualityManager.")

    def remove_eot_tokens(self, response):
        for token in self.EOT_TOKENS:
            response = response.rstrip(token)
        return response.strip()

    def _calculate_relevance(self, user_input, response):
        tokens_input = set(self.tokenizer.tokenize(user_input.lower()))
        tokens_response = set(self.tokenizer.tokenize(response.lower()))
        overlap = len(tokens_input & tokens_response)
        relevance_score = overlap / max(len(tokens_input), 1)
        logger.debug(f"Calculated relevance score: {relevance_score:.4f}")
        return relevance_score

    def _check_fluency(self, response):
        if len(response.split()) < 3:
            logger.debug("Response failed fluency check: less than 3 words.")
            return False
        if re.search(r'[^\x00-\x7F]+', response):
            logger.debug("Response failed fluency check: contains non-ASCII characters.")
            return False
        return True

    def _check_structure(self, response):
        if not response:
            logger.debug("Response failed structure check: empty response.")
            return False
        if not response[0].isupper():
            logger.debug("Response failed structure check: does not start with an uppercase letter.")
            return False
        if response[-1] not in '.!?':
            logger.debug("Response failed structure check: does not end with punctuation.")
            return False
        return True

    def _calculate_windowed_entropy(self, response):
        tokens = self.tokenizer.encode(response, return_tensors='pt').to(device)
        with torch.no_grad():
            outputs = self.model(tokens, labels=tokens)
            logits = outputs.logits

        probabilities = torch.softmax(logits, dim=-1)
        token_probs = probabilities.gather(2, tokens.unsqueeze(-1)).squeeze(-1)
        token_entropy = -torch.log2(token_probs + 1e-10)
        token_entropy = token_entropy.squeeze(0).cpu().numpy()

        window_size = self.WINDOW_SIZE
        num_windows = max(1, len(token_entropy) // window_size)
        entropy_values = []

        for i in range(num_windows):
            start = i * window_size
            end = start + window_size
            window = token_entropy[start:end]
            if len(window) == 0:
                continue
            window_entropy = np.mean(window)
            entropy_values.append(window_entropy)

        if not entropy_values:
            mean_entropy = 0.0
            std_entropy = 0.0
        else:
            mean_entropy = np.mean(entropy_values)
            std_entropy = np.std(entropy_values)

        logger.debug(f"Calculated windowed entropy: Mean={mean_entropy:.4f}, Std={std_entropy:.4f}")
        return mean_entropy, std_entropy

# --------------------------- Entropy-Based Temperature and Sampling Adjustment --------------------------- #

def adjust_temperature_based_on_entropy(entropy, low_threshold=1.5, high_threshold=25.0):
    if entropy > high_threshold:
        new_temp = max(0.7, 1.0 - ((entropy - high_threshold) / 10))
        logger.debug(f"High entropy detected ({entropy:.2f}). Lowering temperature to {new_temp:.2f}.")
        return new_temp
    elif entropy < low_threshold:
        new_temp = min(1.5, 1.0 + ((low_threshold - entropy) / 10))
        logger.debug(f"Low entropy detected ({entropy:.2f}). Increasing temperature to {new_temp:.2f}.")
        return new_temp
    logger.debug(f"Entropy within threshold ({entropy:.2f}). Keeping temperature at 1.0.")
    return 1.0  # Default temperature

def adjust_sampling_parameters(entropy, low_k=50, high_k=5, low_p=0.95, high_p=0.8):
    if entropy > 20.0:
        logger.debug(f"High entropy ({entropy:.2f}). Setting top_k to {high_k} and top_p to {high_p}.")
        return high_k, high_p  # Focused, deterministic sampling
    elif entropy < 10.0:
        logger.debug(f"Low entropy ({entropy:.2f}). Setting top_k to {low_k} and top_p to {low_p}.")
        return low_k, low_p  # More diverse sampling
    # Intermediate adjustment
    adjusted_k = int((high_k + low_k) / 2)
    adjusted_p = (high_p + low_p) / 2
    logger.debug(f"Intermediate entropy ({entropy:.2f}). Setting top_k to {adjusted_k} and top_p to {adjusted_p}.")
    return adjusted_k, adjusted_p

def sample_token(probs, top_k, top_p, temperature, special_tokens_set):
    if temperature != 1.0:
        probs = probs / temperature

    if top_k > 0:
        topk_probs, topk_indices = torch.topk(probs, top_k)
        probs = torch.zeros_like(probs).scatter_(1, topk_indices, topk_probs)
        logger.debug(f"Applied top_k filtering with top_k={top_k}.")

    if top_p > 0.0:
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        sorted_probs[cumulative_probs > top_p] = 0
        probs = torch.zeros_like(probs).scatter_(1, sorted_indices, sorted_probs)
        logger.debug(f"Applied top_p filtering with top_p={top_p}.")

    probs = probs / (probs.sum(dim=-1, keepdim=True) + 1e-10)
    
    for token_id in special_tokens_set:
        if probs[0, token_id] > 0.1:  # Threshold can be adjusted
            logger.info(f"Prioritizing special token: {SPECIAL_TOKEN_MAP.get(token_id, 'UNKNOWN')}")
            return torch.tensor([[token_id]]).to(probs.device)

    token_id = torch.multinomial(probs, num_samples=1)
    logger.debug(f"Sampled token ID: {token_id.item()}")
    return token_id

# --------------------------- Response Generation --------------------------- #

def generate_macroprocessed_response(prompt, model, tokenizer, quality_manager):
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_CONTEXT_LENGTH
    ).to(device)
    input_ids = inputs["input_ids"]

    max_tokens = 2048  # Adjust as needed
    generated_ids = input_ids.clone()

    token_log = []
    entropies = []
    varentropies = []
    kl_divs = []
    perplexities = []
    hidden_states = []

    with torch.no_grad():
        for _ in tqdm(range(max_tokens), desc="Generating Response", unit="token"):
            outputs = model(generated_ids)
            logits = outputs.logits[:, -1, :]
            probs = torch.softmax(logits, dim=-1)

            entropy = calculate_entropy(logits)
            varentropy = calculate_varentropy(entropy)
            kl_div = calculate_kl_divergence(logits)
            perplexity = calculate_perplexity(logits)

            temperature = adjust_temperature_based_on_entropy(entropy.item())
            top_k, top_p = adjust_sampling_parameters(entropy.item())

            special_tokens_set = {
                tokenizer.eos_token_id, 
                tokenizer.convert_tokens_to_ids("<|eom_id|>"),
                tokenizer.convert_tokens_to_ids("<|eot_id|>")
            }

            token_id = sample_token(probs, top_k, top_p, temperature, special_tokens_set)

            if token_id.dim() != 2 or token_id.size(1) != 1:
                logger.error(f"Unexpected token_id shape: {token_id.shape}")
                raise ValueError(f"token_id has incorrect shape: {token_id.shape}")

            generated_ids = torch.cat([generated_ids, token_id], dim=1)

            token_log.append({
                "token_id": token_id.item(),
                "entropy": entropy.item(),
                "temperature": temperature,
                "top_k": top_k,
                "top_p": top_p
            })

            if token_id.item() in tokenizer.all_special_ids:
                logger.info(f"End-of-sequence token detected: {SPECIAL_TOKEN_MAP.get(token_id.item(), 'UNKNOWN')}")
                break

            # Collecting metrics for visualization
            entropies.append(entropy)
            varentropies.append(varentropy)
            kl_divs.append(kl_div)
            perplexities.append(perplexity)
            hidden_states.append(None)  # Placeholder to avoid accumulating tensors

    for log_entry in token_log:
        logger.info(f"Token: {log_entry['token_id']}, Entropy: {log_entry['entropy']:.2f}, "
                    f"Temperature: {log_entry['temperature']:.2f}, top_k: {log_entry['top_k']}, top_p: {log_entry['top_p']}")

    response = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    response = response.split("AI:")[-1].strip()
    response = quality_manager.remove_eot_tokens(response)

    return response, entropies, varentropies, kl_divs, perplexities, hidden_states, range(len(entropies))

def remove_memory_recall(response):
    response = re.sub(r"\[Memory\]:.*\nAI:", "", response, flags=re.DOTALL)
    return response.strip()

def improved_generate_response(input_text, model, tokenizer, history, quality_manager, context_manager):
    sanitized_input = sanitize_input(input_text)
    prompt = context_manager.get_dynamic_prompt(sanitized_input)

    response, entropies, varentropies, kl_divs, perplexities, hidden_states, time_steps = generate_macroprocessed_response(
        prompt, model, tokenizer, quality_manager
    )

    context_manager.update_context(sanitized_input, response)

    # Optionally visualize every 'visualization_frequency' tokens
    config = history.get_config()
    visualization_frequency = config.get("visualization_frequency", 5)
    if len(entropies) % visualization_frequency == 0 and config.get("interactive_visuals", True):
        plot_interactive_3d_space(hidden_states, entropies, time_steps)

    return response, context_manager.conversation_history

def sanitize_input(user_input):
    sanitized = re.sub(r'[^\w\s.,!?]', '', user_input)
    return sanitized[:500]

# --------------------------- Interactive Loop --------------------------- #

def interactive_query(model, tokenizer, quality_manager, context_manager, user_config):
    print("\n--- LLaMA Instruct Model Interactive Query ---")
    print("Type 'exit' to quit.\n")

    while True:
        try:
            user_input = input("Enter your query: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting...")
            break

        if user_input.lower() == 'exit':
            print("Exiting...")
            break

        if not user_input:
            print("Please enter a valid query.")
            continue

        response, _ = improved_generate_response(
            user_input,
            model,
            tokenizer,
            user_config,
            quality_manager,
            context_manager
        )

        print(f"Model Response: {response}\n")

        # Free up unused memory
        torch.cuda.empty_cache()

# --------------------------- Flash Attention Check --------------------------- #

def check_flash_attention():
    try:
        import flash_attn
        logger.info("Flash Attention is available and enabled.")
    except ImportError:
        logger.warning("Flash Attention is not available. Using standard scaled dot product attention.")

# --------------------------- Model Loading --------------------------- #

def load_configuration(config_path):
    with open(config_path, "r") as f:
        config_data = json.load(f)
    config = LlamaConfig(**config_data)
    logger.info(f"Model configuration loaded from {config_path}")
    return config

def load_tokenizer_with_special_tokens(source_dir):
    tokenizer = AutoTokenizer.from_pretrained(source_dir)
    special_tokens_dict = {
        'additional_special_tokens': list(SPECIAL_TOKEN_MAP.values())
    }
    tokenizer.add_special_tokens(special_tokens_dict)
    if "<|finetune_right_pad_id|>" in tokenizer.get_vocab():
        tokenizer.pad_token = "<|finetune_right_pad_id|>"
        logger.info(f"Assigned '<|finetune_right_pad_id|>' as pad_token.")
    else:
        logger.warning(f"'<|finetune_right_pad_id|>' not found in tokenizer vocabulary.")
    return tokenizer

def load_offloaded_weights(model, weights_dir):
    logger.info("Loading model weights from disk.")
    for name, param in model.named_parameters():
        file_name = f"{name.replace('.', '_')}.dat"
        file_path = os.path.join(weights_dir, file_name)

        if os.path.exists(file_path):
            dtype_map = {
                torch.float16: np.float16,
                torch.float32: np.float32,
                torch.int64: np.int64,
                torch.int32: np.int32,
                torch.bfloat16: np.float32,  # Loading bfloat16 as float32 first
            }
            expected_dtype = dtype_map.get(param.dtype, np.float32)
            logger.info(f"Loading {file_name} into {name} with expected type {expected_dtype}")

            try:
                tensor_data = np.fromfile(file_path, dtype=expected_dtype)
                loaded_tensor = torch.from_numpy(tensor_data).to(device)

                if param.dtype == torch.bfloat16:
                    loaded_tensor = loaded_tensor.to(torch.bfloat16)

                with torch.no_grad():
                    param.data.copy_(loaded_tensor.view_as(param))
                logger.debug(f"Successfully loaded {file_name} into {name}")
            except Exception as e:
                logger.error(f"Error loading {file_name} into {name}: {e}")
        else:
            logger.warning(f"Weight file {file_path} not found.")

    logger.info("All available weights loaded successfully.")

# --------------------------- Main Execution --------------------------- #

def main():
    # Load user configuration
    user_config = UserConfig()

    # Load model configuration
    config = load_configuration(MODEL_JSON_PATH)

    # Initialize the model
    model = LlamaForCausalLM(config).to(device)
    logger.info("Initialized LLaMA model on GPU.")

    # Load offloaded weights
    load_offloaded_weights(model, WEIGHTS_DIR)
    model.eval()
    logger.info("Model is set to evaluation mode.")

    # Load tokenizer with special tokens
    tokenizer = load_tokenizer_with_special_tokens(SOURCE_DIR)

    # Resize token embeddings if special tokens were added
    if tokenizer.pad_token and tokenizer.pad_token not in tokenizer.get_vocab():
        model.resize_token_embeddings(len(tokenizer))
        logger.info("Resized model token embeddings to accommodate the new pad_token.")
    else:
        logger.info("pad_token already exists in the tokenizer's vocabulary. No need to resize embeddings.")

    # Check for Flash Attention
    check_flash_attention()

    # Initialize Response Quality Manager
    quality_manager = ImprovedResponseQualityManager(tokenizer, model)

    # Initialize Context Manager
    context_manager = AdvancedContextManager(model, tokenizer)

    logger.info("Model loaded successfully. You can now query the model.")

    # Start interactive query loop
    interactive_query(model, tokenizer, quality_manager, context_manager, user_config)

if __name__ == "__main__":
    main()
