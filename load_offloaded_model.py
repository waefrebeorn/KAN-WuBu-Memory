import os
import json
import re
import gc
import logging
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.animation as animation
from tqdm import tqdm
from sklearn.decomposition import PCA
from transformers import (
    LlamaForCausalLM,
    LlamaConfig,
    AutoTokenizer,
    set_seed
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --------------------------- Configuration --------------------------- #

# Define paths (Update these paths according to your environment)
SOURCE_DIR = "models/Llama_32_1B/"  # Directory containing the tokenizer and config.json
WEIGHTS_DIR = os.path.join(SOURCE_DIR, "offload")  # Directory containing .dat weight files
MODEL_JSON_PATH = os.path.join(SOURCE_DIR, "config.json")  # Path to model config

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s:%(levelname)s:%(name)s:%(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Special tokens map (Update as per your tokenizer's special tokens)
SPECIAL_TOKEN_MAP = {
    "<|eot_id|>": 16770,  # Example mapping, replace with actual IDs
    "<|eom_id|>": 11,
    "<|finetune_right_pad_id|>": 0,
    # Add other special tokens if necessary
}

# Maximum context length
MAX_CONTEXT_LENGTH = 2048

# User Configuration Class
class UserConfig:
    def __init__(self, config_path=None):
        # Default configurations
        self.max_length = 2048
        self.initial_weights = {
            'entropy': 1.0,
            'varentropy': 0.5,
            'kl_div': 0.3,
            'perplexity': 0.2
        }
        self.visualization_frequency = 5
        self.logging_level = "INFO"
        self.interactive_visuals = True
        self.precision = "float16"
        self.kv_cache_enabled = True

        # Load configurations from a JSON file if provided
        if config_path and os.path.exists(config_path):
            with open(config_path, "r") as f:
                config_data = json.load(f)
                self.max_length = config_data.get("max_length", self.max_length)
                self.initial_weights = config_data.get("initial_weights", self.initial_weights)
                self.visualization_frequency = config_data.get("visualization_frequency", self.visualization_frequency)
                self.logging_level = config_data.get("logging_level", self.logging_level)
                self.interactive_visuals = config_data.get("interactive_visuals", self.interactive_visuals)
                self.precision = config_data.get("precision", self.precision)
                self.kv_cache_enabled = config_data.get("kv_cache_enabled", self.kv_cache_enabled)

# --------------------------- Helper Functions --------------------------- #

def calculate_entropy(logits):
    """Calculate entropy from logits."""
    probs = F.softmax(logits, dim=-1)
    entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)
    return entropy

def calculate_varentropy(entropy):
    """Calculate variance of entropy."""
    if entropy.numel() <= 1:
        return torch.tensor(0.0).to(entropy.device)
    varentropy = torch.var(entropy, unbiased=False)
    return varentropy

def calculate_kl_divergence(logits):
    """Calculate KL divergence from logits to uniform distribution."""
    probs = F.softmax(logits, dim=-1)
    uniform = torch.full_like(probs, 1.0 / probs.size(-1))
    kl_div = F.kl_div(probs.log(), uniform, reduction='batchmean')
    return kl_div

def calculate_perplexity(logits):
    """Calculate perplexity from logits."""
    probs = F.softmax(logits, dim=-1)
    perplexity = torch.exp(-torch.sum(probs * torch.log(probs + 1e-10), dim=-1))
    return perplexity

# --------------------------- Adaptive Weighting System --------------------------- #

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

# --------------------------- 4D Visualization --------------------------- #

class VRAMEfficient4DVisualizer:
    """VRAM-efficient 4D Visualizer using Matplotlib for interactive 3D plotting with entropy as color."""
    def __init__(self):
        # Set Matplotlib backend for Windows
        matplotlib.use('TkAgg')  # Ensure compatibility with Windows 11
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.scatter = None
        self.anim = None
        self.is_plotting = False  # To prevent multiple plots

    def plot_4d_visualization(self, hidden_states, entropies, time_steps):
        """Visualizes hidden states along with entropies in 4D space."""
        if self.is_plotting:
            # Avoid plotting multiple times simultaneously
            return

        if len(hidden_states) == 0 or len(entropies) == 0:
            logger.warning("No data available for visualization.")
            return

        if hidden_states.shape[0] < 3:
            logger.warning("Not enough data points for 3D PCA visualization.")
            return

        self.is_plotting = True

        try:
            # Apply PCA for 3D projection
            pca = PCA(n_components=3)
            # Flatten hidden_states if they have more dimensions
            flattened_states = hidden_states.reshape(hidden_states.shape[0], -1)
            projected_states = pca.fit_transform(flattened_states)

            # Normalize colors for colormap based on entropy
            colors = np.array(entropies)
            colors_norm = (colors - np.min(colors)) / (np.max(colors) - np.min(colors) + 1e-10)

            # Clear previous scatter
            self.ax.cla()

            # Initialize scatter plot
            self.scatter = self.ax.scatter(
                projected_states[:, 0], projected_states[:, 1], projected_states[:, 2],
                c=colors_norm, cmap='viridis', marker='o', s=20, alpha=0.6
            )
            self.ax.set_xlabel("PCA 1")
            self.ax.set_ylabel("PCA 2")
            self.ax.set_zlabel("PCA 3")
            self.fig.colorbar(self.scatter, ax=self.ax, label="Entropy")
            self.ax.set_title("4D Visualization of Hidden States (Entropy as Color)")

            # Animation function to rotate the view
            def update(frame):
                self.ax.view_init(elev=10., azim=frame % 360)
                return self.scatter,

            # Create animation
            self.anim = animation.FuncAnimation(
                self.fig, update, frames=range(0, 360, 2), interval=50, blit=False, repeat=True
            )
            plt.show(block=False)  # Non-blocking show
            logger.info("Displayed interactive 4D visualization.")
        except Exception as e:
            logger.error(f"Visualization error: {e}")
        finally:
            self.is_plotting = False

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
            outputs = self.model(tokens, labels=tokens, output_hidden_states=True)
            logits = outputs.logits

        probabilities = torch.softmax(logits, dim=-1)
        token_probs = probabilities.gather(2, tokens.unsqueeze(-1)).squeeze(-1)
        token_entropy = -torch.log(token_probs + 1e-10)
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

# --------------------------- Sampling Adjustments --------------------------- #

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
    
    # Prioritize special tokens if their probability exceeds a threshold
    for token_id in special_tokens_set:
        if probs[0, token_id] > 0.1:  # Threshold can be adjusted
            logger.info(f"Prioritizing special token: {SPECIAL_TOKEN_MAP.get(token_id, 'UNKNOWN')}")
            return torch.tensor([[token_id]]).to(probs.device)

    token_id = torch.multinomial(probs, num_samples=1)
    logger.debug(f"Sampled token ID: {token_id.item()}")

    return token_id

# --------------------------- Token Selection with Adaptive Weighting --------------------------- #

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

    # Apply vertex movement strategy (Assuming vertex_movements is a tensor compatible with logits)
    adjusted_logits += vertex_movements

    # Sample from adjusted probabilities
    probs = F.softmax(adjusted_logits, dim=-1)
    selected_token = torch.multinomial(probs, 1)
    
    logger.debug(f"Selected token ID: {selected_token.item()}")

    return selected_token, adjusted_logits

# --------------------------- Response Generation --------------------------- #

def generate_macroprocessed_response(prompt, model, tokenizer, quality_manager, loss_weighting_system, visualizer, user_config):
    """Generates a response from the model based on the prompt."""
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
    time_steps = []

    with torch.no_grad():
        for step in tqdm(range(max_tokens), desc="Generating Response", unit="token"):
            outputs = model(generated_ids, output_hidden_states=True)
            logits = outputs.logits[:, -1, :]

            # Calculate metrics
            entropy = calculate_entropy(logits)
            varentropy = calculate_varentropy(entropy)
            kl_div = calculate_kl_divergence(logits)
            perplexity = calculate_perplexity(logits)

            # Adjust temperature and sampling parameters
            temperature = adjust_temperature_based_on_entropy(entropy.item())
            top_k, top_p = adjust_sampling_parameters(entropy.item())

            # Sample token
            special_tokens_set = {
                tokenizer.eos_token_id, 
                tokenizer.convert_tokens_to_ids("<|eom_id|>"),
                tokenizer.convert_tokens_to_ids("<|eot_id|>")
            }
            token_id = sample_token(F.softmax(logits, dim=-1), top_k, top_p, temperature, special_tokens_set)

            if token_id.dim() != 2 or token_id.size(1) != 1:
                logger.error(f"Unexpected token_id shape: {token_id.shape}")
                raise ValueError(f"token_id has incorrect shape: {token_id.shape}")

            generated_ids = torch.cat([generated_ids, token_id], dim=1)

            # Log token details
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
            entropies.append(entropy.item())
            varentropies.append(varentropy.item())
            kl_divs.append(kl_div.item())
            perplexities.append(perplexity.item())
            if outputs.hidden_states:
                # Take the last layer's hidden state for the last token
                last_hidden_state = outputs.hidden_states[-1][:, -1, :]  # Shape: [batch_size, hidden_dim]
                hidden_states.append(last_hidden_state.detach().cpu().numpy())
            else:
                hidden_states.append(np.array([]))  # Fallback to empty array if hidden_states is None
            time_steps.append(step)

            # Optional: Visualize at certain intervals
            if visualizer and step % user_config.visualization_frequency == 0:
                hidden_states_cpu = np.concatenate([hs for hs in hidden_states if hs.size > 0], axis=0)
                visualizer.plot_4d_visualization(hidden_states_cpu, entropies, time_steps)

    # Logging token information
    for log_entry in token_log:
        logger.info(f"Token: {log_entry['token_id']}, Entropy: {log_entry['entropy']:.2f}, "
                    f"Temperature: {log_entry['temperature']:.2f}, top_k: {log_entry['top_k']}, top_p: {log_entry['top_p']}")

    # Decode the generated tokens to get the response
    response = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    response = response.split("AI:")[-1].strip()
    response = quality_manager.remove_eot_tokens(response)

    # Concatenate hidden states and prepare for visualization
    hidden_states_cpu = np.concatenate([hs for hs in hidden_states if hs.size > 0], axis=0)

    # Run visualization
    if hidden_states_cpu.size > 0 and entropies:
        visualizer.plot_4d_visualization(hidden_states_cpu, entropies, time_steps)

    # Clear cache and collect garbage
    torch.cuda.empty_cache()
    gc.collect()

    return response, entropies, varentropies, kl_divs, perplexities, hidden_states_cpu

def remove_memory_recall(response):
    response = re.sub(r"\[Memory\]:.*\nAI:", "", response, flags=re.DOTALL)
    return response.strip()

def improved_generate_response(input_text, model, tokenizer, quality_manager, context_manager, loss_weighting_system, visualizer, user_config):
    """Generates a response and handles visualization."""
    sanitized_input = sanitize_input(input_text)
    prompt = context_manager.get_dynamic_prompt(sanitized_input)

    response, entropies, varentropies, kl_divs, perplexities, hidden_states_cpu = generate_macroprocessed_response(
        prompt, model, tokenizer, quality_manager, loss_weighting_system, visualizer, user_config
    )

    context_manager.update_context(sanitized_input, response)

    return response, context_manager.conversation_history

def sanitize_input(user_input):
    """Sanitizes the user input to prevent injection of unwanted tokens or patterns."""
    sanitized = re.sub(r'[^\w\s.,!?]', '', user_input)
    return sanitized[:500]

# --------------------------- Interactive Loop --------------------------- #

def interactive_query(model, tokenizer, quality_manager, context_manager, user_config):
    """Handles the interactive query loop with the user."""
    print("\n--- LLaMA Instruct Model Interactive Query ---")
    print("Type 'exit' to quit.\n")

    # Initialize Adaptive Weighting System
    loss_weighting_system = AdaptiveWeightingSystem(initial_weights=user_config.initial_weights)

    # Initialize Visualizer if enabled
    visualizer = VRAMEfficient4DVisualizer() if user_config.interactive_visuals else None

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

        try:
            response, _ = improved_generate_response(
                user_input,
                model,
                tokenizer,
                quality_manager,
                context_manager,
                loss_weighting_system,
                visualizer,
                user_config  # Pass user_config here
            )
            print(f"Model Response: {response}\n")
        except Exception as e:
            logger.error(f"An error occurred during response generation: {e}")
            print(f"An error occurred: {e}")

# --------------------------- Flash Attention Check --------------------------- #

def check_flash_attention():
    """Checks if Flash Attention is available and logs the status."""
    try:
        import flash_attn
        logger.info("Flash Attention is available and enabled.")
    except ImportError:
        logger.warning("Flash Attention is not available. Using standard scaled dot product attention.")

# --------------------------- Model Loading --------------------------- #

def load_configuration(config_path):
    """Loads the model configuration from a JSON file."""
    if not os.path.exists(config_path):
        logger.error(f"Configuration file not found at {config_path}")
        raise FileNotFoundError(f"Configuration file not found at {config_path}")
    
    with open(config_path, "r") as f:
        config_data = json.load(f)
    config = LlamaConfig(**config_data)
    logger.info(f"Model configuration loaded from {config_path}")
    return config

def load_tokenizer_with_special_tokens(source_dir):
    """Loads the tokenizer and adds special tokens."""
    tokenizer = AutoTokenizer.from_pretrained(source_dir)
    special_tokens_dict = {
        'additional_special_tokens': list(SPECIAL_TOKEN_MAP.keys())
    }
    tokenizer.add_special_tokens(special_tokens_dict)
    if "<|finetune_right_pad_id|>" in tokenizer.get_vocab():
        tokenizer.pad_token = "<|finetune_right_pad_id|>"
        logger.info(f"Assigned '<|finetune_right_pad_id|>' as pad_token.")
    else:
        logger.warning(f"'<|finetune_right_pad_id|>' not found in tokenizer vocabulary.")
    return tokenizer

def load_offloaded_weights(model, weights_dir):
    """Loads the model weights from the offload directory."""
    if not os.path.exists(weights_dir):
        logger.error(f"Weights directory not found at {weights_dir}")
        raise FileNotFoundError(f"Weights directory not found at {weights_dir}")

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

                # Reshape the loaded tensor to match the parameter's shape
                loaded_tensor = loaded_tensor.view_as(param)
                with torch.no_grad():
                    param.data.copy_(loaded_tensor)
                logger.debug(f"Successfully loaded {file_name} into {name}")
            except Exception as e:
                logger.error(f"Error loading {file_name} into {name}: {e}")
        else:
            logger.warning(f"Weight file {file_path} not found.")

    logger.info("All available weights loaded successfully.")

# --------------------------- Main Execution --------------------------- #

def main():
    """Main function to load the model, tokenizer, and start the interactive query loop."""
    # Load user configuration
    user_config = UserConfig()

    # Set logging level
    numeric_level = getattr(logging, user_config.logging_level.upper(), None)
    if isinstance(numeric_level, int):
        logger.setLevel(numeric_level)
    else:
        logger.setLevel(logging.INFO)
        logger.warning(f'Invalid log level: {user_config.logging_level}. Using INFO level.')

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
