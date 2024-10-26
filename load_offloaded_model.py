import os
import json
import re
import gc
import logging
import time
import difflib
from collections import defaultdict

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog

from transformers import (
    LlamaForCausalLM,
    LlamaConfig,
    AutoTokenizer,
    set_seed
)

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
        self.entropy_thresholds = {
            'low': 1.5,
            'high': 25.0
        }
        self.top_k = {
            'low': 50,
            'high': 5
        }
        self.top_p = {
            'low': 0.95,
            'high': 0.8
        }

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
                self.entropy_thresholds = config_data.get("entropy_thresholds", self.entropy_thresholds)
                self.top_k = config_data.get("top_k", self.top_k)
                self.top_p = config_data.get("top_p", self.top_p)

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

# --------------------------- Advanced Entropy Analyzer --------------------------- #

class AdvancedEntropyAnalyzer:
    def __init__(self, window_size=50, history_length=1000):
        self.window_size = window_size
        self.history_length = history_length
        self.entropy_history = []
        self.varentropy_history = []
        self.pattern_history = []
        self.weighted_patterns = {}

    def calculate_contextual_entropy(self, logits, context_window):
        """Calculate entropy with contextual weighting"""
        probs = F.softmax(logits, dim=-1)

        # Base entropy calculation
        base_entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)

        # Context-weighted entropy
        context_weights = self._calculate_context_weights(context_window)
        weighted_entropy = base_entropy * context_weights

        # Add to history
        self.entropy_history.append(weighted_entropy.item())
        if len(self.entropy_history) > self.history_length:
            self.entropy_history.pop(0)

        return weighted_entropy

    def calculate_advanced_varentropy(self, entropy_window):
        """Enhanced varentropy calculation with pattern recognition"""
        if len(entropy_window) < 2:
            return torch.tensor(0.0).to(entropy_window.device)

        # Calculate basic varentropy
        entropy_tensor = torch.tensor(entropy_window).to(entropy_window.device)
        mean_entropy = torch.mean(entropy_tensor)
        varentropy = torch.var(entropy_tensor, unbiased=False)

        # Pattern-based adjustment
        pattern_factor = self._detect_entropy_patterns(entropy_window)
        adjusted_varentropy = varentropy * pattern_factor

        # Store in history
        self.varentropy_history.append(adjusted_varentropy.item())
        if len(self.varentropy_history) > self.history_length:
            self.varentropy_history.pop(0)

        return adjusted_varentropy

    def _calculate_context_weights(self, context_window):
        """Calculate context-based weights for entropy"""
        # Recent context has higher weight
        time_weights = torch.linspace(0.5, 1.0, len(context_window)).to(context_window.device)

        # Calculate semantic relevance
        semantic_weights = self._calculate_semantic_weights(context_window)

        # Combine weights
        combined_weights = time_weights * semantic_weights
        normalized_weights = combined_weights / combined_weights.sum()

        return normalized_weights

    def _detect_entropy_patterns(self, entropy_window):
        """Detect patterns in entropy variations"""
        # Convert to numpy for pattern analysis
        entropy_array = np.array(entropy_window)

        # Calculate various pattern indicators
        trend = np.polyfit(np.arange(len(entropy_array)), entropy_array, 1)[0]
        volatility = np.std(np.diff(entropy_array))
        cyclicity = self._detect_cycles(entropy_array)

        # Combine pattern indicators
        pattern_factor = 1.0 + (abs(trend) * 0.2 + volatility * 0.3 + cyclicity * 0.5)

        # Store pattern
        self.pattern_history.append({
            'trend': trend,
            'volatility': volatility,
            'cyclicity': cyclicity,
            'factor': pattern_factor
        })

        return torch.tensor(pattern_factor).to(device)

    def _detect_cycles(self, data):
        """Detect cyclic patterns in entropy data"""
        if len(data) < 4:
            return 0.0

        # Use autocorrelation to detect cycles
        autocorr = np.correlate(data, data, mode='full')
        autocorr = autocorr[len(autocorr)//2:]

        # Find peaks in autocorrelation
        peaks = []
        for i in range(1, len(autocorr)-1):
            if autocorr[i-1] < autocorr[i] > autocorr[i+1]:
                peaks.append((i, autocorr[i]))

        if not peaks:
            return 0.0

        # Calculate cycle strength
        cycle_strength = max(peak[1] for peak in peaks) / autocorr[0]
        return cycle_strength

    def _calculate_semantic_weights(self, context_window):
        """Calculate semantic relevance weights"""
        if not context_window:
            return torch.ones(1).to(device)

        # Calculate token similarities
        similarities = []
        for i in range(len(context_window)):
            sim = self._token_similarity(context_window[i], context_window[-1])
            similarities.append(sim)

        # Convert to tensor and normalize
        sim_tensor = torch.tensor(similarities).to(device)
        weights = F.softmax(sim_tensor, dim=0)

        return weights

    def _token_similarity(self, token1, token2):
        """Calculate similarity between tokens"""
        # Simple character-based similarity
        chars1 = set(str(token1))
        chars2 = set(str(token2))
        similarity = len(chars1.intersection(chars2)) / max(len(chars1.union(chars2)), 1)
        return similarity

    def get_pattern_analysis(self):
        """Get analysis of recent entropy patterns"""
        if not self.pattern_history:
            return None

        recent_patterns = self.pattern_history[-self.window_size:]

        analysis = {
            'avg_trend': np.mean([p['trend'] for p in recent_patterns]),
            'avg_volatility': np.mean([p['volatility'] for p in recent_patterns]),
            'avg_cyclicity': np.mean([p['cyclicity'] for p in recent_patterns]),
            'pattern_strength': np.mean([p['factor'] for p in recent_patterns])
        }

        return analysis

    def get_entropy_statistics(self):
        """Get statistical analysis of entropy history"""
        if not self.entropy_history:
            return None

        recent_entropy = self.entropy_history[-self.window_size:]
        stats = {
            'mean': np.mean(recent_entropy),
            'std': np.std(recent_entropy),
            'min': np.min(recent_entropy),
            'max': np.max(recent_entropy),
            'trend': np.polyfit(np.arange(len(recent_entropy)), recent_entropy, 1)[0]
        }

        return stats

# --------------------------- Improved Response Quality Manager --------------------------- #

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
        response = re.sub(r"\[Memory\]:.*\nAI:", "", response, flags=re.DOTALL)
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

# --------------------------- Token Selection with Adaptive Weighting --------------------------- #

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
    # Ensure all tensors are on the same device
    device = probs.device

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
            return torch.tensor([[token_id]]).to(device)

    token_id = torch.multinomial(probs, num_samples=1)
    logger.debug(f"Sampled token ID: {token_id.item()}")

    return token_id

def select_token_with_weights(logits, vertex_movements, loss_weighting_system, precision="float16"):
    """Efficient token selection using entropy, varentropy, and configurable loss weights with precision support."""
    # Ensure logits are on the correct device
    logits = logits.to(device)
    
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

    # Ensure vertex_movements is on the same device
    vertex_movements = vertex_movements.to(device)
    # Apply vertex movement strategy (Assuming vertex_movements is a tensor compatible with logits)
    adjusted_logits += vertex_movements

    # Sample from adjusted probabilities
    probs = F.softmax(adjusted_logits, dim=-1)
    selected_token = torch.multinomial(probs, 1)

    logger.debug(f"Selected token ID: {selected_token.item()}")

    return selected_token, adjusted_logits

# --------------------------- Context Management --------------------------- #

class ConversationStateMachine:
    def __init__(self):
        self.states = {
            'initial': {'transitions': ['query', 'greeting']},
            'query': {'transitions': ['response', 'clarification']},
            'response': {'transitions': ['query', 'followup']},
            'clarification': {'transitions': ['query', 'response']},
            'followup': {'transitions': ['response', 'conclusion']},
            'conclusion': {'transitions': ['query', 'end']}
        }
        self.current_state = 'initial'
        self.state_history = []

    def transition(self, new_state):
        if new_state in self.states[self.current_state]['transitions']:
            self.state_history.append(self.current_state)
            self.current_state = new_state
            return True
        return False

class PatternDetector:
    def __init__(self):
        self.patterns = defaultdict(int)
        self.sequence_buffer = []

    def add_sequence(self, sequence):
        self.sequence_buffer.append(sequence)
        if len(self.sequence_buffer) > 5:
            self.sequence_buffer.pop(0)
        self._update_patterns()

    def _update_patterns(self):
        for i in range(len(self.sequence_buffer)):
            for j in range(i + 1, len(self.sequence_buffer)):
                pattern = self._find_common_pattern(
                    self.sequence_buffer[i],
                    self.sequence_buffer[j]
                )
                if pattern:
                    self.patterns[pattern] += 1

    def _find_common_pattern(self, seq1, seq2):
        # Use difflib to find common subsequences
        matcher = difflib.SequenceMatcher(None, seq1, seq2)
        matches = matcher.get_matching_blocks()

        if matches:
            return seq1[matches[0].a:matches[0].a + matches[0].size]
        return None

class AdvancedContextManager:
    def __init__(self, model, tokenizer, entropy_analyzer):
        self.model = model
        self.tokenizer = tokenizer
        self.entropy_analyzer = entropy_analyzer
        self.conversation_history = []
        self.macro_context = []
        self.topic_graph = nx.DiGraph()
        self.state_machine = ConversationStateMachine()
        self.pattern_detector = PatternDetector()

    def update_context(self, user_input, model_output, entropy_data):
        """Update conversation context with entropy-aware processing"""
        # Add to basic history
        self.conversation_history.append({
            'user_input': user_input,
            'model_output': model_output,
            'entropy_data': entropy_data,
            'timestamp': time.time()
        })

        # Update topic graph
        self._update_topic_graph(user_input, model_output)

        # Process macro context
        self._update_macro_context(user_input, model_output, entropy_data)

        # Update pattern detection
        self.pattern_detector.add_sequence(user_input + " " + model_output)

        # Cleanup old context
        self._cleanup_context()

    def _update_topic_graph(self, user_input, model_output):
        """Update topic graph with new conversation elements"""
        # Extract topics
        user_topics = self._extract_topics(user_input)
        model_topics = self._extract_topics(model_output)

        # Add nodes and edges
        for topic in user_topics:
            if topic not in self.topic_graph:
                self.topic_graph.add_node(topic, type='user')

        for topic in model_topics:
            if topic not in self.topic_graph:
                self.topic_graph.add_node(topic, type='model')

        # Connect related topics
        for t1 in user_topics:
            for t2 in model_topics:
                weight = self._calculate_topic_relation(t1, t2)
                self.topic_graph.add_edge(t1, t2, weight=weight)

    def _update_macro_context(self, user_input, model_output, entropy_data):
        """Update macro-level context understanding"""
        # Create macro context entry
        macro_entry = {
            'topics': self._extract_topics(user_input + " " + model_output),
            'entropy_pattern': self.entropy_analyzer.get_pattern_analysis(),
            'conversation_state': self.state_machine.current_state,
            'timestamp': time.time()
        }

        # Update conversation state machine based on user input or model output
        self._update_conversation_state(user_input, model_output)

        # Add semantic analysis
        macro_entry['semantic_vectors'] = self._calculate_semantic_vectors(
            user_input, model_output
        )

        self.macro_context.append(macro_entry)

    def _update_conversation_state(self, user_input, model_output):
        """Update the conversation state machine based on input/output"""
        # Simple heuristic: if user input contains a question mark, transition to 'query'
        if '?' in user_input:
            self.state_machine.transition('query')
        elif 'thank' in user_input.lower():
            self.state_machine.transition('conclusion')
        else:
            self.state_machine.transition('response')

    def _calculate_semantic_vectors(self, user_input, model_output):
        """Calculate semantic vector representations"""
        # Get model embeddings
        with torch.no_grad():
            user_tokens = self.tokenizer(user_input, return_tensors='pt').to(device)
            model_tokens = self.tokenizer(model_output, return_tensors='pt').to(device)

            user_embedding = self.model.get_input_embeddings()(user_tokens.input_ids).mean(dim=1)
            model_embedding = self.model.get_input_embeddings()(model_tokens.input_ids).mean(dim=1)

        return {
            'user_vector': user_embedding.cpu().numpy(),
            'model_vector': model_embedding.cpu().numpy()
        }

    def get_relevant_context(self, current_input, top_k=3):
        """Get most relevant context for current input"""
        if not self.conversation_history:
            return ""

        # Prepare context texts
        context_texts = [f"User: {entry['user_input']}\nAI: {entry['model_output']}" for entry in self.conversation_history]
        current_input_encoded = self.tokenizer(current_input, return_tensors='pt', truncation=True, max_length=512).input_ids
        current_input_decoded = self.tokenizer.decode(current_input_encoded[0], skip_special_tokens=True)

        # Vectorize
        tfidf_vectorizer = TfidfVectorizer().fit(context_texts + [current_input_decoded])
        tfidf_matrix = tfidf_vectorizer.transform(context_texts + [current_input_decoded])

        # Compute similarities
        cosine_similarities = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1]).flatten()
        most_relevant_indices = cosine_similarities.argsort()[-top_k:][::-1]

        # Retrieve relevant context
        relevant_context = "\n".join([context_texts[idx] for idx in most_relevant_indices])

        logger.debug("Retrieved relevant context based on TF-IDF similarities.")
        return relevant_context.strip()

    def select_persona_context(self, user_input):
        """Select persona context based on user input keywords"""
        if any(word in user_input.lower() for word in ["academic", "scientific", "research"]):
            return "You are an academic AI assistant with a focus on scientific accuracy."
        elif any(word in user_input.lower() for word in ["creative", "imagine", "story"]):
            return "You are a creative and imaginative AI assistant."
        elif any(word in user_input.lower() for word in ["formal", "professional", "business"]):
            return "You are a formal and professional AI assistant."
        else:
            return "You are a friendly and casual AI assistant."

    def get_dynamic_prompt(self, user_input):
        """Construct dynamic prompt based on context and persona"""
        relevant_context = self.get_relevant_context(user_input)
        persona_context = self.select_persona_context(user_input)
        dynamic_prompt = f"{persona_context}\n\nRelevant conversation history:\n{relevant_context}\n\nCurrent user input: {user_input}\n\nAI:"
        logger.debug("Constructed dynamic prompt for the model.")
        return dynamic_prompt

    def _extract_topics(self, text):
        """Extract simple topics based on keywords (can be enhanced with NLP techniques)"""
        keywords = re.findall(r'\b\w+\b', text.lower())
        # Simple keyword extraction; replace with more sophisticated methods if needed
        topics = set(keywords)
        return list(topics)

    def _calculate_topic_relation(self, t1, t2):
        """Calculate a simple relation weight between two topics"""
        return 1.0  # Placeholder for more complex relation calculations

    def _calculate_relevance_score(self, current_input, past_user_input, past_model_output, past_entropy):
        """Calculate contextual relevance score"""
        # Semantic similarity
        semantic_score = self._calculate_semantic_similarity(
            current_input,
            past_user_input + " " + past_model_output
        )

        # Topic overlap
        topic_score = self._calculate_topic_overlap(
            current_input,
            past_user_input + " " + past_model_output
        )

        # Entropy pattern similarity
        entropy_score = self._calculate_entropy_pattern_similarity(past_entropy)

        # Combine scores
        total_score = (
            semantic_score * 0.4 +
            topic_score * 0.3 +
            entropy_score * 0.3
        )

        return total_score

    def _calculate_semantic_similarity(self, input_text, context_text):
        """Calculate semantic similarity between input and context"""
        vectorizer = TfidfVectorizer().fit([input_text, context_text])
        tfidf_matrix = vectorizer.transform([input_text, context_text])
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2]).flatten()[0]
        return similarity

    def _calculate_topic_overlap(self, input_text, context_text):
        """Calculate topic overlap between input and context"""
        input_topics = set(self._extract_topics(input_text))
        context_topics = set(self._extract_topics(context_text))
        overlap = len(input_topics & context_topics)
        total = max(len(input_topics | context_topics), 1)
        return overlap / total

    def _calculate_entropy_pattern_similarity(self, entropy_data):
        """Calculate similarity based on entropy patterns"""
        # Placeholder for actual pattern similarity calculations
        return 1.0

    def _cleanup_context(self):
        """Clean up old context entries"""
        # Remove old entries
        current_time = time.time()
        self.conversation_history = [
            entry for entry in self.conversation_history
            if current_time - entry['timestamp'] < 3600  # Keep last hour
        ]

        # Cleanup macro context
        self.macro_context = self.macro_context[-50:]  # Keep last 50 entries

        # Cleanup topic graph
        self._prune_topic_graph()

    def _prune_topic_graph(self):
        """Remove old or irrelevant topics from graph"""
        to_remove = []
        for node in self.topic_graph.nodes():
            if self.topic_graph.degree(node) < 2:  # Remove isolated topics
                to_remove.append(node)

        for node in to_remove:
            self.topic_graph.remove_node(node)

# --------------------------- Vector Space Visualizer --------------------------- #

class VectorSpaceVisualizer:
    def __init__(self, parent):
        self.parent = parent
        self.setup_plot()

    def setup_plot(self):
        """Setup the 3D vector space visualization"""
        self.figure = plt.Figure(figsize=(6, 4), dpi=100)
        self.plot = self.figure.add_subplot(111, projection='3d')

        self.canvas = FigureCanvasTkAgg(self.figure, master=self.parent)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def update(self, vector_data):
        """Update vector space visualization"""
        self.plot.clear()

        # Plot vectors
        vectors = vector_data['vectors']
        colors = vector_data['colors']

        # Use PCA to reduce dimensionality to 3D if necessary
        if vectors.shape[1] > 3:
            pca = PCA(n_components=3)
            vectors = pca.fit_transform(vectors)

        scatter = self.plot.scatter(vectors[:, 0], vectors[:, 1], vectors[:, 2],
                                  c=colors, cmap='viridis', marker='o', s=30)

        # Add colorbar
        self.figure.colorbar(scatter, ax=self.plot)

        # Draw connections between consecutive points
        for i in range(len(vectors)-1):
            self.plot.plot([vectors[i,0], vectors[i+1,0]],
                         [vectors[i,1], vectors[i+1,1]],
                         [vectors[i,2], vectors[i+1,2]], 'gray', alpha=0.5)

        self.update_axis_labels()
        self.canvas.draw()

    def update_axis_labels(self):
        """Update axis labels with PCA variance ratios."""
        # This method assumes that PCA has been applied elsewhere and labels are updated accordingly.
        self.plot.set_xlabel("PCA 1")
        self.plot.set_ylabel("PCA 2")
        self.plot.set_zlabel("PCA 3")

# --------------------------- Enhanced 4D Visualizer --------------------------- #

class Enhanced4DVisualizer:
    """Enhanced 4D Visualizer with vector path animation and temporal controls."""

    def __init__(self):
        matplotlib.use('TkAgg')
        self.root = None
        self.fig = None
        self.ax = None
        self.scatter = None
        self.line_collection = None
        self.annotation = None
        self.current_frame = 0
        self.total_frames = 0
        self.pca = None
        self.cached_data = {}
        self.animation_speed = 1.0
        self.is_playing = False
        self.time_slider = None
        self.play_pause_btn = None
        self.animation_controller = None

    def setup_gui(self):
        if self.root is None:
            self.root = tk.Tk()
            self.root.title("Token Generation Visualization")

            # Create main container
            main_container = tk.Frame(self.root)
            main_container.pack(fill=tk.BOTH, expand=True)

            # Left panel for metrics and information
            info_panel = tk.Frame(main_container, width=300)
            info_panel.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)

            # Token Information Display
            tk.Label(info_panel, text="Token Information", font=('Arial', 10, 'bold')).pack(pady=5)
            self.token_info = tk.Text(info_panel, height=8, width=40, wrap=tk.WORD, state=tk.DISABLED)
            self.token_info.pack(pady=5)

            # PCA Components Information
            tk.Label(info_panel, text="PCA Components", font=('Arial', 10, 'bold')).pack(pady=5)
            self.pca_info = tk.Text(info_panel, height=8, width=40, wrap=tk.WORD, state=tk.DISABLED)
            self.pca_info.pack(pady=5)

            # Temporal Metrics
            tk.Label(info_panel, text="Temporal Metrics", font=('Arial', 10, 'bold')).pack(pady=5)
            self.temporal_info = tk.Text(info_panel, height=8, width=40, wrap=tk.WORD, state=tk.DISABLED)
            self.temporal_info.pack(pady=5)

            # Visualization panel
            viz_panel = tk.Frame(main_container)
            viz_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

            # Create figure
            self.fig = plt.Figure(figsize=(10, 8), dpi=120)
            self.ax = self.fig.add_subplot(111, projection='3d')

            # Create canvas
            canvas = FigureCanvasTkAgg(self.fig, master=viz_panel)
            canvas.draw()
            canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

            # Control panel
            control_frame = tk.Frame(viz_panel)
            control_frame.pack(side=tk.BOTTOM, fill=tk.X)

            # Timeline slider
            self.time_slider = tk.Scale(
                control_frame,
                from_=0,
                to=100,  # Will be updated with actual frame count
                orient=tk.HORIZONTAL,
                command=self.update_timeline,
                length=400
            )
            self.time_slider.pack(side=tk.BOTTOM, fill=tk.X, padx=5, pady=5)

            # Playback controls
            button_frame = tk.Frame(control_frame)
            button_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=5, pady=5)

            self.play_pause_btn = tk.Button(
                button_frame,
                text="Play",
                command=self.toggle_animation,
                width=8
            )
            self.play_pause_btn.pack(side=tk.LEFT, padx=5)

            tk.Button(
                button_frame,
                text="Reset",
                command=self.reset_visualization,
                width=8
            ).pack(side=tk.LEFT, padx=5)

            # Speed control
            tk.Label(button_frame, text="Animation Speed:").pack(side=tk.LEFT, padx=5)
            speed_scale = tk.Scale(
                button_frame,
                from_=0.1,
                to=3.0,
                resolution=0.1,
                orient=tk.HORIZONTAL,
                command=self.update_speed,
                length=150
            )
            speed_scale.set(1.0)
            speed_scale.pack(side=tk.LEFT, padx=5)

            # Mouse events
            self.fig.canvas.mpl_connect('motion_notify_event', self.on_hover)

            # Initialize Animation Controller
            self.animation_controller = AnimationController(self)

    def update_timeline(self, value):
        """Update visualization based on timeline position."""
        frame = int(float(value))
        if frame != self.current_frame:
            self.current_frame = frame
            self.update_visualization(frame)
            self.update_temporal_info(frame)

    def update_visualization(self, frame):
        """Update the visualization state for the given frame."""
        if not self.cached_data.get('projected_states') is None:
            current_points = self.cached_data['projected_states'][:frame+1]
            current_colors = self.cached_data['colors'][:frame+1]

            # Clear previous plot
            self.ax.cla()

            # Plot points up to current frame
            self.scatter = self.ax.scatter(
                current_points[:, 0],
                current_points[:, 1],
                current_points[:, 2],
                c=current_colors,
                cmap='viridis',
                marker='o',
                s=30
            )

            # Draw path up to current frame
            if len(current_points) > 1:
                segments = np.array([[current_points[i], current_points[i+1]] 
                                   for i in range(len(current_points)-1)])
                self.line_collection = Line3DCollection(
                    segments,
                    cmap='coolwarm',
                    linewidth=2,
                    alpha=0.7
                )
                # Optionally, color the lines based on entropy or another metric
                self.line_collection.set_array(current_colors[:-1])
                self.ax.add_collection3d(self.line_collection)

            # Update labels and view
            self.update_axis_labels()
            self.fig.canvas.draw()

            # Update token information
            if self.cached_data['tokens'] and frame < len(self.cached_data['tokens']):
                token = self.cached_data['tokens'][frame]
                entropy = self.cached_data['colors'][frame]
                # Adjusted to treat 'token' as a string, not a dict
                # Removed 'probability' as it's not part of 'tokens'
                self.token_info.config(state=tk.NORMAL)
                self.token_info.delete(1.0, tk.END)
                self.token_info.insert(tk.END, f"Token: {token}\nEntropy: {entropy:.4f}")
                self.token_info.config(state=tk.DISABLED)

    def update_temporal_info(self, frame):
        """Update temporal information display."""
        if self.cached_data.get('tokens') is not None and frame < len(self.cached_data['tokens']):
            current_token = self.cached_data['tokens'][frame]
            current_entropy = self.cached_data['colors'][frame]
            progress = (frame / self.total_frames) * 100 if self.total_frames > 0 else 0

            info_text = f"Frame: {frame}/{self.total_frames}\n"
            info_text += f"Current Token: {current_token}\n"
            info_text += f"Entropy: {current_entropy:.4f}\n"
            info_text += f"Progress: {progress:.1f}%"

            self.temporal_info.config(state=tk.NORMAL)
            self.temporal_info.delete(1.0, tk.END)
            self.temporal_info.insert(tk.END, info_text)
            self.temporal_info.config(state=tk.DISABLED)

    def toggle_animation(self):
        """Toggle animation play/pause state."""
        self.animation_controller.toggle_animation()

    def reset_visualization(self):
        """Reset visualization to initial state."""
        self.current_frame = 0
        self.time_slider.set(0)
        self.animation_controller.stop_animation()
        self.update_visualization(0)
        self.update_temporal_info(0)

    def update_speed(self, value):
        """Update animation speed."""
        self.animation_speed = float(value)

    def plot_4d_visualization(self, hidden_states, entropies, tokens, time_steps):
        """Initialize visualization with data."""
        if len(hidden_states) < 3:
            logger.warning("Insufficient samples for visualization")
            return

        self.setup_gui()

        # Prepare data
        self.pca = PCA(n_components=3)
        flattened_states = hidden_states.reshape(hidden_states.shape[0], -1)
        projected_states = self.pca.fit_transform(flattened_states)

        # Store data for animation
        self.cached_data = {
            'projected_states': projected_states,
            'colors': np.array(entropies),
            'tokens': tokens,
            'time_steps': time_steps
        }

        self.total_frames = len(projected_states) - 1
        self.time_slider.config(to=self.total_frames)

        # Initial visualization
        self.reset_visualization()
        self.update_pca_info()

    def update_axis_labels(self):
        """Update axis labels with PCA variance ratios."""
        if self.pca:
            var_ratio = self.pca.explained_variance_ratio_
            self.ax.set_xlabel(f"PCA 1 ({var_ratio[0]:.1%} var)")
            self.ax.set_ylabel(f"PCA 2 ({var_ratio[1]:.1%} var)")
            self.ax.set_zlabel(f"PCA 3 ({var_ratio[2]:.1%} var)")

    def update_pca_info(self):
        """Update PCA information display."""
        if self.pca:
            var_ratio = self.pca.explained_variance_ratio_
            cum_var = np.cumsum(var_ratio)

            info_text = "PCA Components Analysis:\n\n"
            info_text += f"PCA 1: {var_ratio[0]:.1%}\n"
            info_text += f"PCA 2: {var_ratio[1]:.1%}\n"
            info_text += f"PCA 3: {var_ratio[2]:.1%}\n"
            info_text += f"\nTotal Variance: {cum_var[2]:.1%}"

            self.pca_info.config(state=tk.NORMAL)
            self.pca_info.delete(1.0, tk.END)
            self.pca_info.insert(tk.END, info_text)
            self.pca_info.config(state=tk.DISABLED)

    def on_hover(self, event):
        """Handle mouse hover events."""
        if event.inaxes != self.ax or self.cached_data.get('projected_states') is None:
            return

        if self.annotation:
            self.annotation.remove()
            self.annotation = None

        if self.scatter:
            cont, ind = self.scatter.contains(event)
            if cont:
                point_idx = ind['ind'][0]
                if point_idx < len(self.cached_data['tokens']):
                    token = self.cached_data['tokens'][point_idx]
                    entropy = self.cached_data['colors'][point_idx]
                    # Removed 'probability' since it's not part of 'tokens'
                    hover_text = f"Token: {token}\nEntropy: {entropy:.4f}"

                    pos = self.cached_data['projected_states'][point_idx]
                    self.annotation = self.ax.text(
                        pos[0], pos[1], pos[2],
                        hover_text,
                        bbox=dict(facecolor='white', alpha=0.7)
                    )
                    self.fig.canvas.draw_idle()

    def close(self):
        """Clean up resources."""
        if self.root:
            self.root.quit()
            self.root.destroy()
        plt.close(self.fig)

# --------------------------- Animation Controller --------------------------- #

class AnimationController:
    def __init__(self, visualizer):
        self.visualizer = visualizer
        self.is_playing = False
        self.current_frame = 0
        self.total_frames = 0
        self.animation_speed = 1.0
        self.last_update_time = 0
        self._animation_id = None

    def animate(self):
        """Improved animation handler with proper timing"""
        current_time = time.time()
        time_delta = current_time - self.last_update_time

        if self.is_playing and time_delta >= (1.0 / (30 * self.animation_speed)):  # 30 FPS target
            self.last_update_time = current_time
            if self.visualizer.current_frame < self.visualizer.total_frames:
                self.visualizer.current_frame += 1
                self.visualizer.time_slider.set(self.visualizer.current_frame)
                self._animation_id = self.visualizer.root.after(int(33.3 / self.animation_speed), self.animate)
            else:
                self.stop_animation()

    def start_animation(self):
        """Start animation playback"""
        if not self.is_playing:
            self.is_playing = True
            self.visualizer.last_update_time = time.time()
            self.animate()
            self.visualizer.play_pause_btn.config(text="Pause")

    def stop_animation(self):
        """Stop animation playback"""
        self.is_playing = False
        if self._animation_id:
            self.visualizer.root.after_cancel(self._animation_id)
            self._animation_id = None
        self.visualizer.play_pause_btn.config(text="Play")

    def toggle_animation(self):
        """Toggle between play and pause states"""
        if self.is_playing:
            self.stop_animation()
        else:
            self.start_animation()

# --------------------------- Token Analysis Visualizer --------------------------- #

class TokenAnalysisVisualizer:
    def __init__(self, root, parent):
        self.root = root
        self.parent = parent
        self.token_frame = None
        self.token_canvas = None
        self.token_scroll = None
        self.setup_token_view()

    def setup_token_view(self):
        """Create token analysis view"""
        self.token_frame = tk.Frame(self.parent, width=300)
        self.token_frame.pack(side=tk.RIGHT, fill=tk.Y)

        # Create scrollable canvas for tokens
        self.token_canvas = tk.Canvas(self.token_frame, width=280)
        self.token_scroll = tk.Scrollbar(self.token_frame, orient=tk.VERTICAL, 
                                       command=self.token_canvas.yview)
        
        # Configure scrolling
        self.token_canvas.configure(yscrollcommand=self.token_scroll.set)
        self.token_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.token_canvas.pack(side=tk.LEFT, fill=tk.Y, expand=True)

    def visualize_token_processing(self, token_data):
        """Display token processing information"""
        frame = tk.Frame(self.token_canvas)
        row = 0
        
        for token_info in token_data:
            # Token text with entropy-based background color
            entropy_color = self.get_entropy_color(token_info['entropy'])
            token_label = tk.Label(frame, text=token_info['text'],
                                 bg=entropy_color, width=20, anchor='w')
            token_label.grid(row=row, column=0, sticky='w')
            
            # Metadata display
            meta_text = f"E:{token_info['entropy']:.2f} T:{token_info['temperature']:.2f}"
            meta_label = tk.Label(frame, text=meta_text, width=15, anchor='w')
            meta_label.grid(row=row, column=1, sticky='w')
            
            # Processing indicators
            if token_info.get('is_special'):
                indicator = "⚡"  # Special token indicator
            elif token_info['entropy'] > 3.0:
                indicator = "🤔"  # High entropy indicator
            else:
                indicator = "✓"   # Normal token indicator
            
            ind_label = tk.Label(frame, text=indicator, width=3)
            ind_label.grid(row=row, column=2)
            
            row += 1

        # Update canvas scroll region
        self.token_canvas.create_window((0, 0), window=frame, anchor='nw')
        frame.update_idletasks()
        self.token_canvas.configure(scrollregion=self.token_canvas.bbox('all'))

    def get_entropy_color(self, entropy):
        """Generate color based on entropy value"""
        # Low entropy = blue, High entropy = red
        r = min(255, int(entropy * 25.5))
        b = max(0, 255 - int(entropy * 25.5))
        return f'#{r:02x}00{b:02x}'

# --------------------------- Macro Processor --------------------------- #

class MacroProcessor:
    def __init__(self, tokenizer, model, entropy_analyzer):
        self.tokenizer = tokenizer
        self.model = model
        self.entropy_analyzer = entropy_analyzer

    def generate_macro_response(self, prompt, model, tokenizer, quality_manager, loss_weighting_system, visualizer, user_config):
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
        tokens = []  # To store decoded tokens

        with torch.no_grad():
            for step in tqdm(range(max_tokens), desc="Generating Response", unit="token"):
                outputs = model(generated_ids, output_hidden_states=True)
                logits = outputs.logits[:, -1, :].to(device)

                # Calculate metrics
                entropy = calculate_entropy(logits)
                varentropy = calculate_varentropy(entropy)
                kl_div = calculate_kl_divergence(logits)
                perplexity = calculate_perplexity(logits)

                # Adjust temperature and sampling parameters
                temperature = adjust_temperature_based_on_entropy(entropy.item(), 
                                                                   low_threshold=user_config.entropy_thresholds['low'], 
                                                                   high_threshold=user_config.entropy_thresholds['high'])
                top_k, top_p = adjust_sampling_parameters(entropy.item(), 
                                                         low_k=user_config.top_k['low'], 
                                                         high_k=user_config.top_k['high'], 
                                                         low_p=user_config.top_p['low'], 
                                                         high_p=user_config.top_p['high'])

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

                # Ensure token_id is on the same device as model
                token_id = token_id.to(device)

                generated_ids = torch.cat([generated_ids, token_id], dim=1)

                # Log token details
                token_log.append({
                    "token_id": token_id.item(),
                    "text": tokenizer.decode(token_id.item()),
                    "entropy": entropy.item(),
                    "temperature": temperature,
                    "top_k": top_k,
                    "top_p": top_p,
                    "probability": torch.softmax(logits, dim=-1)[0, token_id.item()].item(),
                    "is_special": token_id.item() in tokenizer.all_special_ids
                })

                # Decode token and store
                token_text = tokenizer.decode(token_id.item())
                tokens.append(token_text)

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
                    last_hidden_state = outputs.hidden_states[-1][:, -1, :].to(device)  # Shape: [batch_size, hidden_dim]
                    hidden_states.append(last_hidden_state.detach().cpu().numpy())
                else:
                    hidden_states.append(np.array([]))  # Fallback to empty array if hidden_states is None
                time_steps.append(step)

                # Optional: Visualize at certain intervals
                if user_config.interactive_visuals and step % user_config.visualization_frequency == 0:
                    if len(hidden_states) > 0 and hidden_states[-1].size > 0:
                        hidden_states_cpu = np.concatenate([hs for hs in hidden_states if hs.size > 0], axis=0)
                        visualizer.plot_4d_visualization(hidden_states_cpu, entropies, tokens, time_steps)

        # Logging token information
        for log_entry in token_log:
            logger.info(f"Token: {log_entry['text']} (ID: {log_entry['token_id']}), Entropy: {log_entry['entropy']:.2f}, "
                        f"Temperature: {log_entry['temperature']:.2f}, top_k: {log_entry['top_k']}, top_p: {log_entry['top_p']}, "
                        f"Probability: {log_entry['probability']:.2f}, Special: {log_entry['is_special']}")

        # Decode the generated tokens to get the response
        response = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        response = response.split("AI:")[-1].strip()
        response = quality_manager.remove_eot_tokens(response)

        # Concatenate hidden states and prepare for visualization
        if hidden_states:
            hidden_states_cpu = np.concatenate([hs for hs in hidden_states if hs.size > 0], axis=0)
        else:
            hidden_states_cpu = np.array([])

        # Run visualization
        if user_config.interactive_visuals and hidden_states_cpu.size > 0 and entropies:
            visualizer.plot_4d_visualization(hidden_states_cpu, entropies, tokens, time_steps)

        # Clear cache and collect garbage
        torch.cuda.empty_cache()
        gc.collect()

        return response, entropies, varentropies, kl_divs, perplexities, hidden_states_cpu

    def _analyze_attention_pattern(self, attention_matrix):
        """Analyze attention patterns for semantic understanding"""
        # Convert attention to numpy for analysis
        att_np = attention_matrix.detach().cpu().numpy()

        # Calculate pattern metrics
        attention_stats = {
            'mean': np.mean(att_np),
            'std': np.std(att_np),
            'max_attention': np.max(att_np),
            'attention_entropy': -np.sum(att_np * np.log(att_np + 1e-10))
        }

        # Detect attention patterns
        patterns = {
            'focused': np.max(att_np) > 0.8,
            'dispersed': attention_stats['attention_entropy'] > 2.0,
            'structured': attention_stats['std'] < 0.1
        }

        return {
            'stats': attention_stats,
            'patterns': patterns,
            'strength': np.mean(list(patterns.values()))
        }

    def _calculate_semantic_coherence(self, hidden_states):
        """Calculate semantic coherence of generated text"""
        # Get last layer representation
        hidden_np = hidden_states.detach().cpu().numpy()

        # Calculate cosine similarity between consecutive states
        similarities = []
        for i in range(hidden_np.shape[0] - 1):
            sim = cosine_similarity(
                hidden_np[i].reshape(1, -1),
                hidden_np[i + 1].reshape(1, -1)
            )[0][0]
            similarities.append(sim)

        return np.mean(similarities) if similarities else 0.0

    def _get_token_vector(self, hidden_states):
        """Extract vector representation for token"""
        return hidden_states[-1].detach().cpu().numpy()

    def _analyze_sequence_patterns(self, generated_ids):
        """Analyze patterns in generated sequence"""
        if len(generated_ids) < 3:
            return {'patterns': [], 'strength': 0.0}

        # Convert to tokens for analysis
        tokens = [self.tokenizer.decode([id]) for id in generated_ids]

        # Find repeated patterns
        patterns = []
        for length in range(2, min(len(tokens), 5)):
            for i in range(len(tokens) - length + 1):
                pattern = tokens[i:i + length]
                pattern_str = ''.join(pattern)

                # Count pattern occurrences
                count = sum(
                    1 for j in range(len(tokens) - length + 1)
                    if ''.join(tokens[j:j + length]) == pattern_str
                )

                if count > 1:
                    patterns.append({
                        'pattern': pattern_str,
                        'length': length,
                        'count': count,
                        'position': i
                    })

        return {
            'patterns': patterns,
            'strength': len(patterns) / max(1, len(tokens))
        }

    def _select_optimal_token(self, logits, token_state, state, context):
        """Select optimal next token based on all available metrics"""
        # Get base probabilities
        probs = F.softmax(logits, dim=-1).to(device)

        # Apply attention-based weighting
        attention_weights = self._get_attention_weights(token_state['patterns'])
        weighted_probs = probs * attention_weights

        # Apply pattern-based adjustments
        pattern_weights = self._get_pattern_weights(state.pattern_data)
        weighted_probs = weighted_probs * pattern_weights

        # Apply semantic coherence adjustments
        semantic_weights = self._get_semantic_weights(state.vector_data)
        weighted_probs = weighted_probs * semantic_weights

        # Apply context-based adjustments
        context_weights = self._get_context_weights(context)
        weighted_probs = weighted_probs * context_weights

        # Sample from adjusted distribution
        return torch.multinomial(weighted_probs, 1)

    def _get_attention_weights(self, attention_patterns):
        """Generate weights based on attention patterns"""
        if not attention_patterns:
            return 1.0

        recent_patterns = attention_patterns[-5:]
        pattern_strength = np.mean([p['strength'] for p in recent_patterns])

        return torch.tensor(pattern_strength).to(device)

    def _get_pattern_weights(self, pattern_data):
        """Generate weights based on detected patterns"""
        if not pattern_data:
            return 1.0

        recent_patterns = pattern_data[-5:]
        pattern_strength = np.mean([p['strength'] for p in recent_patterns])

        return torch.tensor(pattern_strength).to(device)

    def _get_semantic_weights(self, vector_data):
        """Generate weights based on semantic coherence"""
        if len(vector_data) < 2:
            return 1.0

        recent_vectors = [d['vector'] for d in vector_data[-5:]]
        coherence = np.mean([
            cosine_similarity(v1.reshape(1, -1), v2.reshape(1, -1))[0][0]
            for v1, v2 in zip(recent_vectors[:-1], recent_vectors[1:])
        ])

        return torch.tensor(coherence).to(device)

    def _get_context_weights(self, context):
        """Generate weights based on conversation context"""
        if not context:
            return 1.0

        # Calculate context relevance
        relevance = np.mean([c.get('relevance', 0.0) for c in context])

        return torch.tensor(relevance).to(device)

    def _create_token_info(self, token, entropy, varentropy, attention_pattern, vector):
        """Create comprehensive token information"""
        return {
            'token': token.item(),
            'text': self.tokenizer.decode([token.item()]),
            'entropy': entropy.item(),
            'varentropy': varentropy.item(),
            'attention_pattern': attention_pattern,
            'vector': vector,
            'timestamp': time.time()
        }

# --------------------------- Main Execution --------------------------- #

class State:
    """Class to hold various states and configurations."""
    def __init__(self, user_config, visualizer, quality_manager, context_manager, entropy_analyzer):
        self.user_config = user_config
        self.visualizer = visualizer
        self.quality_manager = quality_manager
        self.context_manager = context_manager
        self.entropy_analyzer = entropy_analyzer
        self.pattern_data = []
        self.vector_data = []

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
    
    # Prepare the special tokens as strings instead of IDs
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

def check_flash_attention():
    """Checks if Flash Attention is available and logs the status."""
    try:
        import flash_attn
        logger.info("Flash Attention is available and enabled.")
    except ImportError:
        logger.warning("Flash Attention is not available. Using standard scaled dot product attention.")

def sanitize_input(user_input):
    """Sanitizes the user input to prevent injection of unwanted tokens or patterns."""
    sanitized = re.sub(r'[^\w\s.,!?]', '', user_input)
    return sanitized[:500]

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
    tokens = []  # To store decoded tokens

    with torch.no_grad():
        for step in tqdm(range(max_tokens), desc="Generating Response", unit="token"):
            outputs = model(generated_ids, output_hidden_states=True)
            logits = outputs.logits[:, -1, :].to(device)

            # Calculate metrics
            entropy = calculate_entropy(logits)
            varentropy = calculate_varentropy(entropy)
            kl_div = calculate_kl_divergence(logits)
            perplexity = calculate_perplexity(logits)

            # Adjust temperature and sampling parameters
            temperature = adjust_temperature_based_on_entropy(entropy.item(), 
                                                               low_threshold=user_config.entropy_thresholds['low'], 
                                                               high_threshold=user_config.entropy_thresholds['high'])
            top_k, top_p = adjust_sampling_parameters(entropy.item(), 
                                                     low_k=user_config.top_k['low'], 
                                                     high_k=user_config.top_k['high'], 
                                                     low_p=user_config.top_p['low'], 
                                                     high_p=user_config.top_p['high'])

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

            # Ensure token_id is on the same device as model
            token_id = token_id.to(device)

            generated_ids = torch.cat([generated_ids, token_id], dim=1)

            # Log token details
            token_log.append({
                "token_id": token_id.item(),
                "text": tokenizer.decode(token_id.item()),
                "entropy": entropy.item(),
                "temperature": temperature,
                "top_k": top_k,
                "top_p": top_p,
                "probability": torch.softmax(logits, dim=-1)[0, token_id.item()].item(),
                "is_special": token_id.item() in tokenizer.all_special_ids
            })

            # Decode token and store
            token_text = tokenizer.decode(token_id.item())
            tokens.append(token_text)

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
                last_hidden_state = outputs.hidden_states[-1][:, -1, :].to(device)  # Shape: [batch_size, hidden_dim]
                hidden_states.append(last_hidden_state.detach().cpu().numpy())
            else:
                hidden_states.append(np.array([]))  # Fallback to empty array if hidden_states is None
            time_steps.append(step)

            # Optional: Visualize at certain intervals
            if user_config.interactive_visuals and step % user_config.visualization_frequency == 0:
                if len(hidden_states) > 0 and hidden_states[-1].size > 0:
                    hidden_states_cpu = np.concatenate([hs for hs in hidden_states if hs.size > 0], axis=0)
                    visualizer.plot_4d_visualization(hidden_states_cpu, entropies, tokens, time_steps)

    # Logging token information
    for log_entry in token_log:
        logger.info(f"Token: {log_entry['text']} (ID: {log_entry['token_id']}), Entropy: {log_entry['entropy']:.2f}, "
                    f"Temperature: {log_entry['temperature']:.2f}, top_k: {log_entry['top_k']}, top_p: {log_entry['top_p']}, "
                    f"Probability: {log_entry['probability']:.2f}, Special: {log_entry['is_special']}")

    # Decode the generated tokens to get the response
    response = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    response = response.split("AI:")[-1].strip()
    response = quality_manager.remove_eot_tokens(response)

    # Concatenate hidden states and prepare for visualization
    if hidden_states:
        hidden_states_cpu = np.concatenate([hs for hs in hidden_states if hs.size > 0], axis=0)
    else:
        hidden_states_cpu = np.array([])

    # Run visualization
    if user_config.interactive_visuals and hidden_states_cpu.size > 0 and entropies:
        visualizer.plot_4d_visualization(hidden_states_cpu, entropies, tokens, time_steps)

    # Clear cache and collect garbage
    torch.cuda.empty_cache()
    gc.collect()

    return response, entropies, varentropies, kl_divs, perplexities, hidden_states_cpu

def interactive_query(model, tokenizer, quality_manager, context_manager, state):
    """Handles the interactive query loop with the user."""
    print("\n--- LLaMA Instruct Model Interactive Query ---")
    print("Type 'exit' to quit.\n")

    # Initialize Adaptive Weighting System
    loss_weighting_system = AdaptiveWeightingSystem(initial_weights=state.user_config.initial_weights)

    # Initialize Macro Processor
    macro_processor = MacroProcessor(tokenizer, model, state.entropy_analyzer)

    # Initialize Visualizer if enabled
    visualizer = state.visualizer if state.user_config.interactive_visuals else None

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
            sanitized_input = sanitize_input(user_input)
            prompt = context_manager.get_dynamic_prompt(sanitized_input)
            response, entropies, varentropies, kl_divs, perplexities, hidden_states_cpu = generate_macroprocessed_response(
                prompt, model, tokenizer, quality_manager, loss_weighting_system, visualizer, state.user_config
            )
            context_manager.update_context(sanitized_input, response, entropies)
            print(f"Model Response: {response}\n")
        except Exception as e:
            logger.error(f"An error occurred during response generation: {e}")
            print(f"An error occurred: {e}")

    # Clean up visualizer resources if necessary
    if visualizer:
        visualizer.close()

# --------------------------- Main Function --------------------------- #

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
    if "<|finetune_right_pad_id|>" in tokenizer.get_vocab():
        model.resize_token_embeddings(len(tokenizer))
        logger.info("Resized model token embeddings to accommodate the new pad_token.")
    else:
        logger.info("pad_token already exists in the tokenizer's vocabulary. No need to resize embeddings.")

    # Check for Flash Attention
    check_flash_attention()

    # Initialize Response Quality Manager
    quality_manager = ImprovedResponseQualityManager(tokenizer, model)

    # Initialize Entropy Analyzer
    entropy_analyzer = AdvancedEntropyAnalyzer()

    # Initialize Context Manager
    context_manager = AdvancedContextManager(model, tokenizer, entropy_analyzer)

    # Initialize Visualization System
    visualizer = Enhanced4DVisualizer()
    state = State(user_config, visualizer, quality_manager, context_manager, entropy_analyzer)

    if user_config.interactive_visuals:
        visualizer.setup_gui()

    logger.info("Model loaded successfully. You can now query the model.")

    # Start interactive query loop
    interactive_query(model, tokenizer, quality_manager, context_manager, state)

if __name__ == "__main__":
    main()
