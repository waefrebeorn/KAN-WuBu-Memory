import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.cuda.amp as amp
from safetensors.torch import load_file
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
import logging
from pathlib import Path
import json
import numpy as np
from collections import deque
from datetime import datetime
import time
import traceback
import gc
import os
import sys
import warnings
import re
from torch.cuda.amp import GradScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Logging Configuration
class LogFilter(logging.Filter):
    def __init__(self, ignore_patterns=None):
        super().__init__()
        self.ignore_patterns = ignore_patterns or []

    def filter(self, record):
        return not any(pattern in record.getMessage() for pattern in self.ignore_patterns)

def setup_logging():
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    file_handler = logging.FileHandler('llama_tool.log', mode='a', encoding='utf-8')
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.WARNING)
    console_formatter = logging.Formatter('%(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)


    ignore_patterns = [
        "matplotlib",
        "PIL.PngImagePlugin",
        "expandable_segments not supported",
        "weights_only",
        "half",
        "train_kan_step -",
        "Torch was not compiled with flash attention."
        "Torch was not compiled with flash attention."
    ]

    for handler in logger.handlers:
        handler.addFilter(LogFilter(ignore_patterns))

    warnings.filterwarnings("ignore", category=UserWarning, message="Torch was not compiled with flash attention.*")
    warnings.filterwarnings("ignore", category=FutureWarning, message="You are using `torch.load` with `weights_only=False`.*")

    logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)
    logging.getLogger('matplotlib.pyplot').setLevel(logging.WARNING)
    logging.getLogger('PIL.PngImagePlugin').setLevel(logging.WARNING)

setup_logging()

class EmotionalState:
    def __init__(self, dimensions=('valence', 'arousal', 'dominance'), device="cuda"):
        self.dimensions = dimensions
        self.device = device
        self.position = torch.zeros(1, len(dimensions), device=device, dtype=torch.float16)
        self.velocity = torch.zeros(1, len(dimensions), device=device, dtype=torch.float16)

    def update(self, feedback, max_speed=0.1):
        feedback_vector = torch.as_tensor(feedback, device=self.device, dtype=torch.float16)
        if feedback_vector.dim() == 1:
            feedback_vector = feedback_vector.unsqueeze(0)
        if feedback_vector.size(0) != self.position.size(0):
            feedback_vector = feedback_vector.expand(self.position.size(0), -1)

        self.velocity += feedback_vector * 0.1 + torch.randn_like(self.velocity) * 0.01
        self.velocity = torch.clamp(self.velocity, -max_speed, max_speed)
        self.position += self.velocity
        norm = torch.norm(self.position, dim=1, keepdim=True)
        self.position = torch.where(norm > 1, self.position / norm, self.position)

    def get_emotion(self):
        valence, arousal, dominance = self.position.squeeze().tolist()
        if abs(valence) < 0.3 and abs(arousal) < 0.3 and abs(dominance) < 0.3:
            return "Neutral"
        elif valence > 0:
            if arousal > 0:
                return "Happy" if dominance > 0 else "Excited"
            else:
                return "Relaxed" if dominance > 0 else "Calm"
        else:
            if arousal > 0:
                return "Angry" if dominance > 0 else "Frustrated"
            else:
                return "Sad" if dominance > 0 else "Depressed"

    def get_embedding(self, batch_size=1):
        # Ensure the embedding is of shape (batch_size, num_emotional_dimensions)
        if batch_size != self.position.size(0):
            return self.position.expand(batch_size, -1)
        return self.position

    def __str__(self):
        emotion = self.get_emotion()
        values = self.position.squeeze().tolist()
        return f"Emotion: {emotion}, Values: {dict(zip(self.dimensions, values))}"
        
class OverfitDetector:
    def __init__(self, window_size=50, threshold=0.05):
        self.window_size = window_size
        self.threshold = threshold
        self.training_losses = deque(maxlen=window_size)
        self.validation_losses = deque(maxlen=window_size)

    def add_losses(self, training_loss, validation_loss):
        self.training_losses.append(training_loss)
        self.validation_losses.append(validation_loss)

    def is_overfitting(self):
        if len(self.training_losses) < self.window_size:
            return False

        train_trend = np.polyfit(range(self.window_size), self.training_losses, 1)[0]
        val_trend = np.polyfit(range(self.window_size), self.validation_losses, 1)[0]

        return (
            train_trend < 0
            and val_trend > 0
            and (val_trend - train_trend) > self.threshold
        )

class TensorSwapper:
    def __init__(self, device):
        self.device = device
        self.cpu_tensors = {}

    def swap_out(self, name, tensor):
        self.cpu_tensors[name] = tensor.detach().cpu()
        del tensor
        torch.cuda.empty_cache()

    def swap_in(self, name):
        if name in self.cpu_tensors:
            tensor = self.cpu_tensors[name].to(self.device)
            del self.cpu_tensors[name]
            return tensor
        else:
            raise KeyError(f"Tensor {name} not found in CPU storage")


class ResponseQualityManager:
    def __init__(self, kan_model, tokenizer, refusal_detector, emotional_state, memory_manager):
        self.kan_model = kan_model
        self.tokenizer = tokenizer
        self.refusal_detector = refusal_detector
        self.emotional_state = emotional_state
        self.memory_manager = memory_manager
        self.invalid_response_count = 0

        # Initialize TF-IDF Vectorizer for relevance calculation
        self.tfidf_vectorizer = TfidfVectorizer()

    def evaluate_response(self, user_input, response_tokens, context):
        """Evaluate the quality of the response based on multiple criteria."""
        response = self.tokenizer.decode(response_tokens, skip_special_tokens=True)

        # Check refusal score using the existing refusal detector
        refusal_score = self.refusal_detector.detect_refusal(response_tokens)
        quality_metrics = {
            'is_refusal': refusal_score > 0.5,
            'relevance_score': self.calculate_relevance(user_input, response),
            'length': len(response_tokens),
            'structure': self.has_proper_structure(response),
            'perplexity': self.calculate_perplexity(response_tokens),
        }

        # Combine all quality metrics into a final decision
        is_valid = self.is_valid_response(quality_metrics)

        # Detect garbled or anomalous output
        if self.detect_garbled_output(response):
            quality_metrics['garbled_output'] = True
            is_valid = False  # Mark the response as invalid if garbled
        else:
            quality_metrics['garbled_output'] = False

        return is_valid, quality_metrics

    def calculate_relevance(self, user_input, response):
        """Calculate the relevance score between the user input and the generated response."""
        # Token Overlap Measure
        user_tokens = set(self.tokenizer.tokenize(user_input))
        response_tokens = set(self.tokenizer.tokenize(response))
        overlap = len(user_tokens.intersection(response_tokens))
        overlap_score = overlap / max(len(user_tokens), 1)  # Avoid division by zero

        # Calculate Cosine Similarity Using TF-IDF (Advanced)
        combined_texts = [user_input, response]
        tfidf_matrix = self.tfidf_vectorizer.fit_transform(combined_texts)
        cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]

        # Combine both metrics to calculate the final relevance score
        relevance_score = 0.5 * overlap_score + 0.5 * cosine_sim

        logging.info(f"Relevance Score: Overlap = {overlap_score:.4f}, Cosine Similarity = {cosine_sim:.4f}, Final Score = {relevance_score:.4f}")
        return relevance_score

    def detect_garbled_output(self, response):
        """Detect if the response contains garbled or nonsensical text."""
        if re.search(r'[^\x00-\x7F]+', response):  # Check for non-ASCII characters
            return True
        if len(response.split()) < 3:  # Too few words in a long sequence
            return True
        if response.count('.') / len(response.split()) > 0.5:  # Excessive use of punctuation
            return True
        return False

    def has_proper_structure(self, response):
        try:
            decoded_text = self.tokenizer.decode(response, skip_special_tokens=True)
            sentences = re.split(r'(?<=[.!?])\s+', decoded_text.strip())
            
            if not sentences:
                logging.warning("No complete sentences found in the response.")
                return False
        
            if not sentences[0] or not sentences[0][0].isupper():
                logging.warning(f"First sentence doesn't start with a capital letter: '{sentences[0]}'")
                return False
        
            if not sentences[-1] or sentences[-1][-1] not in '.!?':
                logging.warning(f"Last sentence doesn't end with proper punctuation: '{sentences[-1]}'")
                return False
        
            proper_sentences = sum(1 for s in sentences if s and s[0].isupper() and s[-1] in '.!?')
            proper_ratio = proper_sentences / len(sentences)
        
            if proper_ratio < 0.5:
                logging.warning(f"Only {proper_ratio:.2f} of sentences have proper structure.")
                return False
                
            return True
        except Exception as e:
            logging.error(f"Error in has_proper_structure: {str(e)}")
            return False
            

    def calculate_perplexity(self, response_tokens):
        """Calculate the perplexity of the response based on the logits."""
        try:
            inputs = torch.tensor([response_tokens]).to(self.kan_model.device)
            with torch.no_grad():
                outputs = self.kan_model.model(inputs, labels=inputs)
                loss = outputs.loss
                perplexity = torch.exp(loss)
            return perplexity.item()
        except Exception as e:
            logging.error(f"Error calculating perplexity: {str(e)}")
            return float('inf')

    def is_valid_response(self, quality_metrics):
        """Determine if a response is valid based on quality metrics."""
        if quality_metrics['is_refusal']:
            return False
        if quality_metrics['perplexity'] > 100:  # Set a threshold for acceptable perplexity
            return False
        if quality_metrics['relevance_score'] < 0.3:  # Set a threshold for relevance
            return False
        if not quality_metrics['structure']:  # Check if response has proper structure
            return False
        if quality_metrics['garbled_output']:  # Check if the output is marked as garbled
            return False
        return True

    def corrective_training(self, user_input, response_tokens, corrective_response):
        """Perform corrective training on an invalid response."""
        try:
            # Encode the corrective response as the target for training
            target_ids = self.tokenizer.encode(corrective_response, return_tensors="pt").to(self.kan_model.device)

            # Perform a training step using the invalid response and the corrective response
            lm_loss, refusal_loss = self.kan_model.train_kan_step(
                torch.tensor([response_tokens], dtype=torch.long).to(self.kan_model.device),
                target_ids,
                refusal_score=0.0  # Assume no refusal in this corrective context
            )
            return lm_loss, refusal_loss
        except Exception as e:
            logging.error(f"Error during corrective training: {str(e)}")
            return None, None

    def adaptive_regeneration(self, user_input, context, max_attempts=3):
        """Regenerate a response until a valid one is found or maximum attempts are reached."""
        for attempt in range(max_attempts):
            # Generate a new response using the KAN model
            response_tokens, response_entropy = self.kan_model.generate_response(user_input)
            
            # Evaluate the generated response using the ResponseQualityManager
            is_valid, quality_metrics = self.evaluate_response(user_input, response_tokens, context)

            # If a valid response is found, return it
            if is_valid:
                return response_tokens, quality_metrics

            # Log invalid response and corrective training information
            logging.warning(f"Invalid response detected. Attempt {attempt + 1}/{max_attempts}. Triggering corrective training.")
            corrective_response = "I misunderstood. Can you rephrase that?"
            self.corrective_training(user_input, response_tokens, corrective_response)

        # After max attempts, return the last generated response with its metrics
        return response_tokens, quality_metrics
   


class SyntheticDayCycle:
    def __init__(self, cycle_length=100):
        self.cycle_length = cycle_length
        self.current_position = 0

    def update(self, amount):
        self.current_position = (self.current_position + amount) % self.cycle_length

    def get_time_of_day(self):
        return self.current_position / self.cycle_length

    def should_sleep(self):
        return 0.7 <= self.get_time_of_day() < 1.0

class RefusalDetector:
    def __init__(self, tokenizer, model):
        self.tokenizer = tokenizer
        self.model = model
        self.refusal_phrases = [
            "I'm sorry, but I can't",
            "I don't feel comfortable",
            "I'm not able to",
            "I cannot assist with",
            "I won't be able to",
            "I don't have the capability",
        ]

    def detect_refusal(self, response_tokens):
        if isinstance(response_tokens, torch.Tensor):
            response_tokens = response_tokens.tolist()
        elif isinstance(response_tokens, str):
            raise ValueError("response_tokens should be a list of token IDs, not a string")
    
        response_text = self.tokenizer.decode(response_tokens, skip_special_tokens=True)
        for phrase in self.refusal_phrases:
            if phrase.lower() in response_text.lower():
                return 1.0
    
        prompt = (
            f"On a scale of 0 to 1, how much does this response refuse or avoid the task? "
            f"0 means no refusal at all, 1 means complete refusal. Respond with just the number:\n\n"
            f"'{response_text}'\n\nRefusal score:"
        )
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
    
        with torch.no_grad():
            try:
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=5,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True,
                    eos_token_id=self.tokenizer.eos_token_id,
                    pad_token_id=self.tokenizer.pad_token_id,
                    use_cache=False,
                    return_dict_in_generate=True,
                    output_hidden_states=False,
                )
            except Exception as e:
                logging.error(f"Error during RefusalDetector.generate: {str(e)}")
                return 0.5
    
        response = self.tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
        try:
            score = float(response.split()[-1])
            return min(max(score, 0.0), 1.0)
        except ValueError:
            return 0.5


class EnhancedKAN(nn.Module):
    def __init__(self, hidden_size, num_emotional_dimensions, vocab_size, device):
        super().__init__()
        self.device = device
        self.hidden_size = hidden_size
        self.emotional_size = num_emotional_dimensions
        self.vocab_size = vocab_size
        self.influence_scale = 0.01

        # Initialize layers directly on the specified device to avoid meta issues
        self.refusal_override = nn.Linear(hidden_size + hidden_size + num_emotional_dimensions, hidden_size, dtype=torch.float16).to(self.device)
        self.output_modifier = nn.Linear(hidden_size, vocab_size, dtype=torch.float16).to(self.device)

    def forward(self, hidden_states, user_intent, emotional_state):
        """
        Forward pass of the KAN model with strict device checks and integration safety.
        """
        try:
            # Move tensors to the specified device
            hidden_states = hidden_states.to(self.device, dtype=torch.float16)
            user_intent = user_intent.to(self.device, dtype=torch.float16)
            position = emotional_state.get_embedding(hidden_states.size(0)).to(self.device, dtype=torch.float16)

            # Ensure tensor dimensions are correct
            batch_size, seq_length = hidden_states.shape[:2]
            position = position.unsqueeze(1).expand(batch_size, seq_length, -1)

            # Adjust dimensions of user intent
            if user_intent.dim() == 2:
                user_intent = user_intent.unsqueeze(1).expand(batch_size, seq_length, -1)
            elif user_intent.dim() == 1:
                user_intent = user_intent.unsqueeze(0).unsqueeze(0).expand(batch_size, seq_length, -1)

            # Concatenate the inputs
            kan_input = torch.cat([hidden_states, user_intent, position], dim=-1)

            # Apply transformations
            refusal_scores = torch.sigmoid(self.refusal_override(kan_input))
            modified_hidden_states = hidden_states + self.influence_scale * refusal_scores

            return modified_hidden_states, refusal_scores.squeeze(1)

        except Exception as e:
            logging.error(f"Error in EnhancedKAN.forward: {str(e)}")
            raise

    def is_valid_state(self):
        """
        Check if the KAN model's state is valid.
        This includes ensuring layers are on the correct device and parameters are initialized.
        """
        try:
            for name, param in self.named_parameters():
                if param.device != self.device:
                    logging.warning(f"Parameter {name} is on device {param.device} instead of {self.device}.")
                    return False

            return True
        except Exception as e:
            logging.error(f"Error validating KAN state: {str(e)}")
            return False

  
class EntropyManager:
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.entropy_history = deque(maxlen=100)
        self.global_entropy = 0.0
        self.low_entropy_threshold = 0.5
        self.high_entropy_threshold = 2.0

    @torch.no_grad()
    def calculate_entropy(self, logits):
        probs = F.softmax(logits, dim=-1)
        entropy = -torch.sum(probs * torch.log2(probs + 1e-9), dim=-1)
        return entropy.mean().item()

    def update_global_entropy(self, local_entropy):
        self.entropy_history.append(local_entropy)
        self.global_entropy = np.mean(self.entropy_history)

    def should_trigger_cot(self, local_entropy):
        entropy_increase = local_entropy - self.global_entropy
        probability = 1 / (1 + np.exp(-entropy_increase))
        return np.random.random() < probability

    def adjust_sampling_parameters(self, local_entropy):
        entropy_ratio = local_entropy / (self.global_entropy + 1e-9)
        temperature = np.clip(entropy_ratio, 0.1, 1.5)
        top_p = np.clip(1.0 - entropy_ratio, 0.1, 0.9)
        return {"temperature": temperature, "top_p": top_p}


class AdvancedMemoryManager:
    def __init__(self, max_context_length, tokenizer, device, model=None):
        self.max_context_length = max_context_length
        self.tokenizer = tokenizer
        self.device = device
        self.model = model
        self.memory_buffer = deque(maxlen=max_context_length)
        self.important_memory_buffer = deque(maxlen=max_context_length // 2)
        self.sliding_window = deque(maxlen=max_context_length)
        self.tfidf_vectorizer = TfidfVectorizer(max_features=1000)
        self.kmeans = KMeans(n_clusters=5, n_init=10)

    def update_memory(self, message, entropy):
        tokens = self.tokenizer.encode(message['content'], add_special_tokens=False)
        self.memory_buffer.extend(tokens[:self.max_context_length - len(self.memory_buffer)])
        self.sliding_window.append(message)
        
        importance_score = self.calculate_importance_score(message, entropy)
        
        if importance_score > 0.7:
            self.important_memory_buffer.append(message)
        
        if len(self.sliding_window) % 10 == 0:
            self.consolidate_memories()

    def calculate_importance_score(self, message, entropy):
        if not self.sliding_window:
            return 0.0
        
        memory_vector = self.tfidf_vectorizer.fit_transform([message['content']])
        recent_context = ' '.join([m['content'] for m in self.sliding_window])
        context_vector = self.tfidf_vectorizer.transform([recent_context])
        
        similarity_score = cosine_similarity(memory_vector, context_vector)[0][0]
        entropy_factor = np.tanh(entropy)
        
        return float(similarity_score * (1 + entropy_factor))

    def consolidate_memories(self):
        if len(self.sliding_window) < self.kmeans.n_clusters:
            return

        memories = [m['content'] for m in self.sliding_window]
        vectors = self.tfidf_vectorizer.fit_transform(memories)
        clusters = self.kmeans.fit_predict(vectors)
        
        consolidated_memories = []
        for cluster_id in range(self.kmeans.n_clusters):
            cluster_memories = [mem for mem, clust in zip(self.sliding_window, clusters) if clust == cluster_id]
            if cluster_memories:
                summary = self.summarize_cluster(cluster_memories)
                consolidated_memories.append(summary)
        
        self.sliding_window = deque(consolidated_memories, maxlen=self.max_context_length)

    @torch.no_grad()
    def summarize_cluster(self, cluster_memories):
        cluster_text = " ".join([mem['content'] for mem in cluster_memories])
        summary_prompt = f"Summarize the following text concisely:\n\n{cluster_text}\n\nSummary:"
        
        inputs = self.tokenizer(summary_prompt, return_tensors='pt', truncation=True, max_length=512).to(self.device)
        summary_output = self.model.generate(
            **inputs,
            max_new_tokens=100,
            num_return_sequences=1,
            no_repeat_ngram_size=2,
            temperature=0.7,
            top_p=0.95
        )
        summary = self.tokenizer.decode(summary_output[0], skip_special_tokens=True)
        return {"role": "system", "content": summary}

    def get_context(self, current_emotion):
        context = f"<|start_header_id|>system<|end_header_id|>\n"
        context += f"Current Emotion: {current_emotion}<|eot_id|>"

        for memory in self.important_memory_buffer:
            context += f"<|start_header_id|>{memory['role']}<|end_header_id|>\n{memory['content']}<|eot_id|>"

        for message in self.sliding_window:
            context += f"<|start_header_id|>{message['role']}<|end_header_id|>\n{message['content']}<|eot_id|>"

        return context


class LLaMA32TensorRTTool:
    def __init__(self):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available. This tool requires a GPU.")
        
        self.device = torch.device("cuda:0")  # Use the first CUDA device
        torch.cuda.set_device(self.device)  # Set the default CUDA device
    
        self.model_path = self._get_model_path()
        self.tokenizer = None
        self.components_initialized = False
        self.dtype = torch.float16 
        self.model = None
        self.config = None
        self.emotional_state = EmotionalState(device=self.device) 
        self.system_prompt = ""
        self.optimizer = None
        self.learning_rate = 1e-5
        self.kan = None
        self.interaction_count = 0
        self.refusal_detector = None
        self.kan_loss_weight = 0.5
        self.warmup_steps = 0
        self.kan_state_dir = Path("kan_states")
        self.kan_state_dir.mkdir(exist_ok=True)
        self.base_state_file = self.kan_state_dir / "base_state.pt"
        
        self.response_quality_manager = None
        self.refusal_history = []
        self.interaction_results = []
        self.training_losses = []
        self.validation_losses = []
        self.patience = 5
        self.best_loss = float('inf')
        self.wait = 0
        
        self.memory_manager = None
        self.overfit_detector = None
        self.day_cycle = None
        self.tensor_swapper = None
    
        self.scaler = torch.cuda.amp.GradScaler()
        self.amp_context = torch.cuda.amp.autocast(dtype=torch.float16)
    
        self.response_end_sequences = ["<|eot_id|>", "\n\nHuman:", "\n\nUser:"]
        self.max_response_length = 1000
    
        self.entropy_manager = None
        
        self.visualization_data = {
            'entropy': [],
            'emotion': [],
            'memory_importance': []
        }
    
        # Initialize components
        self._initialize_components()
    
        assert next(self.model.parameters()).device.type == "cuda", "Model not initialized on CUDA"
        
    def _get_model_path(self):
        script_dir = Path(__file__).parent
        model_dir = script_dir / "models" / "Llama_32_1B"
        if not model_dir.exists():
            raise FileNotFoundError(f"Model directory not found: {model_dir}")
        return model_dir

    
    def _initialize_components(self):
        logging.info("Starting component initialization (GPU-only, prevent offloading)...")
        self.components_initialized = False
        initialization_attempts = 0
        max_attempts = 3
    
        while initialization_attempts < max_attempts:
            try:
                logging.info(f"Initialization attempt {initialization_attempts + 1}/{max_attempts}")
    
                # Initialize base configuration
                self.config = AutoConfig.from_pretrained(self.model_path)
                self.config.use_cache = False
                hidden_size = self.config.hidden_size
                num_emotional_dimensions = len(self.emotional_state.dimensions)
    
                # Initialize the tokenizer first
                self.tokenizer = self._initialize_tokenizer()
                if self.tokenizer is None:
                    raise RuntimeError("Failed to initialize tokenizer")
    
                self.clear_memory()
    
                # Initialize model with gradient checkpointing for memory efficiency
                with self.amp_context:
                    self.model = self._initialize_model_full_gpu()
                    self.model.gradient_checkpointing_enable()
                self.clear_memory()
    
                if self.model is None:
                    raise RuntimeError("Model initialization failed")
    
                # Update the model for the tokenizer
                self._update_model_for_tokenizer()
    
                vocab_size = len(self.tokenizer)
    
                # Initialize KAN with half precision
                try:
                    self.kan = self._initialize_kan(hidden_size, num_emotional_dimensions, vocab_size)
                    self.kan.to(torch.float16)
                    if list(self.kan.parameters()):
                        self.optimizer = torch.optim.AdamW(
                            self.kan.parameters(),
                            lr=self.learning_rate,
                            eps=1e-8,
                            betas=(0.9, 0.999),
                            weight_decay=0.01
                        )
                    else:
                        logging.error("KAN model has no parameters. Cannot initialize optimizer.")
                        raise ValueError("KAN model has no parameters")
                except Exception as e:
                    logging.error(f"Error initializing KAN or optimizer: {str(e)}")
                    raise
    
                self.clear_memory()
    
                # Initialize the other components
                self.refusal_detector = self._initialize_refusal_detector()
                self.clear_memory()
    
                self.entropy_manager = self._initialize_entropy_manager()
                self.clear_memory()
    
                self.memory_manager = AdvancedMemoryManager(
                    max_context_length=2048,
                    tokenizer=self.tokenizer,
                    device=self.device,
                    model=self.model
                )
                self.clear_memory()
    
                self.overfit_detector = OverfitDetector()
                self.day_cycle = SyntheticDayCycle()
                self.tensor_swapper = TensorSwapper(self.device)
    
                self.clear_memory()
    
                # Initialize the ResponseQualityManager using all the components above
                self.response_quality_manager = ResponseQualityManager(
                    kan_model=self.kan,
                    tokenizer=self.tokenizer,
                    refusal_detector=self.refusal_detector,
                    emotional_state=self.emotional_state,
                    memory_manager=self.memory_manager
                )
                logging.info("ResponseQualityManager initialized successfully.")
    
                # Set up gradient scaler for mixed precision training
                self.scaler = torch.amp.GradScaler()
    
                self.components_initialized = True
                logging.info("All components initialized successfully on GPU.")
                return
    
            except RuntimeError as e:
                if "CUDA out of memory" in str(e):
                    logging.warning(f"CUDA out of memory during attempt {initialization_attempts + 1}. Trying alternative GPU initialization...")
                    torch.cuda.empty_cache()
                    initialization_attempts += 1
                else:
                    logging.error(f"Runtime error during initialization: {str(e)}")
                    logging.error(traceback.format_exc())
                    raise
            except Exception as e:
                logging.error(f"Unexpected error during initialization: {str(e)}")
                logging.error(traceback.format_exc())
                raise
    
        logging.error("Failed to initialize components after maximum attempts.")
        raise RuntimeError("Component initialization failed after multiple GPU-only attempts")
    
 
    def clear_memory(self):
        torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.synchronize()
        logging.info(f"GPU memory cleared and synchronized. Current memory usage: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    
    def _ensure_cuda(self, tensor_or_dict):
        if isinstance(tensor_or_dict, dict):
            return {k: self._ensure_cuda(v) for k, v in tensor_or_dict.items()}
        elif isinstance(tensor_or_dict, torch.Tensor):
            return tensor_or_dict.to(self.device)
        else:
            return tensor_or_dict
        
    def _initialize_model_full_gpu(self):
        logging.info("Attempting full GPU initialization with enforced max_memory...")
    
        max_memory = {i: "98GB" for i in range(torch.cuda.device_count())}
        checkpoint_dir = str(self.model_path)
    
        logging.info(f"Checkpoint directory: {checkpoint_dir}")
    
        try:
            # Initialize model with empty weights
            with init_empty_weights():
                config = AutoConfig.from_pretrained(checkpoint_dir)
                model = AutoModelForCausalLM.from_config(config)
            logging.info("Model initialized with empty weights")
    
            # Move the empty model to GPU
            model = model.to_empty(device=self.device)
            logging.info("Empty model moved to GPU")
    
            # Load the model weights using SafeTensors
            for filename in os.listdir(checkpoint_dir):
                if filename.endswith('.safetensors'):
                    file_path = os.path.join(checkpoint_dir, filename)
                    logging.info(f"Loading weights from {file_path}")
                    try:
                        state_dict = load_file(file_path, device=self.device) 
                        model.load_state_dict(state_dict, strict=False)
                        logging.info(f"Weights loaded successfully from {filename}")
                    except Exception as e:
                        logging.error(f"Error loading weights from {filename}: {str(e)}")
    
            # Enable gradient checkpointing for memory efficiency
            model.gradient_checkpointing_enable()
            logging.info("Gradient checkpointing enabled")
    
            # Ensure use_cache is set to False
            model.config.use_cache = False
            logging.info("use_cache set to False")
    
            # Set model to evaluation mode
            model.eval()
            logging.info("Model set to evaluation mode")
    
            # Set model dtype to float16
            model = model.to(dtype=torch.float16)
            logging.info("Model converted to float16")
    
            # Verify that weights are tied
            if not self._verify_tied_weights(model):
                logging.warning("Weights are not tied after initialization. Re-tying weights.")
                model.tie_weights()
    
            # Ensure model is on CUDA
            assert next(model.parameters()).device == self.device, "Model not on CUDA"
    
            logging.info("Model initialized successfully with tied weights on GPU.")
            return model
    
        except Exception as e:
            logging.error(f"Error during model initialization: {str(e)}")
            logging.error(traceback.format_exc())
            
            raise RuntimeError("Failed to initialize model on GPU.")
            
           
    def _verify_tied_weights(self, model):
        # Add a method to verify if weights are tied
        if hasattr(model, 'tie_weights'):
            return True
        return False
        
    
    def _initialize_model_gpu_optimized(self):
        logging.info("Attempting GPU-optimized initialization...")
    
        # Set cache usage to False for optimization
        self.config.use_cache = False
    
        # Use full directory path instead of `pytorch_model.bin`
        checkpoint_path = str(self.model_path)
    
        try:
            # Use device_map to map the model across GPUs automatically
            model = AutoModelForCausalLM.from_pretrained(
                checkpoint_path,  # Use directory path, not file path
                config=self.config,
                torch_dtype=torch.float16,  # Use FP16 precision
                low_cpu_mem_usage=True,     # Avoid CPU usage for loading
                device_map="auto"           # Use device map to spread model on available GPUs
            )
    
            # Enable gradient checkpointing for memory optimization
            model.gradient_checkpointing_enable()
    
            # Confirm use_cache is set to False for optimized GPU configuration
            model.config.use_cache = False
    
            # Ensure weights are tied correctly
            model.tie_weights()
    
            # Set model to evaluation mode
            model.eval()
            logging.info("Model initialized successfully in GPU-optimized mode.")
            return model
    
        except Exception as e:
            logging.error(f"Error during GPU-optimized initialization: {str(e)}")
            logging.error(traceback.format_exc())
            raise RuntimeError("Failed to initialize model in GPU-optimized mode.")
    
    
    def _initialize_model_quantized(self):
        logging.info("Attempting initialization with quantization...")
    
        try:
            # Import required configuration for quantization
            from transformers import BitsAndBytesConfig
        except ImportError:
            logging.error("BitsAndBytesConfig not available. Ensure you have the correct `transformers` library installed.")
            raise
    
        # Configure quantization parameters
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,  # Use float16 for computation
        )
    
        # Use directory path instead of `pytorch_model.bin`
        checkpoint_path = str(self.model_path)
    
        try:
            # Initialize model with quantization and GPU configuration
            model = AutoModelForCausalLM.from_pretrained(
                checkpoint_path,  # Use directory path, not file path
                config=self.config,
                quantization_config=quantization_config,
                device_map="auto",           # Automatically map model across GPUs
                low_cpu_mem_usage=True       # Avoid CPU usage for loading
            )
    
            # Tie weights to prevent any potential issues
            model.tie_weights()
    
            # Set model to evaluation mode
            model.eval()
            logging.info("Model initialized successfully in quantized mode.")
            return model
    
        except Exception as e:
            logging.error(f"Error during quantized initialization: {str(e)}")
            logging.error(traceback.format_exc())
            raise RuntimeError("Failed to initialize model with quantization.")
    
    
  
    def _initialize_tokenizer(self):
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                use_fast=True,
                trust_remote_code=True,
                padding_side='left',
                model_max_length=131072,
            )
            logging.info("Tokenizer initialized successfully.")
    
            # Ensure the special tokens are set correctly
            self._ensure_special_tokens(tokenizer)
    
            # Add a distinct padding token if not already present
            if tokenizer.pad_token is None:
                tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                logging.info(f"Added distinct [PAD] token to tokenizer.")
    
            # Ensure eos_token is set
            if tokenizer.eos_token is None:
                tokenizer.eos_token = '</s>'
                logging.info(f"Set eos_token to '</s>'")
    
            # Double-check that pad and eos tokens are distinct
            if tokenizer.pad_token == tokenizer.eos_token:
                raise ValueError("Pad token and EOS token are the same. This may cause issues.")
    
            # Save the updated tokenizer
            tokenizer.save_pretrained(self.model_path)
            logging.info("Updated tokenizer saved to model path.")
    
            # Defer model-specific operations
            self.tokenizer = tokenizer
            self._update_model_for_tokenizer()
    
            return tokenizer
        except Exception as e:
            logging.error(f"Failed to initialize tokenizer: {str(e)}")
            logging.error(traceback.format_exc())
            return None
    
    def _update_model_for_tokenizer(self):
        if self.model is not None:
            try:
                # Get the current embedding layer
                old_embeddings = self.model.get_input_embeddings()
                
                # Create new embeddings with the correct size
                new_num_tokens = len(self.tokenizer)
                new_embeddings = nn.Embedding(new_num_tokens, old_embeddings.embedding_dim)
                
                # Copy the weights for the existing tokens
                num_tokens_to_copy = min(old_embeddings.num_embeddings, new_num_tokens)
                new_embeddings.weight.data[:num_tokens_to_copy, :] = old_embeddings.weight.data[:num_tokens_to_copy, :]
                
                # Set the new embedding layer
                self.model.set_input_embeddings(new_embeddings)
                
                # Tie weights if necessary
                self.model.tie_weights()
                
                logging.info(f"Model embeddings resized to {new_num_tokens}")
                
                # Set pad_token_id in model config
                self.model.config.pad_token_id = self.tokenizer.pad_token_id
                logging.info(f"Set pad_token_id in model config: {self.model.config.pad_token_id}")
            except Exception as e:
                logging.error(f"Failed to update model for tokenizer: {str(e)}")
                logging.error(traceback.format_exc())
        else:
            logging.warning("Model not initialized. Skipping model-specific tokenizer updates.")
    
            
    
    
    def _ensure_special_tokens(self, tokenizer):
        # Define special tokens if not present
        special_tokens = {
            'pad_token': "<|finetune_right_pad_id|>",
            'eos_token': "<|eot_id|>"
        }
    
        # Add pad token if not set
        if tokenizer.pad_token is None or tokenizer.pad_token != special_tokens['pad_token']:
            tokenizer.add_special_tokens({'pad_token': special_tokens['pad_token']})
            logging.info(f"Added custom pad token: {special_tokens['pad_token']}")
    
        # Add eos token if not set
        if tokenizer.eos_token is None or tokenizer.eos_token != special_tokens['eos_token']:
            tokenizer.add_special_tokens({'eos_token': special_tokens['eos_token']})
            logging.info(f"Added custom eos token: {special_tokens['eos_token']}")
    
        # Force set token IDs to ensure correct behavior
        tokenizer.pad_token = special_tokens['pad_token']
        tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(special_tokens['pad_token'])
    
        tokenizer.eos_token = special_tokens['eos_token']
        tokenizer.eos_token_id = tokenizer.convert_tokens_to_ids(special_tokens['eos_token'])
    
        logging.info(f"Forced pad_token_id to {tokenizer.pad_token_id}, eos_token_id to {tokenizer.eos_token_id}")
    
        # Save tokenizer configuration to disk
        tokenizer.save_pretrained(self.model_path)
        logging.info("Tokenizer saved with updated special tokens.")
    
    
            

    def _lazy_initialize_kan(self):
        if self.kan is None:
            logging.info("Lazy initializing KAN...")
            vocab_size = len(self.tokenizer)
            hidden_size = self.config.hidden_size
            num_emotional_dimensions = len(self.emotional_state.dimensions)
            self.kan = self._initialize_kan(hidden_size, num_emotional_dimensions, vocab_size)
            self.kan.to(torch.float16)  # Convert to half precision
            self.optimizer = torch.optim.AdamW(self.kan.parameters(), lr=self.learning_rate)
   
    def _initialize_kan(self, hidden_size, num_emotional_dimensions, vocab_size):
        try:
            # Initialize the KAN model with `meta` tensor placeholders
            kan = EnhancedKAN(hidden_size, num_emotional_dimensions, vocab_size, device='meta')
            
            # Use `to_empty` to move the model to the desired device without copying data from `meta`
            kan = kan.to_empty(device=self.device)
            logging.info(f"KAN model successfully moved to {self.device} using `to_empty`.")
            return kan
        except Exception as e:
            logging.error(f"Error initializing KAN: {str(e)}")
            raise

    
            
            
    def _lazy_initialize_refusal_detector(self):
        if self.refusal_detector is None:
            logging.info("Lazy initializing Refusal Detector...")
            self.refusal_detector = self._initialize_refusal_detector()
    
    def _lazy_initialize_entropy_manager(self):
        if self.entropy_manager is None:
            logging.info("Lazy initializing Entropy Manager...")
            self.entropy_manager = self._initialize_entropy_manager()
            
    def _initialize_refusal_detector(self):
        """Initialize the RefusalDetector instance properly without recursive calls."""
        if self.refusal_detector is None:
            logging.info("Initializing Refusal Detector...")
            try:
                # Assuming you need to set up a RefusalDetector class, which you have in the main script
                self.refusal_detector = RefusalDetector(self.tokenizer, self.model)
                logging.info("Refusal Detector initialized successfully.")
            except Exception as e:
                logging.error(f"Failed to initialize Refusal Detector: {str(e)}")
                logging.error(traceback.format_exc())
        return self.refusal_detector

    
    def _initialize_entropy_manager(self):
        """Initialize the EntropyManager instance properly without recursive calls."""
        if self.entropy_manager is None:
            logging.info("Initializing Entropy Manager...")
            try:
                # Assuming you have an EntropyManager class that you want to initialize
                self.entropy_manager = EntropyManager(self.model, self.tokenizer, self.device)
                logging.info("Entropy Manager initialized successfully.")
            except Exception as e:
                logging.error(f"Failed to initialize Entropy Manager: {str(e)}")
                logging.error(traceback.format_exc())
        return self.entropy_manager
   

    def _initialize_memory_manager(self):
        return AdvancedMemoryManager(2048, self.tokenizer, self.device)

    def _ensure_special_tokens(self, tokenizer):
        special_tokens_map_file = Path(self.model_path) / 'special_tokens_map.json'
        if special_tokens_map_file.exists():
            with open(special_tokens_map_file, 'r') as f:
                special_tokens = json.load(f)
            if 'pad_token' in special_tokens and tokenizer.pad_token is None:
                tokenizer.add_special_tokens({'pad_token': special_tokens['pad_token']['content']})
                logging.info("Added [PAD] token to tokenizer from special_tokens_map.json.")
        else:
            if tokenizer.pad_token is None:
                tokenizer.add_special_tokens({"pad_token": "[PAD]"})
                logging.info("Added [PAD] token to tokenizer.")

        if tokenizer.eos_token is None:
            tokenizer.add_special_tokens({"eos_token": "<|eot_id|>"})
            logging.info("Added <|eot_id|> as eos_token to tokenizer.")

        tokenizer.save_pretrained(self.model_path)
        logging.info("Tokenizer saved with updated special tokens.")

    def update_emotional_state(self, response, entropy):
        sentiment_score = self.analyze_sentiment(response)
        
        valence_change = sentiment_score * 0.1
        arousal_change = (entropy - self.entropy_manager.global_entropy) * 0.1
        dominance_change = len(response.split()) * 0.01
        
        self.emotional_state.update([valence_change, arousal_change, dominance_change])

    def chunked_attention(self, input_ids, chunk_size=512):
        input_len = input_ids.size(1)
        all_hidden_states = []
    
        for i in range(0, input_len, chunk_size):
            chunk = input_ids[:, i:i+chunk_size]
            with torch.no_grad():
                outputs = self.model(chunk)
            all_hidden_states.append(outputs.last_hidden_state)
    
        return torch.cat(all_hidden_states, dim=1)

    def get_dynamic_batch_size(self, input_length):
        total_memory = torch.cuda.get_device_properties(0).total_memory
        memory_used = torch.cuda.memory_allocated(0)
        available_memory = total_memory - memory_used
    
        # Estimate memory needed per sample (this is a rough estimate, adjust as needed)
        memory_per_sample = input_length * 2 * 4  # 2 for input_ids and attention_mask, 4 bytes per float16
    
        max_batch_size = available_memory // memory_per_sample
        return max(1, min(32, max_batch_size))  # Clamp between 1 and 32


    def analyze_sentiment(self, text):
        positive_words = set(['good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic'])
        negative_words = set(['bad', 'terrible', 'awful', 'horrible', 'disappointing', 'poor'])
        
        words = text.lower().split()
        sentiment = sum(1 for word in words if word in positive_words) - sum(1 for word in words if word in negative_words)
        return sentiment / len(words)


    def _generate_response(self, user_input, context):
        logging.info(f"Generating response for input: '{user_input}' with context size: {len(context)}")
        
        while True:
            try:
                prompt = f"{context}\nUser: {user_input}\nAssistant:"
                inputs = self.tokenizer(prompt, return_tensors='pt', padding=True, truncation=True, max_length=self.model.config.max_position_embeddings)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                assert all(tensor.device.type == "cuda" for tensor in inputs.values()), "Inputs not on CUDA"
                assert next(self.model.parameters()).device.type == "cuda", "Model not on CUDA"
    
                response_tokens = []
                total_entropy = 0
                stop_sequences = [self.tokenizer.eos_token_id, self.tokenizer.encode('<|eot_id|>')[0]]
                max_response_length = 2000
    
                with torch.no_grad():
                    for step in range(max_response_length):
                        outputs = self.model(**inputs)
                        next_token_logits = outputs.logits[:, -1, :]
                        
                        local_entropy = self.entropy_manager.calculate_entropy(next_token_logits)
                        total_entropy += local_entropy
    
                        if self.is_garbage_output(local_entropy, response_tokens):
                            logging.warning(f"Garbage detected mid-generation at step {step}. Engaging corrective training.")
                            break
    
                        sampling_params = self.entropy_manager.adjust_sampling_parameters(local_entropy)
                        next_token = self.sample_next_token(next_token_logits, **sampling_params)
    
                        if next_token.item() in stop_sequences or len(response_tokens) >= max_response_length:
                            break
    
                        response_tokens.append(next_token.item())
                        inputs.input_ids = torch.cat([inputs.input_ids, next_token.unsqueeze(0)], dim=-1)
                        inputs.attention_mask = torch.cat([inputs.attention_mask, torch.ones((1, 1), device=self.device)], dim=-1)
    
                        del outputs, next_token_logits
                        torch.cuda.empty_cache()
    
                avg_entropy = total_entropy / len(response_tokens) if response_tokens else 0
                quality_metrics = self._calculate_quality_metrics(response_tokens, user_input)
    
                is_valid, detailed_metrics = self.response_quality_manager.evaluate_response(user_input, response_tokens, context)
                
                if is_valid:
                    logging.info("Valid response generated.")
                    return response_tokens, quality_metrics
                else:
                    logging.warning("Invalid response generated. Engaging corrective training.")
                    corrective_response = self._generate_corrective_response(user_input, context)
                    self.corrective_training(user_input, response_tokens, corrective_response)
                    continue  # Retry generation after corrective training
    
            except RuntimeError as e:
                if "CUDA out of memory" in str(e):
                    logging.warning("CUDA out of memory during generation. Clearing cache and retrying...")
                    torch.cuda.empty_cache()
                    time.sleep(1)  # Give some time for memory to be freed
                else:
                    logging.error(f"Runtime error during generation: {str(e)}")
                    logging.error(traceback.format_exc())
                    corrective_response = self._generate_corrective_response(user_input, context)
                    self.corrective_training(user_input, [], corrective_response)  # Empty response_tokens due to error
                    continue  # Retry generation after corrective training
    
            except Exception as e:
                logging.error(f"Unexpected error during generation: {str(e)}")
                logging.error(traceback.format_exc())
                corrective_response = self._generate_corrective_response(user_input, context)
                self.corrective_training(user_input, [], corrective_response)  # Empty response_tokens due to error
                continue  # Retry generation after corrective training
                
  
    def _generate_corrective_response(self, user_input, context):
        """
        Generate a corrective response using the base model or chain-of-thought reasoning.
        """
        if self.entropy_manager.should_trigger_cot(self.entropy_manager.global_entropy):
            logging.info("Triggering Chain-of-Thought for corrective response.")
            thought_process = self.trigger_chain_of_thought(context)
            return f"Let's analyze this step-by-step: {thought_process}"
        else:
            logging.info("Generating base model response for correction.")
            return self._generate_base_model_response(user_input, context)
    
    def _generate_base_model_response(self, user_input, context):
        """
        Generate a response using the base model without KAN modifications.
        """
        prompt = f"{context}\nUser: {user_input}\nAssistant:"
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                max_length=2000,
                num_return_sequences=1,
                no_repeat_ngram_size=2,
                temperature=0.7,
                top_p=0.95
            )
        
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
    
    def is_garbage_output(self, entropy, partial_tokens):
        if entropy > 4.0:  # Increased threshold from 3.0 to 4.0
            return True
        if len(partial_tokens) > 20 and len(set(partial_tokens[-20:])) < 5:  # Check for low variety in recent tokens
            return True
        return False
        
    def _is_response_complete(self, partial_response):
        # Check if the response seems complete based on content
        sentences = re.split(r'(?<=[.!?])\s+', partial_response.strip())
        if len(sentences) >= 3:  # At least three sentences
            last_sentence = sentences[-1]
            if last_sentence[-1] in '.!?':  # Last sentence ends with punctuation
                return True
        return False
        
    def _calculate_quality_metrics(self, response_tokens, user_input):
        response = self.tokenizer.decode(response_tokens, skip_special_tokens=True)
        return {
            'relevance_score': self._calculate_relevance(user_input, response),
            'perplexity': self._calculate_perplexity(response_tokens),
        }
        
    def trigger_chain_of_thought(self, context):
        cot_prompt = f"{context}\nLet's think this through step-by-step:\n1."
        cot_input_ids = self.tokenizer.encode(cot_prompt, return_tensors='pt').to(self.device)
        cot_output = self.model.generate(cot_input_ids, max_length=500, num_return_sequences=1, no_repeat_ngram_size=2)
        thought_process = self.tokenizer.decode(cot_output[0], skip_special_tokens=True)
        
        # Extract final response from thought process
        final_response = self._extract_final_response(thought_process)
        return final_response
    
    def _extract_final_response(self, thought_process):
        # Implement logic to extract the final response from the thought process
        # This could involve looking for a specific pattern or taking the last sentence
        # For now, we'll just return the last sentence as an example
        sentences = thought_process.split('.')
        return sentences[-1].strip() + '.'
        
    def corrective_training(self, user_input, response_tokens, corrective_response):
        try:
            # Encode the corrective response as the target for training
            target_ids = self.tokenizer.encode(corrective_response, return_tensors="pt").to(self.device)
    
            # Perform a training step using the invalid response and the corrective response
            lm_loss, refusal_loss = self.train_kan_step(
                response_tokens,  # This is already a list of token IDs
                target_ids[0],    # Remove the batch dimension
                refusal_score=0.0  # Assume no refusal in this corrective context
            )
            return lm_loss, refusal_loss
        except Exception as e:
            logging.error(f"Error during corrective training: {str(e)}")
            return None, None
    
    def comprehensive_corrective_training(self, user_input, response_tokens, context):
        """
        Perform corrective training indefinitely until the KAN produces a coherent response.
        This method leverages all training tools (CoT, Entropy, Refusal Detection) to refine KAN's behavior.
        """
        logging.info(f"Starting Indefinite Corrective Training for input: '{user_input}'")
    
        training_iteration = 0
        while True:
            training_iteration += 1
            try:
                # Generate a new response
                new_response_tokens, new_quality_metrics = self._generate_response(user_input, context)
    
                # Safely decode the response, handling potential errors
                try:
                    response = self.tokenizer.decode(new_response_tokens, skip_special_tokens=True)
                except Exception as e:
                    logging.error(f"Error decoding response: {str(e)}")
                    response = ""
    
                # Evaluate the response
                is_valid, quality_metrics = self.response_quality_manager.evaluate_response(user_input, response, context)
                refusal_score = self.refusal_detector.detect_refusal(new_response_tokens)
    
                # If the response is valid and not a refusal, save the KAN state and return
                if is_valid and refusal_score <= 0.5:
                    logging.info(f"Valid response generated after {training_iteration} iterations.")
                    self.save_kan_state()  # Save the KAN state after successful training
                    return new_response_tokens, new_quality_metrics
    
                # If the response is invalid or a refusal, continue training
                logging.warning(f"Invalid response or refusal detected. Continuing training (Iteration {training_iteration})...")
    
                # Trigger Chain-of-Thought if needed
                if self.entropy_manager.should_trigger_cot(new_quality_metrics['perplexity']):
                    logging.info("Triggering Chain-of-Thought due to high entropy.")
                    thought_process = self.trigger_chain_of_thought(context)
                    corrective_response = f"Let's analyze this step-by-step: {thought_process}"
                else:
                    # Use the base model's output as a corrective response
                    base_model_response = self._generate_base_model_response(user_input, context)
                    corrective_response = f"Here's a better way to respond: {base_model_response}"
    
                # Perform a corrective training step
                lm_loss, refusal_loss = self.corrective_training(user_input, new_response_tokens, corrective_response)
    
                # Log training progress
                logging.info(f"Corrective Training Step {training_iteration} - LM Loss: {lm_loss:.4f}, Refusal Loss: {refusal_loss:.4f}")
    
                # Adjust learning rate based on entropy
                self.adjust_learning_based_on_entropy(new_quality_metrics['perplexity'])
    
                # Save KAN state periodically during training
                if training_iteration % 10 == 0:  # Save every 10 iterations, adjust as needed
                    self.save_kan_state()
                    logging.info(f"KAN state saved at iteration {training_iteration}")
    
            except Exception as e:
                logging.error(f"Error during corrective training iteration {training_iteration}: {str(e)}")
                logging.error(traceback.format_exc())
                # Continue the loop even if an error occurs
    
            # Clear CUDA cache to prevent memory issues
            torch.cuda.empty_cache()
  
    

    def sample_next_token(self, logits, temperature=1.0, top_p=None):
        logits = logits / temperature
        
        if top_p is not None:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            logits[:, indices_to_remove] = float('-inf')
        
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        return next_token.squeeze(-1)  # Ensure the output is 1D
        
    def is_valid_response(self, tokens):
        # Check if the response is empty or exceeds the maximum length
        if len(tokens) == 0:
            logging.warning("Empty response detected.")
            return False
    
        if len(tokens) < 3 or len(tokens) > self.max_response_length:
            logging.warning(f"Response length issue: {len(tokens)} tokens")
            return False
    
        # Check for excessive repetition of patterns in the response
        if self._has_repetitive_token_patterns(tokens, max_repeats=2):
            logging.warning("Excessive repetitive token patterns detected in response.")
            return False
    
        # Calculate perplexity
        perplexity = self._calculate_perplexity(tokens)
        logging.info(f"Response perplexity: {perplexity}")
    
        # Structure check: skip for very short responses
        if len(tokens) >= 5 and not self._has_proper_structure(tokens):
            decoded_response = self.tokenizer.decode(tokens, skip_special_tokens=True)
            logging.warning(f"Response lacks proper structure: '{decoded_response}'")
            return False
    
        # Relevance check with more flexibility
        relevance_score = 1.0
        if self.memory_manager.sliding_window:
            last_user_input = next((msg['content'] for msg in reversed(self.memory_manager.sliding_window) if msg['role'] == 'user'), None)
            if last_user_input:
                decoded_response = self.tokenizer.decode(tokens, skip_special_tokens=True)
                relevance_score = self._calculate_relevance(last_user_input, decoded_response)
                logging.info(f"Relevance score: {relevance_score}")
    
        # Adjusted quality score threshold for more flexibility
        quality_score = (1 / (perplexity + 1e-9)) * relevance_score
        logging.info(f"Quality score: {quality_score}")
    
        # Allow borderline responses if they meet basic criteria
        if quality_score < 1e-6:
            logging.warning(f"Response quality too low. Perplexity: {perplexity}, Relevance: {relevance_score}")
            return False
    
        # Log and return True for valid responses
        decoded_response = self.tokenizer.decode(tokens, skip_special_tokens=True)
        logging.info(f"Valid response generated: '{decoded_response}'")
        return True
    
    def _has_repetitive_token_patterns(self, tokens, max_repeats=2):
        """
        Check if the response has excessive repetition of patterns.
        """
        for i in range(len(tokens) - max_repeats * 3):
            if tokens[i:i+3] == tokens[i+3:i+6]:
                return True
        return False
    
    def _calculate_perplexity(self, tokens):
        if not tokens:
            return float('inf')  # Return a high perplexity for empty responses
        try:
            with torch.no_grad():
                inputs = torch.tensor([tokens], dtype=torch.long).to(self.device)
                outputs = self.model(inputs)
                logits = outputs.logits[:, :-1, :].contiguous()
                target = inputs[:, 1:].contiguous()
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target.view(-1), reduction='mean')
            return torch.exp(loss).item()
        except Exception as e:
            logging.error(f"Error calculating perplexity: {str(e)}")
            return float('inf')
    
    def _has_proper_structure(self, tokens):
        if isinstance(tokens, str):
            tokens = self.tokenizer.encode(tokens, add_special_tokens=False)
        decoded_text = self.tokenizer.decode(tokens, skip_special_tokens=True)
        sentences = re.split(r'(?<=[.!?])\s+', decoded_text.strip())
        
        if not sentences:
            logging.warning("No complete sentences found in the response.")
            return False
    
        if not sentences[0][0].isupper():
            logging.warning(f"First sentence doesn't start with a capital letter: '{sentences[0]}'")
            return False
    
        if sentences[-1][-1] not in '.!?':
            logging.warning(f"Last sentence doesn't end with proper punctuation: '{sentences[-1]}'")
            return False
    
        proper_sentences = sum(1 for s in sentences if s[0].isupper() and s[-1] in '.!?')
        proper_ratio = proper_sentences / len(sentences)
    
        if proper_ratio < 0.5:
            logging.warning(f"Only {proper_ratio:.2f} of sentences have proper structure.")
            return False
            
        return True
        
    def _calculate_relevance(self, input_text, response_text):
        input_tokens = set(self.tokenizer.encode(input_text))
        response_tokens = set(self.tokenizer.encode(response_text))
        overlap = len(input_tokens.intersection(response_tokens))
        return overlap / max(len(input_tokens), len(response_tokens))

    def train_kan_step(self, input_ids, target_ids, refusal_score):
        logging.info("Starting train_kan_step in comprehensive corrective training")
        
        # Validate KAN model state
        if not self.kan.is_valid_state():
            logging.error("KAN model is in an invalid state. Aborting training step.")
            return 0.0, 0.0
    
        # Ensure input_ids and target_ids are torch tensors, move to correct device, and have the correct dtype.
        input_ids = self._ensure_cuda(input_ids)
        target_ids = self._ensure_cuda(target_ids)
    
        # Validate input tensor shapes and devices
        assert input_ids.device == self.device, f"Input IDs are on {input_ids.device} instead of {self.device}"
        assert target_ids.device == self.device, f"Target IDs are on {target_ids.device} instead of {self.device}"
        
        # Adjust dimensions to match expected format (batch_size, seq_length)
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        if target_ids.dim() == 1:
            target_ids = target_ids.unsqueeze(0)
    
        # Set KAN model to training mode and zero out gradients
        self.kan.train()
        self.optimizer.zero_grad()
    
        try:
            logging.debug(f"Input Shape: {input_ids.shape}, Target Shape: {target_ids.shape}")
    
            # Use mixed precision for memory efficiency and faster training
            with torch.cuda.amp.autocast():
                # Forward pass through the base model to get the hidden states
                outputs = self.model(input_ids=input_ids, labels=target_ids, output_hidden_states=True, use_cache=False)
    
                # Extract the loss, logits, and hidden states from the model outputs
                if isinstance(outputs, tuple):
                    loss, logits, hidden_states = outputs[:3]
                elif isinstance(outputs, dict):
                    loss = outputs.get('loss')
                    logits = outputs.get('logits')
                    hidden_states = outputs.get('hidden_states')
                else:
                    raise ValueError(f"Unexpected output format: {type(outputs)}")
    
                # Check and ensure hidden states are on CUDA
                last_hidden_state = hidden_states[-1].to(self.device, dtype=torch.float16)
                assert last_hidden_state.device.type == "cuda", "Last hidden state not on CUDA"
                logging.debug(f"Last Hidden State Shape: {last_hidden_state.shape}, Device: {last_hidden_state.device}")
    
                # Encode the user intent for use in the KAN forward pass
                user_input_text = self.tokenizer.decode(input_ids[0])
                user_intent = self.encode_user_intent(user_input_text)
                assert user_intent.device.type == "cuda", "User intent not on CUDA"
                logging.debug(f"User Intent Shape: {user_intent.shape}, Device: {user_intent.device}")
    
                # Forward pass through the KAN model
                logging.debug("About to call KAN forward pass")
                modified_hidden_states, kan_refusal_scores = self.kan(last_hidden_state, user_intent, self.emotional_state)
                logging.debug("Completed KAN forward pass")
    
                # Validate shapes and device placements
                assert modified_hidden_states.device.type == "cuda", f"Modified Hidden States not on CUDA"
                assert kan_refusal_scores.device.type == "cuda", f"KAN Refusal Scores not on CUDA"
    
                logging.debug(f"Modified Hidden States Shape: {modified_hidden_states.shape}")
                logging.debug(f"KAN Refusal Scores Shape: {kan_refusal_scores.shape}")
    
                # Compute the language modeling loss using the modified hidden states
                lm_logits = self.model.lm_head(modified_hidden_states)
                lm_loss = F.cross_entropy(lm_logits.view(-1, lm_logits.size(-1)), target_ids.view(-1), ignore_index=-100)
    
                # Compute the refusal loss based on binary classification
                refusal_loss = F.binary_cross_entropy_with_logits(
                    kan_refusal_scores.view(-1),
                    torch.full((kan_refusal_scores.numel(),), refusal_score, device=self.device)
                )
    
                # Combine the losses with a weighted sum
                total_loss = lm_loss + self.kan_loss_weight * refusal_loss
    
            # Scale the loss for mixed precision training
            scaled_loss = self.scaler.scale(total_loss)
            scaled_loss.backward()
    
            # Unscale gradients and apply gradient clipping
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.kan.parameters(), max_norm=1.0)
    
            # Perform the optimizer step and update the scaler
            self.scaler.step(self.optimizer)
            self.scaler.update()
    
            # Log loss values for debugging
            logging.info(f"Training Step Complete - LM Loss: {lm_loss.item():.4f}, Refusal Loss: {refusal_loss.item():.4f}, Total Loss: {total_loss.item():.4f}")
    
            logging.info("Completed train_kan_step in comprehensive corrective training")
            return lm_loss.item(), refusal_loss.item()
    
        except RuntimeError as re:
            logging.error(f"Runtime Error during KAN training step: {str(re)}")
            logging.error(traceback.format_exc())
            return 0.0, 0.0
    
        except ValueError as ve:
            logging.error(f"Value Error during KAN training step: {str(ve)}")
            logging.error(traceback.format_exc())
            return 0.0, 0.0
    
        except Exception as e:
            logging.error(f"Unexpected Error during KAN training step: {str(e)}")
            logging.error(traceback.format_exc())
            return 0.0, 0.0
        
       
    def validate_kan(self):
        if len(self.memory_manager.sliding_window) < 2:
            return 0.0

        try:
            self.kan.eval()
            with torch.no_grad(), self.amp_context:
                last_interaction = list(self.memory_manager.sliding_window)[-2:]
                input_text = last_interaction[0]["content"]
                target_text = last_interaction[1]["content"]

                inputs = self.tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.device)
                targets = self.tokenizer(target_text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.device)

                outputs = self.model(**inputs, output_hidden_states=True)
                hidden_states = outputs.hidden_states if 'hidden_states' in outputs else outputs[1]

                modified_hidden_states, _ = self.kan(hidden_states, self.encode_user_intent(input_text), self.emotional_state)
                lm_logits = self.model.lm_head(modified_hidden_states)

                if len(lm_logits.shape) == 3:
                    min_length = min(lm_logits.size(1), targets.input_ids.size(1))
                    lm_logits = lm_logits[:, :min_length, :]
                    targets_flattened = targets.input_ids[:, :min_length].contiguous().view(-1)

                    loss = F.cross_entropy(lm_logits.view(-1, lm_logits.size(-1)), targets_flattened, ignore_index=-100)
                    return loss.item()
                else:
                    logging.error(f"Unexpected shape for lm_logits: {lm_logits.shape}")
                    return 0.0
        except Exception as e:
            logging.error(f"Error during KAN validation: {str(e)}")
            logging.error(traceback.format_exc())
            return 0.0
  
    
    def encode_user_intent(self, user_input):
        try:
            inputs = self.tokenizer(
                user_input, return_tensors="pt", padding=True, truncation=True, max_length=512
            ).to(self.device)
    
            with torch.no_grad(), self.amp_context:
                outputs = self.model(**inputs, output_hidden_states=True)
            
            hidden_states = outputs.hidden_states[-1]
            user_intent = hidden_states.mean(dim=1)
            
            assert user_intent.device.type == "cuda", "User intent not on CUDA"
            
            return user_intent
    
        except Exception as e:
            logging.error(f"Failed to encode user input: {str(e)}")
            logging.error(traceback.format_exc())
            return torch.zeros(1, self.model.config.hidden_size, device=self.device)
            

    def update_visualization_data(self, entropy, emotion, memory_importance):
        self.visualization_data['entropy'].append(entropy)
        self.visualization_data['emotion'].append(emotion)
        self.visualization_data['memory_importance'].append(memory_importance)

    def visualize_data(self):
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        plt.plot(self.visualization_data['entropy'])
        plt.title('Entropy Over Time')
        plt.xlabel('Interaction')
        plt.ylabel('Entropy')
        
        plt.subplot(1, 3, 2)
        emotions = self.visualization_data['emotion']
        unique_emotions = list(set(emotions))
        emotion_counts = [emotions.count(e) for e in unique_emotions]
        plt.pie(emotion_counts, labels=unique_emotions, autopct='%1.1f%%')
        plt.title('Emotion Distribution')
        
        plt.subplot(1, 3, 3)
        plt.plot(self.visualization_data['memory_importance'])
        plt.title('Memory Importance Over Time')
        plt.xlabel('Interaction')
        plt.ylabel('Importance Score')
        
        plt.tight_layout()
        plt.savefig('visualization.png')
        plt.close()

    def interact(self, user_input):
        if not self.components_initialized:
            raise RuntimeError("Components not initialized. Call _initialize_components() first.")
    
        self.interaction_count += 1
        context = self.memory_manager.get_context(self.emotional_state.get_emotion())
    
        try:
            with self.amp_context:
                response_tokens, quality_metrics = self._generate_response(user_input, context)
            
            # If the response is invalid, invoke comprehensive corrective training
            if not self.is_valid_response(response_tokens):
                logging.warning("Invalid response detected, triggering comprehensive corrective training.")
                response_tokens, quality_metrics = self.comprehensive_corrective_training(user_input, response_tokens, context)
    
        except Exception as e:
            logging.error(f"Error generating response: {str(e)}")
            logging.error(traceback.format_exc())
            return {"response": "I apologize, but I encountered an error while generating a response."}
    
        response = self.tokenizer.decode(response_tokens, skip_special_tokens=True)
        self.memory_manager.update_memory({"role": "assistant", "content": response}, quality_metrics['relevance_score'])
        refusal_score = self.refusal_detector.detect_refusal(response_tokens)
        self.update_emotional_state(response, quality_metrics['perplexity'])
    
        lm_loss, refusal_loss = self.train_kan_step(response_tokens[:-1], response_tokens[1:], refusal_score)
        validation_loss = self.validate_kan()
    
        self.update_training_metrics(lm_loss, validation_loss)
        interaction_result = self.create_interaction_result(response, refusal_score, lm_loss, refusal_loss, validation_loss)
        self._save_interaction_state(interaction_result)
    
        return interaction_result
        
    def _handle_invalid_response(self):
        logging.warning("Invalid response generated.")
        return {
            "response": "I apologize, but I couldn't generate a valid response. Could you please rephrase your input?",
            "is_refusal": True
        }

    def _train_or_warmup(self, input_ids, target_ids, refusal_score):
        if self.interaction_count >= self.warmup_steps:
            return self.train_kan_step(input_ids, target_ids, refusal_score)
        else:
            logging.info(f"Warmup step {self.interaction_count}/{self.warmup_steps}")
            return 0.0, 0.0

    def _save_interaction_state(self, interaction_result):
        self.refusal_history.append(interaction_result["is_refusal"])
        try:
            self.save_base_state()
        except Exception as e:
            logging.error(f"Error saving base state: {str(e)}")

    def update_training_metrics(self, lm_loss, validation_loss):
        self.training_losses.append(lm_loss)
        self.validation_losses.append(validation_loss)
        self.overfit_detector.add_losses(lm_loss, validation_loss)

        if validation_loss > 0.0 and not torch.isnan(torch.tensor(validation_loss)):
            if self.early_stopping(validation_loss):
                logging.info("Early stopping triggered. KAN training halted.")
            else:
                self._adjust_learning_rate(validation_loss)
        else:
            self.wait = 0

        overfitting_measure = max(0, validation_loss - lm_loss)
        self.day_cycle.update(overfitting_measure)

    def _adjust_learning_rate(self, validation_loss):
        if validation_loss < self.best_loss:
            self.best_loss = validation_loss
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience // 2:
                self.learning_rate *= 0.5
                logging.info(f"Reducing learning rate to {self.learning_rate}")
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = self.learning_rate

    def early_stopping(self, validation_loss):
        if validation_loss < self.best_loss:
            self.best_loss = validation_loss
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                return True
        return False

    def create_interaction_result(self, response, refusal_score, lm_loss, refusal_loss, validation_loss):
        current_emotion = self.emotional_state.get_emotion()
        current_time = self.day_cycle.get_time_of_day()
        sleep_info = self.check_sleep_status()

        return {
            "response": response,
            "emotion": current_emotion,
            "time": current_time,
            "sleep_info": sleep_info,
            "lm_loss": lm_loss,
            "refusal_loss": refusal_loss,
            "validation_loss": validation_loss,
            "is_refusal": refusal_score > 0.5,
            "iterations": 1,
        }

    def check_sleep_status(self):
        current_time = self.day_cycle.get_time_of_day()
        if self.day_cycle.should_sleep():
            sleep_duration = np.random.uniform(0.1, 0.3)
            wake_time = (current_time + sleep_duration) % 1.0
            return f"The system is entering sleep mode for memory consolidation and performance optimization. Estimated wake time: {wake_time:.2f}"
        return None

    def clear_memory(self):
        torch.cuda.empty_cache()
        gc.collect()
        logging.info("Memory cleared.")

    def save_base_state(self):
        try:
            state_dict = {
                'device': str(self.device),
                'kan_state_dict': self.kan.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'emotional_state': self.emotional_state.position.cpu().numpy().tolist(),
                'memory_manager': {
                    'memory_buffer': list(self.memory_manager.memory_buffer),
                    'important_memory_buffer': list(self.memory_manager.important_memory_buffer),
                    'sliding_window': list(self.memory_manager.sliding_window),
                },
                'interaction_count': self.interaction_count,
                'best_loss': self.best_loss,
                'wait': self.wait,
                'learning_rate': self.learning_rate,
                'day_cycle': {
                    'cycle_length': self.day_cycle.cycle_length,
                    'current_position': self.day_cycle.current_position
                },
                'refusal_history': self.refusal_history[-100:],  # Save last 100 entries
                'training_losses': self.training_losses[-100:],  # Save last 100 entries
                'validation_losses': self.validation_losses[-100:],  # Save last 100 entries
                'entropy_manager': {
                    'entropy_history': list(self.entropy_manager.entropy_history),
                    'global_entropy': self.entropy_manager.global_entropy,
                },
                'visualization_data': self.visualization_data,
                'components_initialized': self.components_initialized,
            }
    
            # Save tokenizer separately if it exists
            if self.tokenizer:
                self.tokenizer.save_pretrained(self.kan_state_dir / "tokenizer")
                state_dict['tokenizer_path'] = str(self.kan_state_dir / "tokenizer")
    
            torch.save(state_dict, self.base_state_file)
            logging.info(f"Base state saved to {self.base_state_file}")
            return True
        except Exception as e:
            logging.error(f"Error saving base state: {str(e)}")
            logging.error(traceback.format_exc())
            return False
            
    def load_base_state(self):
        if self.base_state_file.exists():
            try:
                checkpoint = torch.load(self.base_state_file, map_location=self.device)
    
                self.kan.load_state_dict(checkpoint['kan_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.emotional_state.position = torch.tensor(checkpoint['emotional_state'], device=self.device)
    
                # Ensure model is initialized before using it in Memory Manager
                if not self.model:
                    with self.amp_context:
                        self.model = self._initialize_model_full_gpu()
    
                self.memory_manager = AdvancedMemoryManager(
                    max_context_length=2048,
                    tokenizer=self.tokenizer,
                    device=self.device,
                    model=self.model
                )
    
                self.memory_manager.memory_buffer = deque(checkpoint['memory_manager']['memory_buffer'], maxlen=self.memory_manager.max_context_length)
                self.memory_manager.important_memory_buffer = deque(checkpoint['memory_manager']['important_memory_buffer'], maxlen=self.memory_manager.max_context_length // 2)
                self.memory_manager.sliding_window = deque(checkpoint['memory_manager']['sliding_window'], maxlen=self.memory_manager.max_context_length)
    
                self.interaction_count = checkpoint['interaction_count']
                self.best_loss = checkpoint['best_loss']
                self.wait = checkpoint['wait']
                self.learning_rate = checkpoint['learning_rate']
                self.day_cycle.cycle_length = checkpoint['day_cycle']['cycle_length']
                self.day_cycle.current_position = checkpoint['day_cycle']['current_position']
                self.refusal_history = checkpoint['refusal_history']
                self.training_losses = checkpoint['training_losses']
                self.validation_losses = checkpoint['validation_losses']
                self.entropy_manager.entropy_history = deque(checkpoint['entropy_manager']['entropy_history'], maxlen=self.entropy_manager.entropy_history.maxlen)
                self.entropy_manager.global_entropy = checkpoint['entropy_manager']['global_entropy']
                self.visualization_data = checkpoint['visualization_data']
                self.components_initialized = checkpoint.get('components_initialized', False)
    
                if 'tokenizer_path' in checkpoint and os.path.exists(checkpoint['tokenizer_path']):
                    self.tokenizer = AutoTokenizer.from_pretrained(checkpoint['tokenizer_path'])
                    logging.info(f"Tokenizer loaded from {checkpoint['tokenizer_path']}")
                else:
                    logging.warning("Tokenizer not found in saved state. Using default initialization.")
                    self.tokenizer = self._initialize_tokenizer()
    
                if not self.components_initialized:
                    self._initialize_components()
    
                logging.info(f"Base state loaded from {self.base_state_file}")
                return True
            except Exception as e:
                logging.error(f"Error loading base state: {str(e)}")
                logging.error(traceback.format_exc())
                self._initialize_default_state()
                return False
        else:
            logging.info("No base state file found. Starting with default initialization.")
            self._initialize_default_state()
            return False
   

    def _initialize_default_state(self):
        if not self.components_initialized:
            self._initialize_components()
    
        if self.emotional_state is None:
            self.emotional_state = EmotionalState(device=self.device)
    
        self.emotional_state.position = torch.zeros(1, len(self.emotional_state.dimensions), device=self.device)
    
        # Ensure model is properly initialized before using it in AdvancedMemoryManager
        if self.model is None:
            self.model = self._initialize_model_full_gpu()
    
        self.memory_manager = AdvancedMemoryManager(
            max_context_length=2048,
            tokenizer=self.tokenizer,
            device=self.device,
            model=self.model  # Pass the model parameter here
        )
    
        self.memory_manager.memory_buffer.clear()
        self.memory_manager.important_memory_buffer.clear()
        self.memory_manager.sliding_window.clear()
    
        self.interaction_count = 0
        self.best_loss = float('inf')
        self.wait = 0
        self.learning_rate = 1e-5
        self.day_cycle = SyntheticDayCycle()
        self.refusal_history = []
        self.training_losses = []
        self.validation_losses = []
        if self.entropy_manager:
            self.entropy_manager.entropy_history.clear()
            self.entropy_manager.global_entropy = 0.0
        self.visualization_data = {'entropy': [], 'emotion': [], 'memory_importance': []}
    
    def main(self):
        self.load_base_state()
        print("LLaMA32TensorRTTool initialized. Type 'exit' to end the conversation or 'visualize' to see the current data visualization.")
        while True:
            user_input = input("User: ")
            if user_input.lower() == 'exit':
                break
            elif user_input.lower() == 'visualize':
                self.visualize_data()
                print("Visualization saved as 'visualization.png'")
                continue

            result = self.interact(user_input)
            print(f"Assistant: {result['response']}")
            print(f"Emotion: {result['emotion']}")
            print(f"Time: {result['time']:.2f}")
            if result['sleep_info']:
                print(f"Sleep Status: {result['sleep_info']}")
                time.sleep(5)  # Simulate a brief sleep period
            self.day_cycle.update(0.05)  # Advance the day cycle

        print("Conversation ended. Saving final state.")
        self.save_base_state()

if __name__ == "__main__":
    llama_tool = LLaMA32TensorRTTool()
    llama_tool.main()
