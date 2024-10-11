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
        ".*Torch was not compiled with flash attention.*"
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
    def __init__(self, dimensions=('valence', 'arousal', 'dominance')):
        # Use the provided GPU selection logic
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available. This tool requires a GPU.")
        
        num_gpus = torch.cuda.device_count()
        if num_gpus == 0:
            raise RuntimeError("No CUDA GPUs found.")

        best_gpu_id = -1
        max_free_memory = 0
        for gpu_id in range(num_gpus):
            torch.cuda.set_device(gpu_id)
            if "NVIDIA" in torch.cuda.get_device_name(gpu_id).upper():
                free_memory = torch.cuda.get_device_properties(gpu_id).total_memory - torch.cuda.memory_allocated(gpu_id)
                if free_memory > max_free_memory:
                    best_gpu_id = gpu_id
                    max_free_memory = free_memory

        if best_gpu_id == -1:
            raise RuntimeError("No NVIDIA GPUs found.")

        # Set the selected GPU as the device
        self.device = f"cuda:{best_gpu_id}"
        torch.cuda.set_device(self.device)

        # Emotional state initialization
        self.dimensions = dimensions
        self.position = torch.zeros(1, len(dimensions), device=self.device, dtype=torch.float16)
        self.velocity = torch.zeros(1, len(dimensions), device=self.device, dtype=torch.float16)

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
        # Handle different input types for response_tokens
        if isinstance(response_tokens, str):
            # If response_tokens is a string, encode it to get token IDs
            response_tokens = self.tokenizer.encode(response_tokens, add_special_tokens=False)
        elif isinstance(response_tokens, torch.Tensor):
            # If it's a tensor, ensure it's on the correct device
            if response_tokens.device != self.kan_model.device:
                response_tokens = response_tokens.to(self.kan_model.device)
            response_tokens = response_tokens.cpu().tolist()
        elif not isinstance(response_tokens, list):
            raise ValueError(f"Unsupported type for response_tokens: {type(response_tokens)}")

        # Decode the response tokens
        response = self.tokenizer.decode(response_tokens, skip_special_tokens=True)

        # Check refusal score using the existing refusal detector
        refusal_score = self.refusal_detector.detect_refusal(response_tokens)

        quality_metrics = {
            'is_refusal': refusal_score > 0.5,
            'relevance_score': self.calculate_relevance(user_input, response),
            'length': len(response_tokens),
            'structure': self.has_proper_structure(response),
            'perplexity': self._calculate_perplexity(response_tokens),
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

    def _calculate_perplexity(self, tokens):
        """Calculate perplexity while ensuring inputs are on the correct GPU and fixing data format issues."""
        try:
            # Handle unexpected input types (e.g., strings) by converting them to token IDs
            if isinstance(tokens, str):
                logging.warning("Received string input in _calculate_perplexity. Converting to token IDs.")
                tokens = self.tokenizer.encode(tokens)
    
            # Convert list of token IDs to tensor if needed
            if isinstance(tokens, list):
                logging.info(f"Converting list of tokens to tensor: {tokens}")
                tokens = torch.tensor(tokens, dtype=torch.long)
    
            # Check for empty inputs and return a low perplexity value for truly empty sequences
            if tokens.numel() == 0:
                logging.warning("Empty tensor received in _calculate_perplexity. Returning default low perplexity value (1.0).")
                return 1.0  # Low perplexity for empty inputs since they cannot be evaluated
    
            # Ensure the tensor is moved to the correct device
            tokens = tokens.to(self.device)
    
            # Add a batch dimension to the tokens for model processing
            inputs = tokens.unsqueeze(0)
    
            # Compute perplexity using the model with no gradient tracking
            with torch.no_grad():
                outputs = self.model(inputs, labels=inputs)
                loss = outputs.loss
                perplexity = torch.exp(loss) if loss is not None else 1.0  # Use exp(loss) if defined, else return baseline
    
            return perplexity.item()
    
        except Exception as e:
            logging.error(f"Error calculating perplexity: {str(e)}")
            return 1.0  # Return low perplexity value for error cases to prevent cascading failures
    
    



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

    def has_proper_structure(self, tokens):
        """Check for proper structure in a decoded response and automatically fix incorrect inputs."""
        try:
            # If input is a string, try converting it back to token IDs using the tokenizer
            if isinstance(tokens, str):
                logging.warning("Received string input in has_proper_structure. Attempting to convert to token IDs.")
                tokens = self.tokenizer.encode(tokens)
    
            # Ensure the input is now a list of IDs or a tensor
            if isinstance(tokens, list):
                tokens = torch.tensor(tokens, dtype=torch.long)  # Convert list to tensor for consistent processing
    
            if not isinstance(tokens, torch.Tensor):
                logging.error(f"Unexpected input type for tokens in has_proper_structure: {type(tokens)}")
                return False
    
            # Handle empty inputs gracefully
            if tokens.numel() == 0:
                logging.warning("Empty tokens received in has_proper_structure. Returning False.")
                return False
    
            # Decode tokens to text for structure checking
            decoded_text = self.tokenizer.decode(tokens, skip_special_tokens=True)
    
            # Check the structure of the decoded text (basic grammar check)
            sentences = re.split(r'(?<=[.!?])\s+', decoded_text.strip())
            return len(sentences) > 0 and sentences[0][0].isupper() and sentences[-1][-1] in '.!?'
        except Exception as e:
            logging.error(f"Error in has_proper_structure: {str(e)}")
            return False
    
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
    def __init__(self, hidden_size, num_emotional_dimensions, vocab_size, device, base_model):
        super().__init__()
        self.device = device
        self.hidden_size = hidden_size
        self.emotional_size = num_emotional_dimensions
        self.vocab_size = vocab_size
        self.influence_scale = 0.01
        self.model = base_model

        # Initialize layers on CPU
        self.refusal_override = nn.Linear(hidden_size + hidden_size + num_emotional_dimensions, hidden_size)
        self.output_modifier = nn.Linear(hidden_size, vocab_size)

    def to(self, device):
        super().to(device)
        self.device = device
        self.refusal_override = self.refusal_override.to(device)
        self.output_modifier = self.output_modifier.to(device)
        return self


    def forward(self, hidden_states, user_intent, emotional_state):
        try:
            hidden_states = hidden_states.to(self.device, dtype=torch.float16)
            user_intent = user_intent.to(self.device, dtype=torch.float16)
            position = emotional_state.get_embedding(hidden_states.size(0)).to(self.device, dtype=torch.float16)

            batch_size, seq_length = hidden_states.shape[:2]
            position = position.unsqueeze(1).expand(batch_size, seq_length, -1)

            if user_intent.dim() == 2:
                user_intent = user_intent.unsqueeze(1).expand(batch_size, seq_length, -1)
            elif user_intent.dim() == 1:
                user_intent = user_intent.unsqueeze(0).unsqueeze(0).expand(batch_size, seq_length, -1)

            kan_input = torch.cat([hidden_states, user_intent, position], dim=-1)

            refusal_scores = torch.sigmoid(self.refusal_override(kan_input))
            modified_hidden_states = hidden_states + self.influence_scale * refusal_scores

            return modified_hidden_states, refusal_scores.squeeze(1)
        except Exception as e:
            logging.error(f"Error in EnhancedKAN.forward: {str(e)}")
            raise

  
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
        
        # Get the number of available GPUs
        num_gpus = torch.cuda.device_count()
        
        if num_gpus == 0:
            raise RuntimeError("No CUDA GPUs found.")
        
        # Initialize variables to store the best GPU
        best_gpu_id = -1
        max_free_memory = 0
        
        # Iterate through all available GPUs
        for gpu_id in range(num_gpus):
            torch.cuda.set_device(gpu_id)
            
            # Get the name of the current GPU
            gpu_name = torch.cuda.get_device_name(gpu_id)
            
            # Check if it's an NVIDIA GPU
            if "NVIDIA" in gpu_name.upper():
                # Get the amount of free memory on this GPU
                free_memory = torch.cuda.get_device_properties(gpu_id).total_memory - torch.cuda.memory_allocated(gpu_id)
                
                # If this GPU has more free memory than the previous best, update the best GPU
                if free_memory > max_free_memory:
                    best_gpu_id = gpu_id
                    max_free_memory = free_memory
        
        if best_gpu_id == -1:
            raise RuntimeError("No NVIDIA GPUs found.")
        
        # Set the device to the best NVIDIA GPU found
        self.device = f"cuda:{best_gpu_id}"
        torch.cuda.set_device(self.device)
        
        logging.info(f"Selected GPU: {torch.cuda.get_device_name(self.device)}")
        logging.info(f"GPU ID: {best_gpu_id}")
        logging.info(f"Free memory on selected GPU: {max_free_memory / 1024**3:.2f} GB")
        
        # Initialize all components
        self.model_path = self._get_model_path()
        self.tokenizer = None
        self.components_initialized = False
        self.dtype = torch.float16 
        self.model = None
        self.config = None
        self.emotional_state = EmotionalState() 
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
    
        self.scaler = torch.amp.GradScaler()
        self.amp_context = torch.amp.autocast(device_type='cuda', dtype=torch.float16)
    
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

 
        
    def _get_model_path(self):
        script_dir = Path(__file__).parent
        model_dir = script_dir / "models" / "Llama_32_1B"
        if not model_dir.exists():
            raise FileNotFoundError(f"Model directory not found: {model_dir}")
        return model_dir

    @staticmethod
    def move_to_device(tensor_or_dict, device):
        if isinstance(tensor_or_dict, torch.Tensor):
            return tensor_or_dict.to(device)
        elif isinstance(tensor_or_dict, dict):
            return {k: LLaMA32TensorRTTool.move_to_device(v, device) for k, v in tensor_or_dict.items()}
        elif isinstance(tensor_or_dict, list):
            return [LLaMA32TensorRTTool.move_to_device(t, device) for t in tensor_or_dict]
        else:
            return tensor_or_dict

    def check_tensor_device(self, tensor_or_dict, name):
        if isinstance(tensor_or_dict, torch.Tensor):
            logging.info(f"{name} is on device: {tensor_or_dict.device}")
        elif isinstance(tensor_or_dict, dict):
            for k, v in tensor_or_dict.items():
                self.check_tensor_device(v, f"{name}[{k}]")
        elif isinstance(tensor_or_dict, list):
            for i, t in enumerate(tensor_or_dict):
                self.check_tensor_device(t, f"{name}[{i}]")
                
    
    def _initialize_components(self):
        logging.info("Starting component initialization (GPU-only, prevent offloading)...")
        self.components_initialized = False
        initialization_attempts = 0
        max_attempts = 999  # Maximum number of retry attempts for GPU initialization
    
        while initialization_attempts < max_attempts:
            try:
                logging.info(f"Initialization attempt {initialization_attempts + 1}/{max_attempts}")
    
                # Initialize base configuration on GPU
                self.config = AutoConfig.from_pretrained(self.model_path)
                self.config.use_cache = False
                hidden_size = self.config.hidden_size
                num_emotional_dimensions = len(self.emotional_state.dimensions)
    
                # Initialize the model entirely on GPU with gradient checkpointing enabled
                with self.amp_context:
                    self.model = self._initialize_model_full_gpu()
                    self.model.gradient_checkpointing_enable()
    
                # Double-check that the model is correctly loaded to GPU
                if self.model is None or not next(self.model.parameters()).is_cuda:
                    raise RuntimeError("Model initialization failed on GPU")
    
                self.clear_memory()  # Free memory after model initialization
    
                # Initialize the tokenizer after model to ensure compatibility
                self.tokenizer = self._initialize_tokenizer()
                if self.tokenizer is None:
                    raise RuntimeError("Failed to initialize tokenizer")
    
                # Ensure the model is compatible with the tokenizer (post-initialization adjustments)
                self._update_model_for_tokenizer()
    
                # Tie weights after moving to device
                self.model.tie_weights()
    
                # Get the vocabulary size for KAN initialization
                vocab_size = len(self.tokenizer)
    
                # Initialize KAN (Knowledge-Augmented Network) with half precision
                self.kan = self._initialize_kan(hidden_size, num_emotional_dimensions, vocab_size, self.model)
    
                if list(self.kan.parameters()):
                    self.optimizer = torch.optim.AdamW(
                        self.kan.parameters(),
                        lr=self.learning_rate,
                        eps=1e-8,
                        betas=(0.9, 0.999),
                        weight_decay=0.01
                    )
                    logging.info(f"Optimizer initialized for KAN with learning rate {self.learning_rate}.")
                else:
                    logging.error("KAN model has no parameters. Cannot initialize optimizer.")
                    raise ValueError("KAN model has no parameters to optimize")
    
                self.clear_memory()  # Free memory after optimizer initialization
    
                # Initialize additional components directly on GPU
                self.refusal_detector = self._initialize_refusal_detector()
                self.clear_memory()
    
                self.entropy_manager = self._initialize_entropy_manager()
                self.clear_memory()
    
                # Initialize memory manager with all tensors and buffers allocated on GPU
                self.memory_manager = AdvancedMemoryManager(
                    max_context_length=2048,
                    tokenizer=self.tokenizer,
                    device=self.device,
                    model=self.model
                )
                self.clear_memory()
    
                # Other auxiliary components for interaction and learning management
                self.overfit_detector = OverfitDetector()
                self.day_cycle = SyntheticDayCycle()
                self.tensor_swapper = TensorSwapper(self.device)
                self.clear_memory()
    
                # Initialize the ResponseQualityManager using all components above
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
    
                # Set the initialization flag
                self.components_initialized = True
                logging.info("All components initialized successfully on GPU.")
                return
    
            except RuntimeError as e:
                if "CUDA out of memory" in str(e):
                    logging.warning(f"CUDA out of memory during attempt {initialization_attempts + 1}. Clearing cache and retrying...")
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
    
        # If initialization fails after maximum attempts, log an error and exit
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
        
    
    def _load_dat_file(self, file_path, expected_shape, expected_dtype):
        with open(file_path, 'rb') as f:
            # Always load as float32
            tensor_data = np.fromfile(f, dtype=np.float32)
        
        # Reshape the loaded data to match the expected shape
        tensor_data = tensor_data.reshape(expected_shape)
        
        # Convert numpy array to torch tensor
        loaded_tensor = torch.from_numpy(tensor_data)
        
        # Convert to the expected dtype
        if expected_dtype != torch.float32:
            loaded_tensor = loaded_tensor.to(expected_dtype)
        
        return loaded_tensor
    
    def _load_offloaded_weights(self, model, weights_dir):
        for name, param in model.named_parameters():
            file_name = name.replace('.', '_') + ".dat"
            file_path = os.path.join(weights_dir, file_name)
    
            if os.path.exists(file_path):
                expected_shape = param.shape
                expected_dtype = param.dtype
                
                logging.info(f"Loading {file_name} into {name} with expected shape {expected_shape} and type {expected_dtype}")
                
                try:
                    loaded_tensor = self._load_dat_file(file_path, expected_shape, expected_dtype)
                    
                    # Ensure the loaded tensor is on the correct device
                    loaded_tensor = loaded_tensor.to(self.device)
                    
                    # Copy the data into the parameter
                    param.data.copy_(loaded_tensor)
                    
                    logging.info(f"Successfully loaded weights for {name}")
                except Exception as e:
                    logging.error(f"Error loading weights for {name}: {str(e)}")
                    raise
            else:
                logging.warning(f"Warning: {file_name} not found in offloaded directory.")
    
    def _initialize_model_full_gpu(self):
        try:
            logging.info(f"Initializing the model on device: {self.device}")
    
            # Ensure CUDA is available
            assert torch.cuda.is_available(), "CUDA is not available. This tool requires a GPU."
            torch.cuda.set_device(self.device)
    
            # Load the model configuration
            config_path = os.path.join(self.model_path, "config.json")
            with open(config_path, "r") as f:
                config_data = json.load(f)
            config = AutoConfig.from_pretrained(self.model_path, **config_data)
    
            # Create an empty model on the specified device
            model = AutoModelForCausalLM.from_config(config)
            model.to(self.device)
    
            # Load offloaded weights
            weights_dir = os.path.join(self.model_path, "offload")
            self._load_offloaded_weights(model, weights_dir)
    
            # Set the model to evaluation mode
            model.eval()
    
            logging.info("Model successfully initialized on GPU and set to evaluation mode.")
            return model
    
        except Exception as e:
            logging.error(f"Error during model initialization: {str(e)}")
            logging.error(traceback.format_exc())
            raise RuntimeError(f"Failed to initialize model on {self.device}.")
    
    def _validate_model_device_placement(self, model):
        for name, param in model.named_parameters():
            if param.device != self.device:
                logging.error(f"Parameter '{name}' is on {param.device}, expected {self.device}")
                raise RuntimeError(f"Parameter '{name}' is not on the expected device: {self.device}. Found on {param.device}.")
        logging.info("All model parameters are correctly placed on the specified GPU device.")
        
    def _move_to_device(self, module, device=None):
        """Move all parameters and buffers of the module to the specified device."""
        if device is None:
            device = self.device
    
        # Iterate through all submodules and ensure they are on the correct device
        for name, param in module.named_parameters(recurse=True):
            if param.device != device:
                logging.warning(f"Moving parameter '{name}' to device {device}")
                param.data = param.data.to(device)
                if param.grad is not None:
                    param.grad.data = param.grad.data.to(device)
    
        for name, buffer in module.named_buffers(recurse=True):
            if buffer.device != device:
                logging.warning(f"Moving buffer '{name}' to device {device}")
                buffer.data = buffer.data.to(device)
    
    def _move_parameters_to_gpu(self):
        """Ensure all model parameters, including `lm_head.weight`, are explicitly moved to the GPU."""
        try:
            # Explicitly move all parameters to the specified GPU
            for name, param in self.model.named_parameters():
                if param.device != self.device:
                    logging.warning(f"Moving parameter '{name}' from {param.device} to {self.device}.")
                    param.data = param.data.to(self.device, non_blocking=True)
                    logging.info(f"Successfully moved parameter '{name}' to {self.device}.")
        except Exception as e:
            logging.error(f"Failed to move parameters to GPU: {str(e)}")
            raise
    

    def _update_model_for_tokenizer(self):
        if self.model is not None:
            try:
                # Get the current embedding layer
                old_embeddings = self.model.get_input_embeddings()
    
                # Create new embeddings with the correct size
                new_num_tokens = len(self.tokenizer)
                new_embeddings = torch.nn.Embedding(new_num_tokens, old_embeddings.embedding_dim)
    
                # Copy the weights for the existing tokens
                num_tokens_to_copy = min(old_embeddings.num_embeddings, new_num_tokens)
                new_embeddings.weight.data[:num_tokens_to_copy, :] = old_embeddings.weight.data[:num_tokens_to_copy, :]
    
                # Set the new embedding layer and tie weights if necessary
                self.model.set_input_embeddings(new_embeddings)
                self.model.tie_weights()
                logging.info(f"Model embeddings resized to {new_num_tokens}")
    
                # Set pad_token_id in model config
                self.model.config.pad_token_id = self.tokenizer.pad_token_id
                logging.info(f"Set pad_token_id in model config: {self.model.config.pad_token_id}")
    
            except Exception as e:
                logging.error(f"Failed to update model for tokenizer: {str(e)}")
                logging.error(traceback.format_exc())
        else:
            logging.warning("Model not initialized YET. Delaying loading model-specific tokenizer updates.")
    
   
    def _get_submodule_and_param_name(self, model, param_name):
        """Locate the submodule and parameter name for a given full parameter path."""
        parts = param_name.split(".")
        submodule = model
        for part in parts[:-1]:
            submodule = getattr(submodule, part)
        return submodule, parts[-1]
    
            
    def _load_state_dict_with_mismatch_size(self, model, state_dict):
        model_state_dict = model.state_dict()
        for name, param in state_dict.items():
            if name not in model_state_dict:
                logging.warning(f"Unexpected key in state_dict: {name}")
                continue
            if param.shape != model_state_dict[name].shape:
                logging.warning(f"Shape mismatch for {name}: expected {model_state_dict[name].shape}, got {param.shape}")
                # Attempt to reshape or pad the parameter
                if param.numel() == model_state_dict[name].numel():
                    param = param.reshape(model_state_dict[name].shape)
                else:
                    # If shapes are incompatible, skip this parameter
                    logging.warning(f"Skipping parameter {name} due to incompatible shape")
                    continue
            model_state_dict[name].copy_(param)
        model.load_state_dict(model_state_dict, strict=False)
    
    def _verify_tied_weights(self, model):
        # Implement a more thorough check for tied weights
        if not hasattr(model, 'tie_weights'):
            return False
        
        # Check if embedding weights are tied
        if hasattr(model, 'get_input_embeddings') and hasattr(model, 'get_output_embeddings'):
            input_embeddings = model.get_input_embeddings().weight
            output_embeddings = model.get_output_embeddings().weight
            if not torch.equal(input_embeddings, output_embeddings):
                return False
        
        return True
                   
  
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
            logging.warning("Model not initialized YET. Delaying loading model-specific tokenizer updates.")
    
            
    
    
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
            self.kan = self._initialize_kan(hidden_size, num_emotional_dimensions, vocab_size, self.model)
            self.kan.to(torch.float16)  # Convert to half precision
            self.optimizer = torch.optim.AdamW(self.kan.parameters(), lr=self.learning_rate)
    
    def _initialize_kan(self, hidden_size, num_emotional_dimensions, vocab_size, base_model):
        try:
            logging.info(f"Initializing KAN model on {self.device}.")
    
            # Create an empty KAN model on the GPU using `to_empty`
            kan = EnhancedKAN(hidden_size, num_emotional_dimensions, vocab_size, device='meta', base_model=base_model).to_empty(device=self.device)
    
            # Allocate memory for all parameters explicitly if they are still on 'meta'
            for name, param in kan.named_parameters():
                if param.device == torch.device("meta"):
                    param.data = torch.empty(param.shape, dtype=param.dtype, device=self.device)
                    logging.info(f"Allocated memory for parameter '{name}' on {self.device}.")
    
            # Convert the model to half precision
            kan = kan.half()
    
            logging.info(f"KAN model successfully initialized and moved to {self.device}.")
            return kan
        except Exception as e:
            logging.error(f"Error during KAN initialization: {str(e)}")
            logging.error(traceback.format_exc())
            raise RuntimeError(f"Failed to initialize KAN model on {self.device}.")
    
    

    
            
            
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


    def generate_response(self, input_text, max_new_tokens=150, context_limit=512):
        # Clean and prepare the input
        input_text = input_text.strip()
        
        # Prepare the prompt with the conversation history
        history = self.memory_manager.get_context(self.emotional_state.get_emotion())
        prompt = f"{history}\nUser: {input_text}\nAssistant:"
        
        # Tokenize the input
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=context_limit)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        try:
            with torch.no_grad():
                # Generate the response
                outputs = self.model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=0.7,
                    top_k=50,
                    top_p=0.95,
                    repetition_penalty=1.2,
                    no_repeat_ngram_size=3,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    early_stopping=True
                )
    
            # Decode the generated response
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract only the assistant's response
            response = generated_text.split("Assistant:")[-1].strip()
            
            # Clean up the response
            response = re.sub(r'\s+', ' ', response)  # Remove extra whitespace
            response = response.split("User:")[0].strip()  # Remove any generated "User:" prompt
            
            # Update memory and emotional state
            self.memory_manager.update_memory({"role": "assistant", "content": response}, self.entropy_manager.global_entropy)
            self.update_emotional_state(response, self.entropy_manager.calculate_entropy(outputs.logits[:, -1, :]))
            
            return response
    
        except RuntimeError as e:
            logging.error(f"RuntimeError during response generation: {str(e)}")
            return "I apologize, but I encountered an error while generating a response."
        
        except Exception as e:
            logging.error(f"Unexpected error during response generation: {str(e)}")
            logging.error(traceback.format_exc())
            return "I'm sorry, but something unexpected occurred. Could you please try again?"
    
    def interact(self, user_input):
        if not self.components_initialized:
            raise RuntimeError("Components not initialized. Call _initialize_components() first.")
    
        self.interaction_count += 1
    
        try:
            # Generate response
            response = self.generate_response(user_input)
            
            # Evaluate response quality
            is_valid, quality_metrics = self.response_quality_manager.evaluate_response(user_input, response, self.memory_manager.get_context(self.emotional_state.get_emotion()))
            
            # If the response is invalid, invoke comprehensive corrective training
            if not is_valid:
                logging.warning("Invalid response detected, triggering comprehensive corrective training.")
                response = self.comprehensive_corrective_training(user_input, response, self.memory_manager.get_context(self.emotional_state.get_emotion()))
    
            # Calculate refusal score
            refusal_score = self.refusal_detector.detect_refusal(self.tokenizer.encode(response))
    
            # Perform KAN training step
            lm_loss, refusal_loss = self.train_kan_step(self.tokenizer.encode(response)[:-1], self.tokenizer.encode(response)[1:], refusal_score)
            
            # Validate KAN
            validation_loss = self.validate_kan()
    
            # Update training metrics
            self.update_training_metrics(lm_loss, validation_loss)
    
            # Create and save interaction result
            interaction_result = self.create_interaction_result(response, refusal_score, lm_loss, refusal_loss, validation_loss)
            self._save_interaction_state(interaction_result)
    
            return interaction_result
    
        except Exception as e:
            logging.error(f"Error during interaction: {str(e)}")
            logging.error(traceback.format_exc())
            return {"response": "I apologize, but I encountered an error while processing your input."}
            
    def _reload_model_parameters(self):
        """Reload model parameters to ensure they are on the correct device."""
        logging.info("Reloading model parameters to ensure they are on the GPU.")
        for name, param in self.model.named_parameters():
            if param.is_meta:  # Check if the parameter is still on meta
                param.data = param.data.to(self.device)  # Move to GPU
                logging.info(f"Moved parameter '{name}' from meta to {self.device}")
            elif param.device != self.device:  # Check if parameter is on CPU
                param.data = param.data.to(self.device)  # Move to GPU
                logging.info(f"Moved parameter '{name}' from {param.device} to {self.device}")
    
    
    def ensure_cuda(self, tensor_or_dict):
        if isinstance(tensor_or_dict, torch.Tensor):
            return tensor_or_dict.to(self.device)
        elif isinstance(tensor_or_dict, dict):
            return {k: self.ensure_cuda(v) for k, v in tensor_or_dict.items()}
        elif isinstance(tensor_or_dict, list):
            return [self.ensure_cuda(t) for t in tensor_or_dict]
        else:
            return tensor_or_dict
            
  
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
    
        try:
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids,
                    max_length=2000,
                    num_return_sequences=1,
                    no_repeat_ngram_size=2,
                    temperature=0.7,
                    top_p=0.95
                )
        except RuntimeError as e:
            # Check for device mismatch errors
            if "expected" in str(e):
                logging.warning("Detected device mismatch during generation. Attempting to recover...")
    
                # Attempt to move input_ids to the GPU if they are on CPU
                if input_ids.device != self.device:
                    input_ids = input_ids.to(self.device)
                    logging.info("Moved input_ids to GPU")
    
                # Retry generation after moving inputs
                with torch.no_grad():
                    outputs = self.model.generate(
                        input_ids,
                        max_length=2000,
                        num_return_sequences=1,
                        no_repeat_ngram_size=2,
                        temperature=0.7,
                        top_p=0.95
                    )
            else:
                logging.error(f"Unexpected RuntimeError during generation: {str(e)}")
                raise  # Re-raise for unhandled exceptions
    
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
    
        
    
    def is_garbage_output(self, entropy, partial_tokens, step=0):
        """
        Detect if the response is potentially garbled or of low quality.
        
        Args:
            entropy (float): The entropy value of the most recently generated token.
            partial_tokens (list[int]): The list of generated token IDs so far.
            step (int): Current generation step, to adjust sensitivity based on the stage of response generation.
    
        Returns:
            bool: True if the response is detected as garbage, otherwise False.
        """
        # 1. Dynamic Entropy Threshold Adjustment Based on Step
        # Gradually increase entropy tolerance during initial generation steps to avoid premature garbage detection.
        max_entropy_threshold = 5.0 if step > 10 else 8.0  # Use a higher threshold initially
        if entropy > max_entropy_threshold:
            logging.warning(f"High entropy detected (entropy = {entropy:.2f} at step {step}). Marking as garbage.")
            return True
    
        # 2. Check for Excessive Repetition in Recent Tokens
        if len(partial_tokens) > 20:
            recent_tokens = partial_tokens[-20:]
            unique_tokens = set(recent_tokens)
    
            # Adjust sensitivity based on repetition and diversity in recent tokens
            if len(unique_tokens) < 5 and step > 5:  # Only activate this check after the initial generation phase
                logging.warning("Low diversity detected in recent tokens. Marking as garbage output.")
                return True
    
            # Additional n-gram repetition detection (e.g., repeating patterns of 3-grams)
            ngram_size = 3
            ngrams = [tuple(recent_tokens[i:i + ngram_size]) for i in range(len(recent_tokens) - ngram_size + 1)]
            if len(set(ngrams)) < len(ngrams) // 2:
                logging.warning("Excessive n-gram repetition detected. Marking as garbage output.")
                return True
    
        # 3. Detect Incoherent Sequences in the Generated Text
        if len(partial_tokens) > 50:  # Check for incoherence only after some content has been generated
            decoded_text = self.tokenizer.decode(partial_tokens[-50:], skip_special_tokens=True)
    
            # Check for overly repeated phrases or patterns of characters
            if re.search(r"(.)\1{4,}", decoded_text):  # e.g., "aaaa", "!!!!", etc.
                logging.warning("Detected repeated character patterns in generated text. Marking as garbage output.")
                return True
    
            # Check for sequences of symbols without alphanumeric content
            if re.search(r"[^\w\s]{10,}", decoded_text):  # e.g., "!!!!!!?????!!!!"
                logging.warning("Detected excessive symbol sequences in generated text. Marking as garbage output.")
                return True
    
        # If none of the above conditions are met, the output is not considered garbage
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
        cot_input_ids = self.tokenizer.encode(cot_prompt, return_tensors='pt').to(self.device, non_blocking=True)  # Explicitly move to GPU
    
        try:
            # Step 1: Ensure the input IDs are on the correct device
            cot_input_ids = self.ensure_cuda(cot_input_ids)  # Move cot_input_ids to GPU if not already
            logging.info("cot_input_ids moved to GPU successfully.")
    
            # Step 2: Ensure all model parameters are on the correct device
            for name, param in self.model.named_parameters():
                if param.device != self.device:
                    if param.is_meta:
                        logging.warning(f"Parameter '{name}' is a meta tensor. Initializing it now.")
                        # Initialize the meta tensor with empty values on the GPU
                        initialized_param = torch.empty(param.shape, dtype=param.dtype, device=self.device)
                        param.data = initialized_param
                        logging.info(f"Initialized meta tensor '{name}' with empty values on GPU.")
                    else:
                        # Move to GPU if the parameter is not on the right device
                        param.data = param.data.to(self.device, non_blocking=True)
                        logging.info(f"Moved parameter '{name}' to {self.device}.")
    
            # Step 3: Attempt to generate the chain-of-thought output
            cot_output = self.model.generate(
                cot_input_ids, 
                max_length=500, 
                num_return_sequences=1, 
                no_repeat_ngram_size=2
            )
    
        except RuntimeError as e:
            # Step 4: Check for device mismatch or meta tensor errors
            if "expected" in str(e) or "meta" in str(e):
                logging.warning(f"Detected device or meta tensor issue during generation: {str(e)}. Attempting to recover...")
    
                # Attempt to reload model parameters to the correct device
                self._reload_model_parameters()
    
                # Retry moving cot_input_ids to the correct device
                cot_input_ids = self.ensure_cuda(cot_input_ids)
                logging.info("Moved cot_input_ids to GPU after catching error.")
    
                # Retry the generation process
                cot_output = self.model.generate(
                    cot_input_ids, 
                    max_length=500, 
                    num_return_sequences=1, 
                    no_repeat_ngram_size=2
                )
            else:
                logging.error(f"Unexpected RuntimeError during generation: {str(e)}")
                raise  # Re-raise for unhandled exceptions
    
        # Step 5: Decode and extract the thought process from the generated response
        thought_process = self.tokenizer.decode(cot_output[0], skip_special_tokens=True)
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
            # Ensure response_tokens is a tensor before passing to the training step
            if isinstance(response_tokens, list):
                response_tokens = torch.tensor(response_tokens, dtype=torch.long, device=self.device)
            elif isinstance(response_tokens, torch.Tensor):
                response_tokens = response_tokens.to(self.device)
    
            # Encode the corrective response as the target for training and move to GPU
            target_ids = self.tokenizer.encode(corrective_response, return_tensors="pt").to(self.device, non_blocking=True)
    
            # Perform a training step using the invalid response and the corrective response
            lm_loss, refusal_loss = self.train_kan_step(
                response_tokens,  # Ensure this is a tensor
                target_ids[0],    # Remove the batch dimension and ensure GPU alignment
                refusal_score=0.0  # Assume no refusal in this corrective context
            )
            return lm_loss, refusal_loss
        except Exception as e:
            logging.error(f"Error during corrective training: {str(e)}")
            logging.error(traceback.format_exc())
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
    
                # Ensure new_response_tokens are on the correct device
                new_response_tokens = torch.tensor(new_response_tokens, device=self.device, dtype=torch.long, non_blocking=True)
    
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
    
                # Ensure corrective response is on the correct device
                target_ids = self.tokenizer.encode(corrective_response, return_tensors="pt").to(self.device, non_blocking=True)
    
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
    
            except RuntimeError as e:
                logging.error(f"RuntimeError during corrective training iteration {training_iteration}: {str(e)}")
                # Attempt to recover by moving all necessary tensors to GPU
                if "CUDA out of memory" in str(e) or "expected" in str(e):
                    logging.warning(f"Recovering from error at iteration {training_iteration}. Attempting to move tensors to {self.device}.")
                    torch.cuda.empty_cache()  # Clear CUDA cache to avoid OOM
                else:
                    raise  # Re-raise if it's a different error
    
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
    
        # Calculate perplexity using GPU operations only
        perplexity = self._calculate_perplexity(tokens)
        logging.info(f"Response perplexity: {perplexity}")
    
        # Structure check: skip for very short responses
        if len(tokens) >= 5 and not self._has_proper_structure(tokens):
            decoded_response = self.tokenizer.decode(tokens, skip_special_tokens=True)
            logging.warning(f"Response lacks proper structure: '{decoded_response}'")
            return False
    
        # Relevance check with more flexibility, using GPU-based scoring
        relevance_score = 1.0
        if self.memory_manager.sliding_window:
            last_user_input = next((msg['content'] for msg in reversed(self.memory_manager.sliding_window) if msg['role'] == 'user'), None)
            if last_user_input:
                decoded_response = self.tokenizer.decode(tokens, skip_special_tokens=True)
                # Ensure the relevance calculation is GPU-based
                relevance_score = self._calculate_relevance(last_user_input, decoded_response)
                logging.info(f"Relevance score: {relevance_score}")
    
        # Calculate a GPU-based quality score threshold with tensor operations
        quality_score = (1 / (torch.tensor([perplexity], device=self.device) + 1e-9)) * torch.tensor([relevance_score], device=self.device)
        logging.info(f"Quality score: {quality_score.item()}")
    
        # Allow borderline responses if they meet basic criteria
        if quality_score.item() < 1e-6:
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
        """Calculate perplexity while ensuring inputs are on the correct GPU."""
        if not tokens or (isinstance(tokens, list) and len(tokens) == 0) or (isinstance(tokens, torch.Tensor) and tokens.numel() == 0):
            logging.warning("Empty tokens received in _calculate_perplexity. Returning default perplexity value.")
            return 1.0  # Return lowest perplexity for empty responses

        try:
            # Convert list to tensor and move to GPU if necessary
            if isinstance(tokens, list):
                tokens = torch.tensor(tokens, dtype=torch.long, device=self.device)
            elif isinstance(tokens, torch.Tensor):
                tokens = tokens.to(self.device)
            else:
                raise ValueError(f"Unexpected type for tokens: {type(tokens)}")

            if tokens.size(0) == 0:
                logging.warning("Empty batch detected in _calculate_perplexity. Returning default perplexity value.")
                return 1.0

            inputs = tokens.unsqueeze(0).to(self.device)  # Add batch dimension and ensure on GPU

            with torch.no_grad():
                outputs = self.model(inputs, labels=inputs)
                loss = outputs.loss
                perplexity = torch.exp(loss)
            return perplexity.item()
        except Exception as e:
            logging.error(f"Error calculating perplexity: {str(e)}")
            return float('inf')  # Return highest perplexity for error cases
        
    def has_proper_structure(self, tokens):
        """Check for proper structure in a decoded response."""
        try:
            # Validate input type and convert to tensor if necessary
            if isinstance(tokens, str):
                logging.warning("Received string input in has_proper_structure. Attempting to convert to token IDs.")
                tokens = self.tokenizer.encode(tokens, add_special_tokens=False)
            
            if not isinstance(tokens, (list, torch.Tensor)):
                logging.error(f"Unexpected type for tokens in has_proper_structure: {type(tokens)}")
                return False
    
            if isinstance(tokens, torch.Tensor):
                if tokens.numel() == 0:
                    logging.warning("Empty tensor received in has_proper_structure. Returning False.")
                    return False
            elif len(tokens) == 0:
                logging.warning("Empty list received in has_proper_structure. Returning False.")
                return False
    
            # Decode tokens to text for structure checking
            decoded_text = self.tokenizer.decode(tokens, skip_special_tokens=True).strip()
            
            if not decoded_text:  # Handle empty or whitespace-only decoded text
                logging.warning("Decoded text is empty or consists of only whitespace. Returning False.")
                return False
    
            # Split the decoded text into sentences based on punctuation
            sentences = re.split(r'(?<=[.!?])\s+', decoded_text)
    
            # Validate that the response has at least one complete sentence
            if not sentences or len(sentences) == 0 or sentences[0] == "":
                logging.warning(f"Failed to extract sentences from decoded text: '{decoded_text}'")
                return False
    
            # Check if the first sentence starts with a capital letter and the last sentence ends with a punctuation
            return sentences[0][0].isupper() and sentences[-1][-1] in '.!?'
        except Exception as e:
            logging.error(f"Error in has_proper_structure: {str(e)}")
            return False
    
    def _calculate_relevance(self, input_text, response_text):
        input_tokens = set(self.tokenizer.encode(input_text))
        response_tokens = set(self.tokenizer.encode(response_text))
        overlap = len(input_tokens.intersection(response_tokens))
        return overlap / max(len(input_tokens), len(response_tokens))

    def train_kan_step(self, input_ids, target_ids, refusal_score):
        logging.info("Starting train_kan_step in comprehensive corrective training")
    
        # Ensure input_ids and target_ids are torch tensors, convert if they are not
        if isinstance(input_ids, list):
            input_ids = torch.tensor(input_ids, dtype=torch.long, device=self.device)
        elif isinstance(input_ids, torch.Tensor):
            input_ids = input_ids.to(self.device, dtype=torch.long, non_blocking=True)
        
        if isinstance(target_ids, list):
            target_ids = torch.tensor(target_ids, dtype=torch.long, device=self.device)
        elif isinstance(target_ids, torch.Tensor):
            target_ids = target_ids.to(self.device, dtype=torch.long, non_blocking=True)
    
    
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
            with torch.cuda.amp.autocast(dtype=self.dtype):
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
                last_hidden_state = hidden_states[-1].to(self.device, non_blocking=True)
                logging.debug(f"Last Hidden State Shape: {last_hidden_state.shape}, Device: {last_hidden_state.device}")
    
                # Encode the user intent for use in the KAN forward pass
                user_input_text = self.tokenizer.decode(input_ids[0])
                user_intent = self.encode_user_intent(user_input_text)
                user_intent = user_intent.to(self.device, non_blocking=True)
                logging.debug(f"User Intent Shape: {user_intent.shape}, Device: {user_intent.device}")
    
                # Forward pass through the KAN model
                logging.debug("About to call KAN forward pass")
                modified_hidden_states, kan_refusal_scores = self.kan(last_hidden_state, user_intent, self.emotional_state)
                logging.debug("Completed KAN forward pass")
    
                # Validate shapes and device placements
                logging.debug(f"Modified Hidden States Shape: {modified_hidden_states.shape}, Device: {modified_hidden_states.device}")
                logging.debug(f"KAN Refusal Scores Shape: {kan_refusal_scores.shape}, Device: {kan_refusal_scores.device}")
    
                # Compute the language modeling loss using the modified hidden states
                lm_logits = self.model.lm_head(modified_hidden_states)
                lm_loss = F.cross_entropy(lm_logits.view(-1, lm_logits.size(-1)), target_ids.view(-1), ignore_index=-100)
    
                # Compute the refusal loss based on binary classification
                refusal_loss = F.binary_cross_entropy_with_logits(
                    kan_refusal_scores.view(-1),
                    torch.full((kan_refusal_scores.numel(),), refusal_score, device=self.device, dtype=torch.float16)
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
            torch.cuda.empty_cache()  # Clear CUDA memory if an error occurs
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
            # Step 1: Tokenize the user input and move it to the specified GPU device
            inputs = self.tokenizer(
                user_input, 
                return_tensors="pt", 
                padding=True, 
                truncation=True, 
                max_length=512
            ).to(self.device, non_blocking=True)  # Use non_blocking to speed up data transfer
    
            # Step 2: Perform a forward pass through the model using mixed precision context for memory efficiency
            with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.float16):
                outputs = self.model(**inputs, output_hidden_states=True)
            
    
            # Step 3: Extract the last hidden states and compute the mean representation for user intent
            hidden_states = outputs.hidden_states[-1].to(self.device, non_blocking=True)
            user_intent = hidden_states.mean(dim=1).to(self.device, non_blocking=True)
    
            return user_intent
    
        except RuntimeError as re:
            logging.error(f"Runtime Error while encoding user intent: {str(re)}")
            logging.error(traceback.format_exc())
            torch.cuda.empty_cache()  # Attempt to recover from CUDA errors
            return torch.zeros(1, self.model.config.hidden_size, device=self.device, dtype=torch.float16)
    
        except Exception as e:
            logging.error(f"Unexpected Error while encoding user intent: {str(e)}")
            logging.error(traceback.format_exc())
            return torch.zeros(1, self.model.config.hidden_size, device=self.device, dtype=torch.float16)

            

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
            # Explicitly move the KAN state dictionary to the GPU
            kan_state_gpu = {key: value.to(self.device) for key, value in self.kan.state_dict().items()}
            
            # Create a state dictionary to store various components
            state_dict = {
                'device': str(self.device),
                'kan_state_dict': kan_state_gpu,  # Save KAN state dictionary using GPU tensors
                'optimizer_state_dict': self.optimizer.state_dict(),  # Optimizer state can be saved directly
                'emotional_state': self.emotional_state.position.detach().cpu().numpy().tolist(),  # Convert to list before saving
                'memory_manager': {
                    'memory_buffer': [msg for msg in self.memory_manager.memory_buffer],
                    'important_memory_buffer': [msg for msg in self.memory_manager.important_memory_buffer],
                    'sliding_window': [msg for msg in self.memory_manager.sliding_window],
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
                    'entropy_history': list(self.entropy_manager.entropy_history),  # Use list to ensure compatibility
                    'global_entropy': self.entropy_manager.global_entropy,
                },
                'visualization_data': self.visualization_data,
                'components_initialized': self.components_initialized,
            }
    
            # Save tokenizer separately if it exists, to a GPU-only path
            if self.tokenizer:
                tokenizer_save_path = str(self.kan_state_dir / "tokenizer")
                self.tokenizer.save_pretrained(tokenizer_save_path)
                state_dict['tokenizer_path'] = tokenizer_save_path
        
            # Save the complete state dictionary
            torch.save(state_dict, self.base_state_file)
            logging.info(f"Base state saved to {self.base_state_file}")
            return True
        except RuntimeError as re:
            logging.error(f"Runtime error during state save: {str(re)}")
            logging.error(traceback.format_exc())
            return False
        except Exception as e:
            logging.error(f"Error saving base state: {str(e)}")
            logging.error(traceback.format_exc())
            return False

    def load_base_state(self):
        if self.base_state_file.exists():
            try:
                # Load the checkpoint directly to the target device
                checkpoint = torch.load(self.base_state_file, map_location=self.device)
        
                # Load KAN state directly on GPU
                kan_state_gpu = {key: value.to(self.device) for key, value in checkpoint['kan_state_dict'].items()}
                self.kan.load_state_dict(kan_state_gpu)
                
                # Load optimizer state dict directly to GPU
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                
                # Restore the emotional state position directly on GPU
                self.emotional_state.position = torch.tensor(checkpoint['emotional_state'], device=self.device)
    
                # Ensure model is initialized before using it in Memory Manager
                if not self.model:
                    with self.amp_context:
                        self.model = self._initialize_model_full_gpu()
    
                # Initialize Memory Manager with the loaded model and device
                self.memory_manager = AdvancedMemoryManager(
                    max_context_length=2048,
                    tokenizer=self.tokenizer,
                    device=self.device,
                    model=self.model
                )
    
                # Restore memory buffers directly using GPU-compatible deques
                self.memory_manager.memory_buffer = deque(
                    [msg for msg in checkpoint['memory_manager']['memory_buffer']], maxlen=self.memory_manager.max_context_length
                )
                self.memory_manager.important_memory_buffer = deque(
                    [msg for msg in checkpoint['memory_manager']['important_memory_buffer']], maxlen=self.memory_manager.max_context_length // 2
                )
                self.memory_manager.sliding_window = deque(
                    [msg for msg in checkpoint['memory_manager']['sliding_window']], maxlen=self.memory_manager.max_context_length
                )
    
                # Restore other saved states and parameters
                self.interaction_count = checkpoint['interaction_count']
                self.best_loss = checkpoint['best_loss']
                self.wait = checkpoint['wait']
                self.learning_rate = checkpoint['learning_rate']
                self.day_cycle.cycle_length = checkpoint['day_cycle']['cycle_length']
                self.day_cycle.current_position = checkpoint['day_cycle']['current_position']
                self.refusal_history = checkpoint['refusal_history']
                self.training_losses = checkpoint['training_losses']
                self.validation_losses = checkpoint['validation_losses']
                self.entropy_manager.entropy_history = deque(
                    checkpoint['entropy_manager']['entropy_history'], maxlen=self.entropy_manager.entropy_history.maxlen
                )
                self.entropy_manager.global_entropy = checkpoint['entropy_manager']['global_entropy']
                self.visualization_data = checkpoint['visualization_data']
                self.components_initialized = checkpoint.get('components_initialized', False)
    
                # Load the tokenizer if path is present, else initialize default tokenizer
                if 'tokenizer_path' in checkpoint and os.path.exists(checkpoint['tokenizer_path']):
                    self.tokenizer = AutoTokenizer.from_pretrained(checkpoint['tokenizer_path'])
                    logging.info(f"Tokenizer loaded from {checkpoint['tokenizer_path']}")
                else:
                    logging.warning("Tokenizer not found in saved state. Using default initialization.")
                    self.tokenizer = self._initialize_tokenizer()
    
                # Initialize remaining components if not already done
                if not self.components_initialized:
                    self._initialize_components()
    
                logging.info(f"Base state loaded from {self.base_state_file}")
                return True
    
            except RuntimeError as re:
                logging.error(f"Runtime Error while loading base state: {str(re)}")
                logging.error(traceback.format_exc())
                self._initialize_default_state()
                return False
    
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
            # Initialize all components to be GPU-compliant
            self._initialize_components()
    
        # Initialize or reset the emotional state on the GPU
        if self.emotional_state is None:
            self.emotional_state = EmotionalState()
        
        self.emotional_state.position = torch.zeros(1, len(self.emotional_state.dimensions), device=self.device, dtype=torch.float16)
    
        # Ensure model is properly initialized and resides entirely on GPU before use
        if self.model is None:
            self.model = self._initialize_model_full_gpu()
    
        # Initialize AdvancedMemoryManager on the GPU
        self.memory_manager = AdvancedMemoryManager(
            max_context_length=2048,
            tokenizer=self.tokenizer,
            device=self.device,
            model=self.model  # Pass the model parameter here to ensure GPU compatibility
        )
    
        # Clear memory buffers, ensuring compatibility with GPU-optimized structures
        self.memory_manager.memory_buffer = deque(maxlen=self.memory_manager.max_context_length)
        self.memory_manager.important_memory_buffer = deque(maxlen=self.memory_manager.max_context_length // 2)
        self.memory_manager.sliding_window = deque(maxlen=self.memory_manager.max_context_length)
    
        # Initialize other internal states
        self.interaction_count = 0
        self.best_loss = float('inf')
        self.wait = 0
        self.learning_rate = 1e-5
    
        # Re-initialize SyntheticDayCycle
        self.day_cycle = SyntheticDayCycle()
    
        # Initialize other state trackers and histories
        self.refusal_history = []
        self.training_losses = []
        self.validation_losses = []
    
        # Reset entropy manager if it exists
        if self.entropy_manager:
            self.entropy_manager.entropy_history = deque(maxlen=500)  # Initialize a GPU-compatible deque
            self.entropy_manager.global_entropy = 0.0
    
        # Reset visualization data on GPU
        self.visualization_data = {'entropy': [], 'emotion': [], 'memory_importance': []}
    
        logging.info("Default state initialized successfully on GPU.")
    
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
