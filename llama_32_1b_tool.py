import torch
import torch.nn as nn
import torch.nn.functional as F
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

    file_handler = logging.FileHandler('llama_tool.log', mode='a')
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)

    console_handler = logging.StreamHandler()
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
        "1Torch was not compiled with flash attention."
    ]

    console_handler.addFilter(LogFilter(ignore_patterns))

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
        self.position = torch.zeros(1, len(dimensions), device=device, dtype=torch.float32)
        self.velocity = torch.zeros(1, len(dimensions), device=device, dtype=torch.float32)

    def update(self, feedback, max_speed=0.1):
        feedback_vector = torch.as_tensor(feedback, device=self.device, dtype=torch.float32)
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

    def get_embedding(self):
        return self.position

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
        dtype = torch.float16 

        self.refusal_override = nn.Linear(hidden_size * 2 + num_emotional_dimensions + 1, hidden_size).to(device)
        self.output_modifier = nn.Linear(hidden_size, vocab_size).to(device)
        self.influence_scale = 0.01

    def forward(self, hidden_states, user_intent, emotional_state):
        try:
            if self.training:
                hidden_states = hidden_states.requires_grad_()

            hidden_states = hidden_states.to(self.device).to(torch.float16)
            user_intent = user_intent.to(self.device).to(torch.float16)
            position = emotional_state.get_embedding().to(self.device).to(torch.float16)

            if hidden_states.dim() == 3:
                batch_size, seq_length, hidden_dim = hidden_states.shape
                hidden_states = hidden_states.view(batch_size * seq_length, hidden_dim)
            else:
                batch_size, hidden_dim = hidden_states.shape

            if user_intent.shape[0] != hidden_states.shape[0]:
                user_intent = user_intent.expand(hidden_states.shape[0], -1)
            if position.shape[0] != hidden_states.shape[0]:
                position = position.expand(hidden_states.shape[0], -1)

            refusal_scores = torch.sigmoid(self.refusal_override(hidden_states))

            override_input = torch.cat([hidden_states, user_intent, position, refusal_scores], dim=1).to(hidden_states.dtype)

            override = self.refusal_override(override_input)

            modified_hidden_states = hidden_states + self.influence_scale * (override - hidden_states)

            return modified_hidden_states, refusal_scores
        except Exception as e:
            logging.error(f"Error in EnhancedKAN.forward: {str(e)}")
            logging.error(traceback.format_exc())
            return hidden_states, torch.zeros_like(hidden_states[:, :1])

class EntropyManager:
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.entropy_history = deque(maxlen=100)
        self.global_entropy = 0.0
        self.low_entropy_threshold = 0.5
        self.high_entropy_threshold = 2.0

    def calculate_entropy(self, logits):
        probs = F.softmax(logits, dim=-1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-9), dim=-1)
        return entropy.mean().item()

    def update_global_entropy(self, local_entropy):
        self.entropy_history.append(local_entropy)
        self.global_entropy = np.mean(self.entropy_history)

    def should_trigger_cot(self, local_entropy):
        entropy_increase = local_entropy - self.global_entropy
        probability = 1 / (1 + np.exp(-entropy_increase))  # Sigmoid function
        return np.random.random() < probability

    def adjust_sampling_parameters(self, local_entropy):
        entropy_ratio = local_entropy / (self.global_entropy + 1e-9)
        temperature = max(0.1, min(1.5, entropy_ratio))
        top_p = max(0.1, min(0.9, 1.0 - entropy_ratio))
        return {"temperature": temperature, "top_p": top_p}

class AdvancedMemoryManager:
    def __init__(self, max_context_length, tokenizer, device):
        self.max_context_length = max_context_length
        self.tokenizer = tokenizer
        self.device = device
        self.memory_buffer = deque(maxlen=max_context_length)
        self.important_memory_buffer = deque(maxlen=max_context_length // 2)
        self.sliding_window = deque(maxlen=max_context_length)
        self.tfidf_vectorizer = TfidfVectorizer()
        self.kmeans = KMeans(n_clusters=5)

    def update_memory(self, message, entropy):
        tokens = self.tokenizer.encode(message['content'])
        self.memory_buffer.extend(tokens)
        self.sliding_window.append(message)
        
        importance_score = self.calculate_importance_score(message, entropy)
        
        if importance_score > 0.7:
            self.important_memory_buffer.append(message)
        
        if len(self.sliding_window) % 10 == 0:
            self.consolidate_memories()

    def calculate_importance_score(self, message, entropy):
        memory_vector = self.tfidf_vectorizer.fit_transform([message['content']])
        recent_context = ' '.join([m['content'] for m in self.sliding_window])
        context_vector = self.tfidf_vectorizer.transform([recent_context])
        
        similarity_score = cosine_similarity(memory_vector, context_vector)[0][0]
        entropy_factor = np.tanh(entropy)
        
        return similarity_score * (1 + entropy_factor)

    def consolidate_memories(self):
        memories = [m['content'] for m in self.sliding_window]
        vectors = self.tfidf_vectorizer.fit_transform(memories)
        clusters = self.kmeans.fit_predict(vectors)
        
        consolidated_memories = []
        for cluster_id in range(self.kmeans.n_clusters):
            cluster_memories = [mem for mem, clust in zip(self.sliding_window, clusters) if clust == cluster_id]
            summary = self.summarize_cluster(cluster_memories)
            consolidated_memories.append(summary)
        
        self.sliding_window = deque(consolidated_memories, maxlen=self.max_context_length)

    def summarize_cluster(self, cluster_memories):
        cluster_text = " ".join([mem['content'] for mem in cluster_memories])
        summary_prompt = f"Summarize the following text concisely:\n\n{cluster_text}\n\nSummary:"
        summary_ids = self.tokenizer.encode(summary_prompt, return_tensors='pt').to(self.device)
        summary_output = self.model.generate(summary_ids, max_length=100, num_return_sequences=1, no_repeat_ngram_size=2)
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
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = self._get_model_path()
        self.tokenizer = None
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
        self.warmup_steps = 10
        self.kan_state_dir = Path("kan_states")
        self.kan_state_dir.mkdir(exist_ok=True)
        self.base_state_file = self.kan_state_dir / "base_state.pt"

        self.refusal_history = []
        self.interaction_results = []
        self.training_losses = []
        self.validation_losses = []
        self.patience = 5
        self.best_loss = float('inf')
        self.wait = 0

        self.overfit_detector = OverfitDetector()
        self.day_cycle = SyntheticDayCycle()

        self.scaler = GradScaler()

        self.response_end_sequences = ["<|eot_id|>", "\n\nHuman:", "\n\nUser:"]
        self.max_response_length = 1000

        self.entropy_manager = None
        self.memory_manager = None

        self.visualization_data = {
            'entropy': [],
            'emotion': [],
            'memory_importance': []
        }

        self._initialize_components()

    def _get_model_path(self):
        script_dir = Path(__file__).parent
        model_dir = script_dir / "models" / "Llama_32_1B"
        if not model_dir.exists():
            raise FileNotFoundError(f"Model directory not found: {model_dir}")
        return model_dir

    def _initialize_components(self):
        try:
            self.config = AutoConfig.from_pretrained(self.model_path)
            hidden_size = self.config.hidden_size
            num_emotional_dimensions = len(self.emotional_state.dimensions)

            self.tokenizer = self._initialize_tokenizer()
            self.model = self._initialize_model()
            
            vocab_size = len(self.tokenizer)
            self.kan = EnhancedKAN(hidden_size, num_emotional_dimensions, vocab_size, self.device).to(self.device)

            self.optimizer = torch.optim.AdamW(self.kan.parameters(), lr=self.learning_rate)
            self.refusal_detector = RefusalDetector(self.tokenizer, self.model)
            self.entropy_manager = EntropyManager(self.model, self.tokenizer, self.device)
            self.memory_manager = AdvancedMemoryManager(2048, self.tokenizer, self.device)

            self.clear_memory()
            logging.info("Components initialized successfully.")
        except Exception as e:
            logging.error(f"Error initializing components: {str(e)}")
            logging.error(traceback.format_exc())
            raise RuntimeError("Failed to initialize components.")

    def _initialize_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            use_fast=True,
            trust_remote_code=True,
        )
        self._ensure_special_tokens(tokenizer)
        return tokenizer

    def _initialize_model(self):
        with init_empty_weights():
            model = AutoModelForCausalLM.from_config(self.config)
        model.config.use_cache = False
    
        model.tie_weights()
    
        model = load_checkpoint_and_dispatch(
            model,
            self.model_path,
            device_map="auto",
            no_split_module_classes=["LlamaDecoderLayer"],
            dtype=torch.float16
        )
    
        model.gradient_checkpointing_enable()
    
        model.resize_token_embeddings(len(self.tokenizer))
    
        logging.debug(f"Model loaded on device: {self.device}")
        logging.debug(f"Tokenizer vocab size: {len(self.tokenizer)}")
        logging.debug(f"Model vocab size: {model.config.vocab_size}")
    
        for param in model.parameters():
            param.requires_grad = True
    
        return model

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

    def analyze_sentiment(self, text):
        positive_words = set(['good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic'])
        negative_words = set(['bad', 'terrible', 'awful', 'horrible', 'disappointing', 'poor'])
        
        words = text.lower().split()
        sentiment = sum(1 for word in words if word in positive_words) - sum(1 for word in words if word in negative_words)
        return sentiment / len(words)

    def generate_response(self, user_input):
        try:
            current_emotion = self.emotional_state.get_emotion()
            context = self.memory_manager.get_context(current_emotion)
            prompt = f"{context}\nUser: {user_input}\nCowgirl assistant:"
    
            input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)
            
            response_tokens = []
            total_entropy = 0
            
            for _ in range(100):  # Maximum response length
                with torch.no_grad():
                    outputs = self.model(input_ids=input_ids)
                    next_token_logits = outputs.logits[:, -1, :]
                    
                    local_entropy = self.entropy_manager.calculate_entropy(next_token_logits)
                    total_entropy += local_entropy
                    
                    if self.entropy_manager.should_trigger_cot(local_entropy):
                        cot_response = self.trigger_chain_of_thought(prompt + self.tokenizer.decode(response_tokens))
                        response_tokens.extend(self.tokenizer.encode(cot_response))
                        break
                    
                    sampling_params = self.entropy_manager.adjust_sampling_parameters(local_entropy)
                    next_token = self.sample_next_token(next_token_logits, **sampling_params)
                    
                    if next_token.item() == self.tokenizer.encode('<|eot_id|>')[0]:
                        break
                    
                    response_tokens.append(next_token.item())
                    input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=-1)
    
            response = self.tokenizer.decode(response_tokens, skip_special_tokens=True)
            avg_entropy = total_entropy / len(response_tokens)
            
            logging.debug(f"Generated response: {response}")
            
            return response_tokens, avg_entropy
        except Exception as e:
            logging.error(f"Error during response generation: {str(e)}")
            return self.tokenizer.encode("An error occurred while generating the response.<|eot_id|>"), 0.0
    
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
        return torch.multinomial(probs, num_samples=1)

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
        try:
            with torch.no_grad():
                inputs = torch.tensor([tokens]).to(self.device)
                outputs = self.model(inputs)
                logits = outputs.logits[:, :-1, :].contiguous()
                target = inputs[:, 1:].contiguous()
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target.view(-1), reduction='mean')
            return torch.exp(loss).item()
        except Exception as e:
            logging.error(f"Error calculating perplexity: {str(e)}")
            return float('inf')
    
    def _has_proper_structure(self, tokens):
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
        self.kan.train()
        self.optimizer.zero_grad()
    
        try:
            with torch.amp.autocast('cuda'):
                outputs = self.model(input_ids=input_ids, labels=target_ids, output_hidden_states=True)
                hidden_states = outputs.hidden_states[-1]
    
                if self.training:
                    hidden_states = hidden_states.requires_grad_()
    
                modified_hidden_states, kan_refusal_scores = self.kan(
                    hidden_states, self.encode_user_intent(self.tokenizer.decode(input_ids[0])), self.emotional_state
                )
    
                lm_logits = self.model.lm_head(modified_hidden_states)
    
                lm_loss = F.cross_entropy(lm_logits.view(-1, lm_logits.size(-1)), target_ids.view(-1), ignore_index=-100)
    
                refusal_loss = F.binary_cross_entropy_with_logits(
                    kan_refusal_scores.squeeze(), torch.tensor([refusal_score], device=self.device)
                )
    
                loss = lm_loss + self.kan_loss_weight * refusal_loss
    
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
    
            return lm_loss.item(), refusal_loss.item()
        except RuntimeError as e:
            if "out of memory" in str(e):
                logging.error("CUDA out of memory during KAN training. Clearing cache and skipping step.")
                self.clear_memory()
                return 0.0, 0.0
            else:
                raise e
        except Exception as e:
            logging.error(f"Error during KAN training step: {str(e)}")
            logging.error(traceback.format_exc())
            return 0.0, 0.0
    
    def validate_kan(self):
        if len(self.memory_manager.sliding_window) < 2:
            return 0.0
    
        try:
            self.kan.eval()
            with torch.no_grad():
                last_interaction = list(self.memory_manager.sliding_window)[-2:]
                input_text = last_interaction[0]["content"]
                target_text = last_interaction[1]["content"]
    
                inputs = self.tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.device)
                targets = self.tokenizer(target_text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.device)
    
                outputs = self.model(**inputs, output_hidden_states=True)
                hidden_states = outputs.hidden_states[-1]
    
                modified_hidden_states, _ = self.kan(hidden_states, self.encode_user_intent(input_text), self.emotional_state)
                lm_logits = self.model.lm_head(modified_hidden_states)
    
                min_length = min(lm_logits.size(1), targets.input_ids.size(1))
                lm_logits = lm_logits[:, :min_length, :]
                targets_flattened = targets.input_ids[:, :min_length].contiguous().view(-1)
    
                loss = F.cross_entropy(lm_logits.view(-1, lm_logits.size(-1)), targets_flattened, ignore_index=-100)
                return loss.item()
        except Exception as e:
            logging.error(f"Error during KAN validation: {str(e)}")
            logging.error(traceback.format_exc())
            return 0.0

    def encode_user_intent(self, user_input):
        try:
            inputs = self.tokenizer(
                user_input,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs, output_hidden_states=True)
                last_hidden_state = outputs.hidden_states[-1]
                user_intent = last_hidden_state.mean(dim=1)

            return user_intent
        except Exception as e:
            logging.error(f"Failed to encode user input: {str(e)}")
            return torch.zeros(1, self.model.config.hidden_size).to(self.device)

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
        self.interaction_count += 1
    
        try:
            self.memory_manager.update_memory({"role": "user", "content": user_input}, 0.0)
            response_tokens, response_entropy = self.generate_response(user_input)
            
            if not self.is_valid_response(response_tokens):
                return self._handle_invalid_response()
    
            response = self.tokenizer.decode(response_tokens, skip_special_tokens=True)
            refusal_score = self.refusal_detector.detect_refusal(response_tokens)
            
            self.update_emotional_state(response, response_entropy)
            self.memory_manager.update_memory({"role": "assistant", "content": response}, response_entropy)
            
            lm_loss, refusal_loss = self._train_or_warmup(response_tokens[:-1], response_tokens[1:], refusal_score)
            validation_loss = self.validate_kan()
    
            self.update_training_metrics(lm_loss, validation_loss)
            
            interaction_result = self.create_interaction_result(response, refusal_score, lm_loss, refusal_loss, validation_loss)
            self._save_interaction_state(interaction_result)
    
            return interaction_result
        except Exception as e:
            logging.error(f"Error in interaction process: {str(e)}")
            logging.error(traceback.format_exc())
            return {"response": "An error occurred during the interaction process.", "is_refusal": True}

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
                'refusal_history': self.refusal_history[-100:],
                'training_losses': self.training_losses[-100:],
                'validation_losses': self.validation_losses[-100:],
                'entropy_manager': {
                    'entropy_history': list(self.entropy_manager.entropy_history),
                    'global_entropy': self.entropy_manager.global_entropy,
                },
                'visualization_data': self.visualization_data,
            }
            torch.save(state_dict, self.base_state_file)
            logging.info(f"Base state saved to {self.base_state_file}")
        except Exception as e:
            logging.error(f"Error saving base state: {str(e)}")
            logging.error(traceback.format_exc())

    def load_base_state(self):
        if self.base_state_file.exists():
            try:
                checkpoint = torch.load(self.base_state_file, map_location=self.device)
                self.kan.load_state_dict(checkpoint['kan_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.emotional_state.position = torch.tensor(checkpoint['emotional_state'], device=self.device)
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
                logging.info(f"Base state loaded from {self.base_state_file}")
                return True
            except Exception as e:
                logging.error(f"Error loading base state: {str(e)}")
                logging.error(traceback.format_exc())
                logging.info("Initializing with default state.")
                self._initialize_default_state()
                return False
        else:
            logging.info("No base state file found. Starting with default initialization.")
            self._initialize_default_state()
            return False

    def _initialize_default_state(self):
        self.emotional_state.position = torch.zeros(1, len(self.emotional_state.dimensions), device=self.device)
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