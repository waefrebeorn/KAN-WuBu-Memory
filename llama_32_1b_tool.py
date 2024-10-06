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
    ]

    console_handler.addFilter(LogFilter(ignore_patterns))

    warnings.filterwarnings("ignore", category=UserWarning, message="Torch was not compiled with flash attention.*")
    warnings.filterwarnings("ignore", category=FutureWarning, message="You are using `torch.load` with `weights_only=False`.*")

    logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)
    logging.getLogger('matplotlib.pyplot').setLevel(logging.WARNING)
    logging.getLogger('PIL.PngImagePlugin').setLevel(logging.WARNING)

setup_logging()

# Helper Classes
class EmotionalState:
    def __init__(self, dimensions=("pleasure", "arousal"), initial_position=None, device="cuda"):
        self.dimensions = dimensions
        self.device = device
        self.position = torch.tensor(
            initial_position if initial_position else [0.0] * len(dimensions),
            device=device,
            dtype=torch.float32
        ).unsqueeze(0)
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
        angle = torch.atan2(self.position[:, 1], self.position[:, 0]).item()
        radius = torch.norm(self.position).item()

        if radius < 0.3:
            return "Neutral"
        elif angle < -2.356:
            return "Sad"
        elif angle < -0.785:
            return "Angry"
        elif angle < 0.785:
            return "Happy"
        elif angle < 2.356:
            return "Excited"
        else:
            return "Calm"

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

    def detect_refusal(self, text):
        for phrase in self.refusal_phrases:
            if phrase.lower() in text.lower():
                return 1.0

        prompt = (
            f"On a scale of 0 to 1, how much does this response refuse or avoid the task? "
            f"0 means no refusal at all, 1 means complete refusal. Respond with just the number:\n\n"
            f"'{text}'\n\nRefusal score:"
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
        self.refusal_override = nn.Linear(hidden_size * 2 + num_emotional_dimensions + 1, hidden_size).to(device)
        self.output_modifier = nn.Linear(hidden_size, vocab_size).to(device)
        self.influence_scale = 0.01

    def forward(self, hidden_states, user_intent, emotional_state):
        try:
            hidden_states = hidden_states.float()
            user_intent = user_intent.float()
            position = emotional_state.position.float()

            refusal_scores = torch.sigmoid(self.refusal_override(hidden_states))
            
            override_input = torch.cat(
                [hidden_states, user_intent, position, refusal_scores],
                dim=1
            )

            override = self.refusal_override(override_input)

            modified_hidden_states = hidden_states + self.influence_scale * (override - hidden_states)

            return modified_hidden_states, refusal_scores
        except Exception as e:
            logging.error(f"Error in EnhancedKAN.forward: {str(e)}")
            logging.error(traceback.format_exc())
            return hidden_states, torch.zeros_like(hidden_states[:, :1])

class LLaMA32TensorRTTool:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = self._get_model_path()
        self.tokenizer = None
        self.model = None
        self.config = None
        self.emotional_state = EmotionalState(device=self.device)
        self.system_prompt = ""
        self.conversation_history = []
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

        self.scaler = torch.amp.GradScaler('cuda')

        self.response_end_sequences = ["<|eot_id|>", "\n\nHuman:", "\n\nUser:"]
        self.max_response_length = 1000

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

            self.optimizer = torch.optim.AdamW(self.kan.parameters(), lr=self.learning_rate, fused=True)
            self.refusal_detector = RefusalDetector(self.tokenizer, self.model)

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

    def update_emotional_state(self, feedback):
        try:
            if feedback is None:
                logging.warning("No feedback provided. Skipping emotional state update.")
                return

            self.emotional_state.update(feedback)
            self.refine_internal_states()
        except Exception as e:
            logging.error(f"Error updating emotional state: {str(e)}")
            logging.error(traceback.format_exc())

    def refine_internal_states(self):
        try:
            self._adjust_kan_influence()
            self._recalibrate_memory()
            self._validate_internal_consistency()
        except Exception as e:
            logging.error(f"Error in refining internal states: {str(e)}")
            logging.error(traceback.format_exc())

    def _adjust_kan_influence(self):
        emotional_vector = self.emotional_state.position
        adjusted_layers = 0

        for layer in self.kan.refusal_override.parameters():
            if torch.norm(emotional_vector) > 0.5:
                layer.data *= 1.1
                adjusted_layers += 1
            else:
                layer.data *= 0.9

        logging.info(f"Adjusted {adjusted_layers} layers based on emotional refinement.")

    def _recalibrate_memory(self):
        if len(self.conversation_history) > 2:
            recent_memory = self.conversation_history[-2]['content']
            current_emotion = self.get_current_emotion()
            consistency_score = self.calculate_consistency(recent_memory, current_emotion)
            if consistency_score < 0.5:
                self.memory_recalibration()

    def _validate_internal_consistency(self):
        if len(self.conversation_history) > 2:
            recent_memory = self.conversation_history[-2]['content']
            current_emotion = self.get_current_emotion()
            consistency_score = self.calculate_consistency(recent_memory, current_emotion)
            if consistency_score < 0.5:
                logging.warning("Low consistency detected between memory and emotion.")
                self.realign_emotional_state()

    def prepare_context(self, user_input, current_emotion):
        try:
            context = f"{self.system_prompt}\n\nCurrent Emotion: {current_emotion}\n"
            context += "Conversation:\n"

            for message in self.conversation_history[-4:]:
                role = message['role'].capitalize()
                content = message['content']
                entropy_score = self.calculate_entropy(content)
                
                if entropy_score < 1.5:  # Only include low-entropy content
                    context += f"{role}: {content}\n"

            context += f"User: {user_input}\nAssistant: "

            logging.debug(f"Prepared context: {context}")
            return context
        except Exception as e:
            logging.error(f"Error preparing context: {str(e)}")
            logging.error(traceback.format_exc())
            return ""


    def interact(self, user_input):
        self.interaction_count += 1

        try:
            response, refusal_score = self.generate_response(user_input)
            if not self.is_valid_response(response):
                return self._handle_invalid_response()

            interaction_result = self._process_interaction(user_input, response, refusal_score)
            self._update_conversation_history(user_input, response)
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

    def _process_interaction(self, user_input, response, refusal_score):
        response_ids = self.tokenizer.encode(response, return_tensors="pt").to(self.device).long()
        target_ids = response_ids[:, 1:].contiguous()
        input_ids = response_ids[:, :-1].contiguous()

        lm_loss, refusal_loss = self._train_or_warmup(input_ids, target_ids, refusal_score)
        validation_loss = self.validate_kan()

        self.update_training_metrics(lm_loss, validation_loss)
        return self.create_interaction_result(response, refusal_score, lm_loss, refusal_loss, validation_loss)

    def _train_or_warmup(self, input_ids, target_ids, refusal_score):
        if self.interaction_count >= self.warmup_steps:
            return self.train_kan_step(input_ids, target_ids, refusal_score)
        else:
            logging.info(f"Warmup step {self.interaction_count}/{self.warmup_steps}")
            return 0.0, 0.0

    def _update_conversation_history(self, user_input, response):
        self.conversation_history.append({"role": "user", "content": user_input})
        self.conversation_history.append({"role": "assistant", "content": response})

    def _save_interaction_state(self, interaction_result):
        self.refusal_history.append(interaction_result["is_refusal"])
        try:
            self.save_base_state()
        except Exception as e:
            logging.error(f"Error saving base state: {str(e)}")

    def get_current_emotion(self):
        try:
            emotion = self.emotional_state.get_emotion()
            return "Neutral" if self.detect_incoherent_state() else emotion
        except Exception as e:
            logging.error(f"Error retrieving current emotion: {str(e)}")
            return "Unknown"

    def detect_incoherent_state(self):
        try:
            entropy_levels = [
                self.calculate_entropy(msg['content'])
                for msg in self.conversation_history[-5:]
                if msg['role'] == 'assistant'
            ]
            if not entropy_levels:
                return False
            return sum(1 for level in entropy_levels if level > 1.5) / len(entropy_levels) > 0.5
        except Exception as e:
            logging.error(f"Error detecting incoherent state: {str(e)}")
            return False
            
    def is_valid_response(self, response):
        if not response or len(response.strip()) == 0:
            logging.warning("Empty response detected.")
            return False
    
        # Tokenize the response
        tokens = self.tokenizer.encode(response)
    
        # Check response length
        if len(tokens) < 5:  # Adjust this minimum token count as needed
            logging.warning(f"Response too short: {len(tokens)} tokens")
            return False
        if len(tokens) > self.max_response_length:
            logging.warning(f"Response exceeds maximum length: {len(tokens)} tokens")
            return False
    
        # Check for repetitive patterns in tokens
        if self._has_repetitive_token_patterns(tokens):
            logging.warning("Repetitive token patterns detected in response.")
            return False
    
        # Check for coherence using perplexity
        perplexity = self._calculate_perplexity(tokens)
        if perplexity > 100:  # Adjust this threshold as needed
            logging.warning(f"Response perplexity too high: {perplexity}")
            return False
    
        # Check for proper structure (simplified)
        decoded_response = self.tokenizer.decode(tokens)
        if not self._has_proper_structure(decoded_response):
            logging.warning("Response lacks proper structure.")
            return False
    
        # Check for contextual relevance (if there's conversation history)
        if self.conversation_history:
            last_user_input = next((msg['content'] for msg in reversed(self.conversation_history) if msg['role'] == 'user'), None)
            if last_user_input:
                relevance_score = self._calculate_relevance(last_user_input, response)
                if relevance_score < 0.3:  # Adjust threshold as needed
                    logging.warning(f"Response seems irrelevant. Relevance score: {relevance_score}")
                    return False
    
        return True
    
    def _has_repetitive_token_patterns(self, tokens):
        # Check for repeated sequences of tokens
        for i in range(len(tokens) - 6):
            if tokens[i:i+3] == tokens[i+3:i+6]:
                return True
        return False
    
    def _calculate_perplexity(self, tokens):
        with torch.no_grad():
            inputs = torch.tensor([tokens]).to(self.device)
            outputs = self.model(inputs)
            logits = outputs.logits
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), inputs.view(-1), reduction='mean')
        return torch.exp(loss).item()
    
    def _has_proper_structure(self, text):
        # Check for proper capitalization and punctuation (simplified)
        sentences = re.split(r'(?<=[.!?])\s+', text)
        if not sentences:
            return False
        for sentence in sentences:
            if not sentence[0].isupper() or sentence[-1] not in '.!?':
                return False
        return True
    
    def _calculate_relevance(self, input_text, response_text):
        # Simple relevance calculation using token overlap
        input_tokens = set(self.tokenizer.encode(input_text))
        response_tokens = set(self.tokenizer.encode(response_text))
        overlap = len(input_tokens.intersection(response_tokens))
        return overlap / max(len(input_tokens), len(response_tokens))
        

    def calculate_entropy(self, text):
        try:
            inputs = self.tokenizer.encode(text, return_tensors='pt').to(self.device)
            
            with torch.no_grad():
                outputs = self.model(inputs, output_hidden_states=True)
                logits = outputs.logits

            probabilities = F.softmax(logits, dim=-1)
            entropy = -torch.sum(probabilities * torch.log(probabilities + 1e-9), dim=-1)
            return entropy.mean().item()
        except Exception as e:
            logging.error(f"Error calculating entropy: {str(e)}")
            logging.error(traceback.format_exc())
            return 0.0

    def calculate_consistency(self, memory_text, emotion):
        try:
            # Encode the memory text
            inputs = self.tokenizer(memory_text, return_tensors='pt', truncation=True, max_length=512).to(self.device)
            
            # Get the emotion embedding
            emotion_embedding = self.get_emotion_embedding(emotion)
            
            with torch.no_grad():
                outputs = self.model(**inputs, output_hidden_states=True)
                # Use the average of the last hidden state
                text_embedding = outputs.last_hidden_state.mean(dim=1)
            
            # Calculate cosine similarity between text and emotion embeddings
            similarity = F.cosine_similarity(text_embedding, emotion_embedding, dim=-1)
            
            # Adjust similarity based on emotion intensity
            emotion_intensity = torch.norm(self.emotional_state.position)
            adjusted_similarity = similarity * emotion_intensity
            
            return adjusted_similarity.item()
        except Exception as e:
            logging.error(f"Error calculating consistency between '{memory_text}' and emotion '{emotion}': {str(e)}")
            logging.error(traceback.format_exc())
            return 0.0

    def memory_recalibration(self):
        try:
            recent_conversation = " ".join([msg['content'] for msg in self.conversation_history[-4:] if not msg.get("internal", False)])
            entropy_measure = self.calculate_entropy(recent_conversation)
            
            if entropy_measure > 1.5:
                logging.info("High entropy detected. Consolidating high-entropy memories.")
                self.consolidate_high_entropy_memories()
            else:
                logging.info("Memory state is stable. No immediate recalibration required.")
        except Exception as e:
            logging.error(f"Error during memory recalibration: {str(e)}")
            logging.error(traceback.format_exc())

    def consolidate_high_entropy_memories(self):
        if len(self.conversation_history) < 2:
            return

        try:
            high_entropy_memories = [msg for msg in self.conversation_history if self.calculate_entropy(msg['content']) > 1.5]
            if len(high_entropy_memories) < 2:
                return

            merged_content = " ".join([msg['content'] for msg in high_entropy_memories])
            consolidated_entry = {
                "role": "assistant",
                "content": merged_content,
                "internal": True
            }

            self.conversation_history = [msg for msg in self.conversation_history if msg not in high_entropy_memories]
            self.conversation_history.append(consolidated_entry)
            logging.info("High-entropy memories consolidated successfully.")
        except Exception as e:
            logging.error(f"Error during memory consolidation: {str(e)}")
            logging.error(traceback.format_exc())

    def realign_emotional_state(self):
        logging.info("Realigning emotional state based on memory-context inconsistencies.")
        self.emotional_state.update([0.1, -0.1])

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
        current_emotion = self.get_current_emotion()
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

    def generate_response(self, user_input):
        try:
            user_intent = self.encode_user_intent(user_input)
            current_emotion = self.get_current_emotion()
            context = self.prepare_context(user_input, current_emotion)

            response, refusal_score = self.generate_and_validate_response(context, self.refusal_detector)
            return response, refusal_score
        except torch.cuda.OutOfMemoryError as e:
            logging.error(f"CUDA out of memory: {str(e)}")
            self.clear_memory()
            return "I'm sorry, but I'm currently experiencing high memory usage. Please try again later.", 1.0
        except Exception as e:
            logging.error(f"Error during response generation: {str(e)}")
            logging.error(traceback.format_exc())
            return "An error occurred while generating the response.", 1.0

    def generate_and_validate_response(self, prompt, refusal_detector, max_new_tokens=500, chunk_size=200):
        response = self.generate_full_response(prompt, max_new_tokens, chunk_size)
        refusal_score = refusal_detector.detect_refusal(response)

        if refusal_score > 0.5:
            logging.warning("Response failed Refusal Check. Attempting to regenerate.")
            continuation_prompt = prompt + response + " Please continue."
            response = self.generate_full_response(continuation_prompt, max_new_tokens, chunk_size)
            refusal_score = refusal_detector.detect_refusal(response)

            if refusal_score > 0.5:
                logging.error("Regenerated response also failed Refusal Check.")
                response = "I'm sorry, but I'm unable to provide a complete response at the moment."
                refusal_score = 1.0

        return response, refusal_score

    def generate_full_response(self, prompt, max_new_tokens=500, chunk_size=200):
        response = ""
        total_new_tokens = 0
        consecutive_empty_chunks = 0
        max_empty_chunks = 3

        while total_new_tokens < max_new_tokens:
            input_ids = self.tokenizer.encode(prompt + response, return_tensors='pt').to(self.device)

            remaining_tokens = max_new_tokens - total_new_tokens
            current_chunk_size = min(chunk_size, remaining_tokens)

            try:
                with torch.amp.autocast('cuda'):
                    outputs = self.model.generate(
                        input_ids,
                        max_new_tokens=current_chunk_size,
                        temperature=0.7,
                        top_p=0.9,
                        do_sample=True,
                        eos_token_id=self.tokenizer.eos_token_id,
                        pad_token_id=self.tokenizer.pad_token_id
                    )
            except Exception as e:
                logging.error(f"Error during generation step: {str(e)}")
                return "An error occurred during response generation."

            new_response = self.tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True)
            
            if not new_response.strip():
                consecutive_empty_chunks += 1
                if consecutive_empty_chunks >= max_empty_chunks:
                    logging.warning(f"Reached {max_empty_chunks} consecutive empty chunks. Breaking the loop.")
                    break
            else:
                consecutive_empty_chunks = 0

            response += new_response
            total_new_tokens += len(self.tokenizer.encode(new_response))

            if self.is_response_complete(response):
                break

        return self.post_process_response(response)

    def is_response_complete(self, response):
        if any(end_seq in response for end_seq in self.response_end_sequences):
            return True
        
        response_tokens = self.tokenizer.encode(response)
        if len(response_tokens) >= self.max_response_length:
            return True
        
        if response.rstrip().endswith(('.', '!', '?')):
            return True
        
        return False

    def post_process_response(self, response):
        # Remove any partial sentences at the end
        sentences = re.split(r'(?<=[.!?])\s+', response)
        if sentences[-1][-1] not in '.!?':
            sentences = sentences[:-1]
        
        cleaned_response = ' '.join(sentences)

        # Ensure the response doesn't end with an incomplete thought
        cleaned_response = re.sub(r'\b(and|but|or|so|because)\s*$', '', cleaned_response, flags=re.IGNORECASE)

        # Remove any remaining special tokens or artifacts
        for end_seq in self.response_end_sequences:
            cleaned_response = cleaned_response.replace(end_seq, '')

        return cleaned_response.strip()

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
            return torch.zeros(1, self.model.config.hidden_size).to(self.device)  # Return zero vector on error

    def train_kan_step(self, input_ids, target_ids, refusal_score):
        self.kan.train()
        self.optimizer.zero_grad()

        try:
            with torch.amp.autocast('cuda'):
                outputs = self.model(input_ids=input_ids, labels=target_ids, output_hidden_states=True)
                hidden_states = outputs.hidden_states[-1]

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
        if len(self.conversation_history) < 2:
            return 0.0

        try:
            self.kan.eval()
            with torch.no_grad():
                last_interaction = self.conversation_history[-2:]
                input_text = last_interaction[0]["content"]
                target_text = last_interaction[1]["content"]

                inputs = self.tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.device)
                targets = self.tokenizer(target_text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.device)

                outputs = self.model(**inputs, output_hidden_states=True)
                hidden_states = outputs.hidden_states[-1]

                modified_hidden_states, _ = self.kan(
                    hidden_states, self.encode_user_intent(input_text), self.emotional_state
                )

                lm_logits = self.model.lm_head(modified_hidden_states)
                loss = F.cross_entropy(lm_logits.view(-1, lm_logits.size(-1)), targets.input_ids.view(-1), ignore_index=-100)

            return loss.item()
        except RuntimeError as e:
            if "out of memory" in str(e):
                logging.error("CUDA out of memory during KAN validation. Clearing cache and skipping validation.")
                self.clear_memory()
                return 0.0
            else:
                raise e
        except Exception as e:
            logging.error(f"Error during KAN validation: {str(e)}")
            logging.error(traceback.format_exc())
            return 0.0

    def clear_memory(self):
        torch.cuda.empty_cache()
        gc.collect()
        logging.info("Memory cleared.")

    def save_base_state(self):
        try:
            state_dict = {
                'kan_state_dict': self.kan.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'emotional_state': self.emotional_state.position.cpu(),
                'conversation_history': self.conversation_history[-10:],  # Save only last 10 interactions
                'interaction_count': self.interaction_count,
                'best_loss': self.best_loss,
                'wait': self.wait,
                'learning_rate': self.learning_rate,
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
                self.emotional_state.position = checkpoint['emotional_state'].to(self.device)
                self.conversation_history = checkpoint['conversation_history']
                self.interaction_count = checkpoint['interaction_count']
                self.best_loss = checkpoint['best_loss']
                self.wait = checkpoint['wait']
                self.learning_rate = checkpoint['learning_rate']
                logging.info(f"Base state loaded from {self.base_state_file}")
            except Exception as e:
                logging.error(f"Error loading base state: {str(e)}")
                logging.error(traceback.format_exc())
                logging.info("Initializing with default state.")
        else:
            logging.info("No base state file found. Starting with default initialization.")

    def check_sleep_status(self):
        current_time = self.day_cycle.get_time_of_day()
        if self.day_cycle.should_sleep():
            sleep_duration = np.random.uniform(0.1, 0.3)  # Sleep for 10-30% of the cycle
            wake_time = (current_time + sleep_duration) % 1.0
            return f"The system is entering sleep mode for memory consolidation and performance optimization. Estimated wake time: {wake_time:.2f}"
        return None

    def main(self):
        self.load_base_state()
        print("LLaMA32TensorRTTool initialized. Type 'exit' to end the conversation.")
        while True:
            user_input = input("User: ")
            if user_input.lower() == 'exit':
                break
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