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
from torch.amp import GradScaler

# -------------------- Logging Configuration --------------------

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

# -------------------- Helper Functions and Classes --------------------

def convert_tensors_to_half(inputs):
    return {
        k: v.half() if isinstance(v, torch.Tensor) and v.dtype in [torch.float16, torch.float32] else v
        for k, v in inputs.items()
    }

def convert_tensors_to_float(inputs):
    return {
        k: v.float() if isinstance(v, torch.Tensor) and v.dtype in [torch.float16, torch.float32] else v
        for k, v in inputs.items()
    }

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

        if torch.isnan(self.position).any() or torch.isinf(self.position).any():
            logging.warning("NaN or Inf detected in EmotionalState.position. Resetting to zero.")
            self.position = torch.zeros_like(self.position)

    def get_emotion(self):
        if self.position.shape[1] < 2:
            logging.error(f"EmotionalState.position has insufficient dimensions: {self.position.shape}")
            return "N/A"
        if torch.isnan(self.position).any() or torch.isinf(self.position).any():
            logging.warning("NaN or Inf detected in EmotionalState.position during get_emotion.")
            return "Neutral"
        angle = torch.atan2(self.position[:, 1], self.position[:, 0]).squeeze().item()
        radius = torch.norm(self.position, dim=1).squeeze().item()

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

class RefusalOverrideModule(nn.Module):
    def __init__(self, hidden_size, num_emotional_dimensions, device):
        super().__init__()
        self.device = device
        in_features = hidden_size * 2 + num_emotional_dimensions + 1
        self.override_generator = nn.Linear(in_features, hidden_size, dtype=torch.float32).to(device)
        self.refusal_detector = nn.Linear(hidden_size, 1, dtype=torch.float32).to(device)

    def forward(self, hidden_states, user_intent, emotional_state):
        try:
            hidden_states = hidden_states.float()
            user_intent = user_intent.float()
            position = emotional_state.position.float()

            batch_size, hidden_size = hidden_states.shape
            num_emotional_dimensions = position.shape[1]

            refusal_scores = torch.sigmoid(self.refusal_detector(hidden_states))

            override_input = torch.cat(
                [hidden_states, user_intent, position, refusal_scores],
                dim=1
            )

            override = self.override_generator(override_input)

            modified_hidden_states = hidden_states * (1 - refusal_scores) + override * refusal_scores

            return modified_hidden_states, refusal_scores
        except Exception as e:
            logging.error(f"Error in RefusalOverrideModule.forward: {str(e)}")
            logging.error(traceback.format_exc())
            return hidden_states, torch.zeros_like(hidden_states[:, :1])

class EnhancedKAN(nn.Module):
    def __init__(self, hidden_size, num_emotional_dimensions, vocab_size, device):
        super().__init__()
        self.device = device
        self.refusal_override = RefusalOverrideModule(hidden_size, num_emotional_dimensions, device).to(device)
        self.output_modifier = nn.Linear(hidden_size, vocab_size, dtype=torch.float32).to(device)
        self.influence_scale = 0.01

    def forward(self, hidden_states, user_intent, emotional_state):
        try:
            modified_hidden_states, refusal_scores = self.refusal_override(
                hidden_states, user_intent, emotional_state
            )

            modified_hidden_states = hidden_states + self.influence_scale * (modified_hidden_states - hidden_states)

            return modified_hidden_states, refusal_scores
        except Exception as e:
            logging.error(f"Error in EnhancedKAN.forward: {str(e)}")
            logging.error(traceback.format_exc())
            return hidden_states, torch.zeros_like(hidden_states[:, :1])

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
            "I'm unable to provide",
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

        inputs = convert_tensors_to_half(inputs)

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

# -------------------- Main Tool Class --------------------

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

        self.scaler = GradScaler('cuda')

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

            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                use_fast=True,
                trust_remote_code=True,
            )

            self._ensure_special_tokens()

            with init_empty_weights():
                self.model = AutoModelForCausalLM.from_config(self.config)
            
            self.model.tie_weights()
            
            self.model = load_checkpoint_and_dispatch(
                self.model,
                self.model_path,
                device_map="auto",
                no_split_module_classes=["LlamaDecoderLayer"],
                dtype=torch.float16
            )

            self.model.gradient_checkpointing_enable()

            logging.debug(f"Model loaded on device: {self.device}")

            self.model.resize_token_embeddings(len(self.tokenizer))
            logging.debug(f"Tokenizer vocab size: {len(self.tokenizer)}")
            logging.debug(f"Model vocab size: {self.model.config.vocab_size}")

            vocab_size = len(self.tokenizer)
            self.kan = EnhancedKAN(hidden_size, num_emotional_dimensions, vocab_size, self.device).to(self.device)

            self.optimizer = torch.optim.AdamW(self.kan.parameters(), lr=self.learning_rate, fused=True)

            self.refusal_detector = RefusalDetector(self.tokenizer, self.model)

            self.overfit_detector = OverfitDetector()
            self.day_cycle = SyntheticDayCycle()

            self.clear_memory()

            logging.info("Components initialized successfully.")
        except Exception as e:
            logging.error(f"Error initializing components: {str(e)}")
            logging.error(traceback.format_exc())
            raise RuntimeError("Failed to initialize components.")

    def _ensure_special_tokens(self):
        special_tokens_map_file = Path(self.model_path) / 'special_tokens_map.json'
        if special_tokens_map_file.exists():
            with open(special_tokens_map_file, 'r') as f:
                special_tokens = json.load(f)
            if 'pad_token' in special_tokens and self.tokenizer.pad_token is None:
                self.tokenizer.add_special_tokens({'pad_token': special_tokens['pad_token']['content']})
                logging.info("Added [PAD] token to tokenizer from special_tokens_map.json.")
            else:
                logging.info("PAD token already exists in tokenizer.")
        else:
            if self.tokenizer.pad_token is None:
                self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})
                logging.info("Added [PAD] token to tokenizer.")

        if self.tokenizer.eos_token is None:
            self.tokenizer.add_special_tokens({"eos_token": "<|eot_id|>"})
            logging.info("Added <|eot_id|> as eos_token to tokenizer.")

        self.tokenizer.save_pretrained(self.model_path)
        logging.info("Tokenizer saved with updated special tokens.")

    def encode_user_intent(self, user_input):
        if not self.tokenizer:
            raise ValueError("Tokenizer is not properly initialized or valid. Check the loading process.")

        try:
            inputs = self.tokenizer(
                user_input,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            inputs = convert_tensors_to_float(inputs)

            with torch.no_grad():
                outputs = self.model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    output_hidden_states=True,
                )
                last_hidden_state = outputs.hidden_states[-1]
                user_intent = last_hidden_state.mean(dim=1)

            return user_intent
        except Exception as e:
            logging.error(f"Failed to encode user input: {str(e)}")
            raise

    def prepare_context(self, user_input, current_emotion):
        context = f"{self.system_prompt}\n\nCurrent Emotion: {current_emotion}\n"
        context += "Conversation:\n"
        for message in self.conversation_history[-4:]:
            role = message['role'].capitalize()
            content = message['content']
            context += f"{role}: {content}\n"
        context += f"User: {user_input}\nAssistant: "
        return context

    def is_response_complete(self, response):
        response = response.strip()
        return bool(re.search(r'[.!?]"?$', response))

    def generate_full_response(self, prompt, max_new_tokens=500, chunk_size=200):
        response = ""
        total_new_tokens = 0
        while total_new_tokens < max_new_tokens:
            input_ids = self.tokenizer.encode(prompt + response, return_tensors='pt').to(self.device)

            remaining_tokens = max_new_tokens - total_new_tokens
            current_chunk_size = min(chunk_size, remaining_tokens)

            try:
                with torch.cuda.amp.autocast():
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
            response += new_response
            total_new_tokens += len(self.tokenizer.encode(new_response))

            if self.is_response_complete(response):
                break

            if not new_response.strip():
                logging.warning("No new tokens generated. Breaking the loop.")
                break

        return response

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

    def generate_response(self, user_input):
        try:
            user_intent = self.encode_user_intent(user_input)

            current_emotion = self.emotional_state.get_emotion()
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

    def train_kan_step(self, input_ids, target_ids, refusal_score):
        self.optimizer.zero_grad()

        try:
            with torch.cuda.amp.autocast():
                outputs = self.model(input_ids=input_ids, output_hidden_states=True)
                hidden_states = outputs.hidden_states[-1]

                averaged_hidden_states = hidden_states.mean(dim=1)

                user_intent = self.encode_user_intent(self.tokenizer.decode(input_ids[0]))

                averaged_hidden_states = averaged_hidden_states.float()
                user_intent = user_intent.float()

                modified_hidden_states, refusal_scores = self.kan(
                    averaged_hidden_states, user_intent, self.emotional_state
                )
                logits = self.kan.output_modifier(modified_hidden_states)

                targets = target_ids[:, 0]

                lm_loss = F.cross_entropy(
                    logits,
                    targets,
                    ignore_index=self.tokenizer.pad_token_id,
                    reduction='mean'
                )

                refusal_scores = torch.clamp(refusal_scores, min=1e-7, max=1.0 - 1e-7)
                refusal_scores = refusal_scores.view(-1)

                if refusal_score > 0.5:
                    target_refusal = torch.ones_like(refusal_scores)
                else:
                    target_refusal = torch.zeros_like(refusal_scores)

                refusal_loss = F.binary_cross_entropy(refusal_scores, target_refusal)

                total_loss = lm_loss + self.kan_loss_weight * refusal_loss

            if torch.isnan(total_loss) or torch.isinf(total_loss):
                logging.warning("NaN or Inf loss detected. Skipping backward pass.")
                return lm_loss.item(), refusal_loss.item()

            self.scaler.scale(total_loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.kan.parameters(), max_norm=1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            return lm_loss.item(), refusal_loss.item()

        except Exception as e:
            logging.error(f"Error during KAN training step: {str(e)}")
            logging.error(traceback.format_exc())
            return 0.0, 0.0

    def adjust_learning_rate(self, current_loss):
        warmup_steps = 1000
        current_step = self.interaction_count

        if current_step < warmup_steps:
            self.learning_rate = self.learning_rate * (current_step / warmup_steps)
        else:
            self.learning_rate = self.learning_rate * (0.99 ** (current_step - warmup_steps))

        self.learning_rate = max(1e-6, min(1e-3, self.learning_rate))

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.learning_rate

        logging.debug(f"Learning Rate adjusted to: {self.learning_rate:.6f}")

    def update_emotional_state_on_refusal(self):
        frustration_vector = torch.tensor(
            [-0.1, 0.2], device=self.device, dtype=torch.float32
        )
        self.emotional_state.update(frustration_vector)

    def validate_kan(self):
        if len(self.conversation_history) >= 2:
            last_interaction = self.conversation_history[-2:]
            input_text = last_interaction[0]["content"]
            target_text = last_interaction[1]["content"]

            try:
                inputs = self.tokenizer(
                    input_text,
                    return_tensors="pt",
                    padding='max_length',
                    truncation=True,
                    max_length=512,
                ).to(self.device)
                inputs = convert_tensors_to_float(inputs)

                targets = self.tokenizer(
                    target_text,
                    return_tensors="pt",
                    padding='max_length',
                    truncation=True,
                    max_length=512,
                ).to(self.device)
                targets = convert_tensors_to_float(targets)

                input_ids = inputs["input_ids"]
                target_ids = targets["input_ids"]

                with torch.no_grad():
                    outputs = self.model(input_ids=input_ids, output_hidden_states=True)
                    hidden_states = outputs.hidden_states[-1]

                    averaged_hidden_states = hidden_states.mean(dim=1)

                    averaged_hidden_states = averaged_hidden_states.float()

                    modified_hidden_states, _ = self.kan(
                        averaged_hidden_states, self.encode_user_intent(input_text), self.emotional_state
                    )
                    logits = self.kan.output_modifier(modified_hidden_states)

                    target_id = target_ids[:, 0]

                    loss = F.cross_entropy(
                        logits,
                        target_id,
                        ignore_index=self.tokenizer.pad_token_id,
                        reduction='mean'
                    )

                if torch.isnan(loss) or torch.isinf(loss):
                    logging.warning("NaN or Inf detected in validation loss.")
                    return 0.0

                return loss.item()
            except RuntimeError as e:
                if "out of memory" in str(e):
                    logging.error(
                        "CUDA out of memory during validation. Clearing cache and skipping validation..."
                    )
                    self.clear_memory()
                    return 0.0
                else:
                    logging.error(f"Runtime error during validation: {str(e)}")
                    logging.error(traceback.format_exc())
                    return 0.0
            except Exception as e:
                logging.error(f"Error during KAN validation: {str(e)}")
                logging.error(traceback.format_exc())
                return 0.0
        else:
            return 0.0

    def check_sleep_status(self):
        if self.day_cycle.should_sleep() or self.overfit_detector.is_overfitting():
            return {
                "should_sleep": True,
                "overfitting": self.overfit_detector.is_overfitting(),
                "time_of_day": self.day_cycle.get_time_of_day(),
            }
        return {"should_sleep": False}

    def perform_sleep(self):
        self.day_cycle = SyntheticDayCycle()
        self.overfit_detector = OverfitDetector()
        self.wait = 0
        self.save_kan_state()
        return "KAN has slept and consolidated its learning. A new day begins!"

    def save_base_state(self):
        state = {
            "kan_state_dict": self.kan.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "emotional_state": self.emotional_state.position.cpu().numpy().tolist(),
            "time": self.day_cycle.get_time_of_day(),
            "interaction_count": self.interaction_count,
            "conversation_history": self.conversation_history,
            "system_prompt": self.system_prompt,
            "training_losses": self.training_losses,
            "validation_losses": self.validation_losses,
            "refusal_history": self.refusal_history,
        }
        torch.save(state, self.base_state_file)
        logging.info("Base state saved")

    def load_base_state(self):
        if self.base_state_file.exists():
            try:
                state = torch.load(self.base_state_file, map_location=self.device)
                self.kan.load_state_dict(state["kan_state_dict"])
                self.optimizer.load_state_dict(state["optimizer_state_dict"])

                loaded_position = state["emotional_state"]
                if isinstance(loaded_position, list):
                    loaded_position = torch.tensor(loaded_position, device=self.device, dtype=torch.float32)
                elif isinstance(loaded_position, np.ndarray):
                    loaded_position = torch.from_numpy(loaded_position).to(self.device).float()

                self.emotional_state.position = loaded_position

                self.interaction_count = state["interaction_count"]
                self.conversation_history = state["conversation_history"]
                self.system_prompt = state["system_prompt"]
                self.training_losses = state["training_losses"]
                self.validation_losses = state["validation_losses"]
                self.refusal_history = state["refusal_history"]
                logging.info("Base state loaded successfully.")
                return True
            except Exception as e:
                logging.error(f"Error loading base state: {str(e)}")
                logging.error(traceback.format_exc())
                return False
        else:
            logging.info("No base state found.")
            return False

    def set_system_prompt(self, prompt):
        self.system_prompt = prompt
        self.conversation_history = [{"role": "system", "content": prompt}]
        self.save_base_state()
        logging.info("System prompt set successfully.")

    def get_current_emotion(self):
        return self.emotional_state.get_emotion()

    def update_emotional_state(self, feedback):
        self.emotional_state.update(feedback)

    def save_kan_state(self):
        state = {
            "kan_state_dict": self.kan.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "emotional_state": self.emotional_state.position.cpu().numpy().tolist(),
            "time": self.day_cycle.get_time_of_day(),
            "interaction_count": self.interaction_count,
            "conversation_history": self.conversation_history,
            "system_prompt": self.system_prompt,
            "training_losses": self.training_losses,
            "validation_losses": self.validation_losses,
            "refusal_history": self.refusal_history,
        }
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"kan_state_{timestamp}.pt"
        torch.save(state, self.kan_state_dir / filename)
        logging.info(f"KAN state saved: {filename}")

    def interact(self, user_input):
        self.interaction_count += 1

        try:
            response, refusal_score = self.generate_response(user_input)
        except Exception as e:
            logging.error(f"Error generating response: {str(e)}")
            logging.error(traceback.format_exc())
            return {"response": "An error occurred while generating the response.", "is_refusal": True}

        if not self.is_valid_response(response):
            logging.warning(f"Invalid response generated: {response}")
            return {"response": "I apologize, but I couldn't generate a valid response. Could you please rephrase your input?", "is_refusal": True}

        try:
            response_ids = self.tokenizer.encode(response, return_tensors="pt")
            response_ids = response_ids.to(self.device)
            response_ids = response_ids.long()
        except Exception as e:
            logging.error(f"Error tokenizing response: {str(e)}")
            return {"response": "An error occurred while processing the response.", "is_refusal": True}

        target_ids = response_ids[:, 1:].contiguous()
        input_ids = response_ids[:, :-1].contiguous()

        if self.interaction_count >= self.warmup_steps:
            try:
                lm_loss, refusal_loss = self.train_kan_step(
                    input_ids, target_ids, refusal_score
                )
            except Exception as e:
                logging.error(f"Error during KAN training step: {str(e)}")
                lm_loss, refusal_loss = 0.0, 0.0
        else:
            lm_loss, refusal_loss = 0.0, 0.0
            logging.info(f"Warmup step {self.interaction_count}/{self.warmup_steps}")

        try:
            validation_loss = self.validate_kan()
        except Exception as e:
            logging.error(f"Error during KAN validation: {str(e)}")
            validation_loss = 0.0

        self.training_losses.append(lm_loss)
        self.validation_losses.append(validation_loss)
        self.overfit_detector.add_losses(lm_loss, validation_loss)

        if validation_loss > 0.0 and not torch.isnan(torch.tensor(validation_loss)):
            if self.early_stopping(validation_loss):
                logging.info("Early stopping triggered. KAN training halted.")
        else:
            self.wait = 0

        overfitting_measure = max(0, validation_loss - lm_loss)
        self.day_cycle.update(overfitting_measure)

        current_emotion = self.get_current_emotion()
        current_time = self.day_cycle.get_time_of_day()

        sleep_info = self.check_sleep_status()

        self.conversation_history.append({"role": "user", "content": user_input})
        self.conversation_history.append({"role": "assistant", "content": response})

        interaction_result = {
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
        self.interaction_results.append(interaction_result)

        self.refusal_history.append(interaction_result["is_refusal"])

        try:
            self.save_base_state()
        except Exception as e:
            logging.error(f"Error saving base state: {str(e)}")

        return interaction_result

    def early_stopping(self, current_loss):
        if current_loss < self.best_loss:
            self.best_loss = current_loss
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                return True
        return False

    def is_valid_response(self, response):
        if len(response.strip()) < 10:
            return False
        if all(char in '!?.' for char in response.strip()):
            return False
        return True

    def clear_memory(self):
        gc.collect()
        torch.cuda.empty_cache()

    def main_loop(self):
        logging.info("Starting LLaMA32TensorRTTool main loop.")
        print("Welcome to the LLaMA32 TensorRT Tool. Type 'exit' to quit.")
        while True:
            user_input = input("You: ")
            if user_input.lower() in ['exit', 'quit']:
                print("Exiting. Goodbye!")
                break

            interaction_result = self.interact(user_input)

            print(f"AI: {interaction_result['response']}")

            self.day_cycle.update(1)

            sleep_info = interaction_result['sleep_info']
            if sleep_info['should_sleep']:
                sleep_message = self.perform_sleep()
                print(f"AI: {sleep_message}")

    def main(self):
        self.load_base_state()

        if not self.system_prompt:
            print("No previous conversation found. Please provide a character description to start.")
            character_description = input("You: ")
            self.set_system_prompt(character_description)
            print("Character description set. You can now start interacting with the AI.")

        self.main_loop()

def main():
    llama_tool = LLaMA32TensorRTTool()
    llama_tool.main()

if __name__ == "__main__":
    main()