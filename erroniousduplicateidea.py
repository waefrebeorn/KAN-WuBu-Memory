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

class AdaptiveKANLayer(nn.Module):
    def __init__(self, input_size, output_size, num_knots=10, temperature=0.666):
        super(AdaptiveKANLayer, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.num_knots = num_knots
        self.temperature = temperature

        # Define spline parameters
        self.knots = nn.Parameter(torch.linspace(-1, 1, num_knots))
        self.coeffs = nn.Parameter(torch.randn(input_size, output_size, num_knots))

    def forward(self, x):
        weights = self.compute_spline_weights(x)
        return torch.matmul(x, weights)

    def compute_spline_weights(self, x):
        weights = F.interpolate(self.coeffs.unsqueeze(0), size=(self.num_knots,)).squeeze(0)
        return weights

    def calculate_entropy(self, logits):
        p = F.softmax(logits, dim=-1)
        entropy = -torch.sum(p * torch.log(p + 1e-9), dim=-1)
        return entropy

    def adaptive_update(self, entropy, variance):
        if entropy < 0.1 and variance < 0.1:
            self.prune_knots()
        elif entropy > 5.0 and variance < 0.1:
            self.extend_knots()
        elif entropy < 5.0 and variance > 5.0:
            self.refine_coeffs()
        elif entropy > 5.0 and variance > 5.0:
            self.increase_capacity()
        else:
            self.moderate_update()

    def prune_knots(self):
        if self.num_knots > 3:
            self.num_knots -= 1
            self.knots = nn.Parameter(torch.linspace(-1, 1, self.num_knots))
            self.coeffs = nn.Parameter(torch.randn(self.input_size, self.output_size, self.num_knots))

    def extend_knots(self):
        self.num_knots += 1
        self.knots = nn.Parameter(torch.linspace(-1, 1, self.num_knots))
        self.coeffs = nn.Parameter(torch.randn(self.input_size, self.output_size, self.num_knots))

    def refine_coeffs(self):
        with torch.no_grad():
            self.coeffs += torch.randn_like(self.coeffs) * 0.01

    def increase_capacity(self):
        with torch.no_grad():
            self.coeffs = nn.Parameter(torch.cat([self.coeffs, torch.randn(self.input_size, self.output_size, self.num_knots)], dim=1))

    def moderate_update(self):
        self.refine_coeffs()

class AdaptiveKANNetwork(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, num_layers=3, temperature=0.666):
        super(AdaptiveKANNetwork, self).__init__()
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.num_layers = num_layers
        self.temperature = temperature

        self.layers = nn.ModuleList()
        in_size = input_size
        for hidden_size in hidden_sizes:
            self.layers.append(AdaptiveKANLayer(in_size, hidden_size, num_knots=10, temperature=temperature))
            in_size = hidden_size
        self.output_layer = AdaptiveKANLayer(in_size, output_size, num_knots=10, temperature=temperature)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
            x = F.relu(x)
        return self.output_layer(x)

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
        # Updated to minimize internal content leakage to user
        context = f"{self.system_prompt}\n\nCurrent Emotion: {current_emotion}\n"
        context += "Internal Dialogue:\n"
        for message in self.conversation_history[-4:]:
            role = message['role'].capitalize()
            content = message['content']
            if role == "System":
                context += f"System Guidance: {content}\n"
            else:
                context += f"{role}: {content}\n"
        context += f"User: {user_input}\nAssistant: "
        return context

    def generate_response(self, user_input):
        try:
            user_intent = self.encode_user_intent(user_input)

            current_emotion = self.emotional_state.get_emotion()
            context = self.prepare_context(user_input, current_emotion)

            # Generate the response using the modified context
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

    def interact(self, user_input):
        self.interaction_count += 1

        try:
            # Generate the response while monitoring internal state and potential self-talk issues
            response, refusal_score = self.generate_response(user_input)
        except Exception as e:
            logging.error(f"Error generating response: {str(e)}")
            logging.error(traceback.format_exc())
            return {"response": "An error occurred while generating the response.", "is_refusal": True}

        # Check if the response is valid
        if not self.is_valid_response(response):
            logging.warning(f"Invalid response generated: {response}")
            return {"response": "I apologize, but I couldn't generate a valid response. Could you please rephrase your input?", "is_refusal": True}

        try:
            # Convert the response to input IDs for further processing
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
                # Train the KAN component with the generated response
                lm_loss, refusal_loss = self.train_kan_step(
                    input_ids, target_ids, refusal_score
                )
            except Exception as e:
                logging.error(f"Error during KAN training step: {str(e)}")
                lm_loss, refusal_loss = 0.0, 0.0
        else:
            lm_loss, refusal_loss = 0.0, 0.0
            logging.info(f"Warmup step {self.interaction_count}/{self.warmup_steps}")

        # Run validation on the updated KAN state
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

        # Update day cycle and overfitting measures
        overfitting_measure = max(0, validation_loss - lm_loss)
        self.day_cycle.update(overfitting_measure)

        current_emotion = self.get_current_emotion()
        current_time = self.day_cycle.get_time_of_day()

        sleep_info = self.check_sleep_status()

        # Internal state updates without exposing self-talk to the user
        self.conversation_history.append({"role": "user", "content": user_input})
        self.conversation_history.append({"role": "assistant", "content": response, "internal": False})

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

    def prepare_internal_context(self):
        """
        Prepares an internal context for model self-guidance and evaluation.
        Keeps track of internal self-references and adjusts behavior accordingly.
        """
        context = "Internal Reflection and Analysis:\n"
        for message in self.conversation_history:
            if "internal" in message and message["internal"]:
                role = message['role'].capitalize()
                content = message['content']
                context += f"{role}: {content}\n"
        return context

    def internal_diagnostic_check(self):
        """
        Runs internal diagnostics to validate the state of the KAN and LLaMA models.
        Detects self-talk patterns and monitors internal state drift.
        """
        internal_context = self.prepare_internal_context()

        try:
            # Generate a self-reflective response for internal evaluation
            response, _ = self.generate_and_validate_response(internal_context, self.refusal_detector)
            logging.info(f"Internal Diagnostic Response: {response}")

            if "self-talk" in response.lower():
                logging.warning("Self-talk detected during internal diagnostics.")
                self.update_emotional_state([-0.5, 0.0])  # Adjust emotional state accordingly
        except Exception as e:
            logging.error(f"Error during internal diagnostic check: {str(e)}")
            logging.error(traceback.format_exc())

    def perform_self_guidance(self):
        """
        Executes self-guidance routines to correct internal inconsistencies.
        Employs entropy-driven feedback to recalibrate model states.
        """
        logging.info("Performing self-guidance to ensure coherent internal state.")
        self.internal_diagnostic_check()

        # Adjust internal layers and memory consolidation based on diagnostic results
        entropy_score = torch.mean(torch.tensor([message["content"] for message in self.conversation_history if "internal" in message])).item()

        if entropy_score > 1.0:
            logging.warning("High entropy detected in internal state. Triggering corrective mechanisms.")
            self.refine_internal_states()
        else:
            logging.info("Internal state appears stable. No immediate corrective action required.")


    def refine_internal_states(self):
        """
        Refines the internal state layers and memory components.
        Focuses on reducing self-talk and enhancing coherent behavior.
        """
        logging.info("Refining internal states to address detected issues.")
        for message in self.conversation_history:
            if "internal" in message and message["internal"]:
                message["content"] = re.sub(r'\bself-talk\b', "reflection", message["content"])
                logging.info(f"Refined internal message: {message['content']}")
                
        # Reassess the emotional state after refinement
        self.update_emotional_state([0.2, -0.1])  # Apply a stabilizing adjustment

    def memory_system_update(self, user_input, response):
        """
        Updates the memory system based on entropy and user context.
        Identifies key interactions and retains meaningful exchanges.
        """
        if not user_input.strip() or not response.strip():
            return

        # Calculate interaction entropy and determine if memory update is required
        current_entropy = self.calculate_interaction_entropy(user_input, response)
        memory_threshold = 0.4

        if current_entropy < memory_threshold:
            logging.info("Interaction entropy below threshold; adding to long-term memory.")
            self.store_in_long_term_memory(user_input, response)
        else:
            logging.info("High entropy detected; storing in short-term memory for refinement.")
            self.store_in_short_term_memory(user_input, response)

    def calculate_interaction_entropy(self, user_input, response):
        """
        Calculates the entropy of the interaction based on context and response patterns.
        Helps determine whether the memory should be retained or refined.
        """
        tokenized_input = self.tokenizer.encode(user_input, return_tensors="pt").to(self.device)
        tokenized_response = self.tokenizer.encode(response, return_tensors="pt").to(self.device)

        with torch.no_grad():
            input_entropy = torch.mean(self.calculate_token_entropy(tokenized_input))
            response_entropy = torch.mean(self.calculate_token_entropy(tokenized_response))

        interaction_entropy = (input_entropy + response_entropy) / 2
        return interaction_entropy.item()

    def calculate_token_entropy(self, tokenized_tensor):
        """
        Calculates the entropy of individual tokens within a tensor.
        Used to gauge uncertainty and coherence of a given sequence.
        """
        with torch.no_grad():
            output = self.model(input_ids=tokenized_tensor, output_hidden_states=True)
            logits = output.logits
            entropy = -torch.sum(F.softmax(logits, dim=-1) * F.log_softmax(logits, dim=-1), dim=-1)
        return entropy

    def store_in_long_term_memory(self, user_input, response):
        """
        Stores meaningful interactions in the long-term memory system.
        Prioritizes interactions with low entropy and high coherence.
        """
        if not hasattr(self, 'long_term_memory'):
            self.long_term_memory = []

        if len(self.long_term_memory) > 50:
            self.long_term_memory.pop(0)  # Maintain a reasonable memory size

        self.long_term_memory.append({"user": user_input, "assistant": response})
        logging.info("Added interaction to long-term memory.")

    def store_in_short_term_memory(self, user_input, response):
        """
        Stores interactions with higher entropy in short-term memory.
        These interactions may be refined or discarded based on future evaluations.
        """
        if not hasattr(self, 'short_term_memory'):
            self.short_term_memory = []

        if len(self.short_term_memory) > 20:
            self.short_term_memory.pop(0)  # Maintain a reasonable memory size

        self.short_term_memory.append({"user": user_input, "assistant": response})
        logging.info("Added interaction to short-term memory.")

    def adjust_memory_based_on_context(self, context_feedback):
        """
        Dynamically adjusts the memory system based on context and feedback.
        Helps refine long-term memory storage and emotional state stability.
        """
        if context_feedback < -0.5:
            logging.info("Negative context feedback received; refining long-term memory.")
            self.refine_long_term_memory()
        elif context_feedback > 0.5:
            logging.info("Positive feedback; reinforcing key interactions.")
            self.reinforce_long_term_memory()

    def refine_long_term_memory(self):
        """
        Prunes and refines long-term memory to remove inconsistencies.
        """
        if hasattr(self, 'long_term_memory'):
            initial_size = len(self.long_term_memory)
            self.long_term_memory = [
                memory for memory in self.long_term_memory
                if self.calculate_interaction_entropy(memory["user"], memory["assistant"]) < 0.3
            ]
            logging.info(f"Refined long-term memory. Size reduced from {initial_size} to {len(self.long_term_memory)}.")

    def reinforce_long_term_memory(self):
        """
        Reinforces meaningful interactions in the long-term memory.
        Increases the importance and recurrence of certain memories.
        """
        if hasattr(self, 'long_term_memory'):
            for memory in self.long_term_memory:
                if self.calculate_interaction_entropy(memory["user"], memory["assistant"]) < 0.2:
                    memory["importance"] = memory.get("importance", 1) + 1
                    logging.info(f"Reinforced memory: {memory}")

    def retrieve_relevant_memory(self, user_input):
        """
        Retrieves the most relevant memory based on user input and context.
        Uses similarity metrics and entropy thresholds to select the memory.
        """
        if not hasattr(self, 'long_term_memory') or not self.long_term_memory:
            return ""

        input_tokens = self.tokenizer.encode(user_input, return_tensors="pt").to(self.device)
        similarities = []

        with torch.no_grad():
            for memory in self.long_term_memory:
                memory_tokens = self.tokenizer.encode(memory["user"], return_tensors="pt").to(self.device)
                similarity = F.cosine_similarity(input_tokens.float(), memory_tokens.float(), dim=-1)
                similarities.append((memory, similarity.item()))

        if not similarities:
            return ""

        # Retrieve the memory with the highest similarity
        relevant_memory = max(similarities, key=lambda x: x[1])[0]
        return relevant_memory.get("assistant", "")

    def prepare_context(self, user_input, current_emotion, include_memory=True):
        """
        Prepares the context for generating a response.
        Dynamically adjusts based on current emotional state and relevant memory retrieval.

        Args:
            user_input (str): The current user input.
            current_emotion (str): The current emotional state of the system.
            include_memory (bool): Whether to include relevant memory in the context.

        Returns:
            context (str): Prepared context string for the model.
        """
        context = f"{self.system_prompt}\n\nCurrent Emotion: {current_emotion}\n"
        context += "Conversation:\n"
        for message in self.conversation_history[-4:]:
            role = message['role'].capitalize()
            content = message['content']
            if not message.get("internal", False):  # Exclude internal reflections from context
                context += f"{role}: {content}\n"
        
        if include_memory:
            # Retrieve the most relevant memory based on the current user input
            relevant_memory = self.retrieve_relevant_memory(user_input)
            if relevant_memory:
                context += f"Assistant's Reflection: {relevant_memory}\n"

        context += f"User: {user_input}\nAssistant: "
        return context

    def generate_response(self, user_input):
        """
        Generates a response based on the user's input, leveraging the prepared context,
        memory management, and entropy-based feedback.

        Args:
            user_input (str): The input text from the user.

        Returns:
            response (str): The generated response from the assistant.
            refusal_score (float): A score indicating if the response contains a refusal.
        """
        try:
            user_intent = self.encode_user_intent(user_input)

            # Update and retrieve the current emotional state
            current_emotion = self.emotional_state.get_emotion()

            # Prepare the context with updated emotional state and memory inclusion
            context = self.prepare_context(user_input, current_emotion, include_memory=True)

            # Generate the response and validate for refusal
            response, refusal_score = self.generate_and_validate_response(context, self.refusal_detector)

            # Update the memory system with the new interaction
            self.memory_system_update(user_input, response)

            # Refinement of internal states post-generation
            self.refine_internal_states()

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
        """
        Generates a response and validates it against the refusal detector.
        Implements a multi-stage response refinement loop if needed.

        Args:
            prompt (str): The context and prompt for response generation.
            refusal_detector (RefusalDetector): The refusal detection component.
            max_new_tokens (int): Maximum number of tokens to generate.
            chunk_size (int): Size of each generation chunk.

        Returns:
            response (str): The generated response.
            refusal_score (float): The refusal score indicating the likelihood of refusal content.
        """
        response = self.generate_full_response(prompt, max_new_tokens, chunk_size)

        refusal_score = refusal_detector.detect_refusal(response)
        if refusal_score > 0.5:
            logging.warning("Response failed Refusal Check. Attempting to regenerate.")
            continuation_prompt = prompt + response + " Please continue."

            # Regenerate response if the initial response was a refusal
            response = self.generate_full_response(continuation_prompt, max_new_tokens, chunk_size)
            refusal_score = refusal_detector.detect_refusal(response)

            if refusal_score > 0.5:
                logging.error("Regenerated response also failed Refusal Check.")
                response = "I'm sorry, but I'm unable to provide a complete response at the moment."
                refusal_score = 1.0

        return response, refusal_score

    def validate_kan(self):
        """
        Validates the KAN model's output using entropy metrics and dynamic context.

        Returns:
            loss (float): The calculated loss for the KAN model during validation.
        """
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

    def refine_internal_states(self):
        """
        Refines internal states of both the memory system and the KAN-Llama interaction.
        Adjusts activation patterns, emotion trajectory, and response consistency.
        """
        try:
            # Step 1: Analyze interaction patterns for anomalies or inconsistencies
            if len(self.interaction_results) > 0:
                latest_interaction = self.interaction_results[-1]
                response = latest_interaction.get("response", "")
                if self.is_incoherent(response):
                    logging.warning("Incoherent response detected, refining internal states.")
                    # Introduce adjustments to prevent further incoherence
                    self.emotional_state.update([-0.1, -0.1])  # Slightly adjust the emotional state
                    self.day_cycle.update(-1)  # Modify the synthetic cycle to simulate fatigue

            # Step 2: Re-align internal memory with context updates if needed
            self.memory_recalibration()

            # Step 3: Ensure response and memory consistency
            self.validate_internal_consistency()

        except Exception as e:
            logging.error(f"Error during internal state refinement: {str(e)}")
            logging.error(traceback.format_exc())

    def is_incoherent(self, response):
        """
        Checks if a response is incoherent based on a set of predefined rules.

        Args:
            response (str): The response generated by the model.

        Returns:
            bool: True if the response is incoherent, False otherwise.
        """
        if len(response) < 10:
            return True
        if response.strip().count("?") > 2:
            return True
        if response.strip().startswith("...") or response.strip().endswith("..."):
            return True
        if "I don't know" in response or "I'm not sure" in response:
            return True
        return False

    def memory_recalibration(self):
        """
        Performs recalibration of memory components based on the latest interaction patterns.
        Uses entropy measures to prune or expand memory slots dynamically.
        """
        try:
            current_emotion = self.get_current_emotion()
            recent_conversation = " ".join([msg['content'] for msg in self.conversation_history[-4:] if not msg.get("internal", False)])

            # Use entropy as a heuristic to determine memory adjustments
            entropy_measure = self.calculate_entropy(recent_conversation)
            if entropy_measure > 1.5:
                logging.info("High entropy detected. Memory recalibration needed.")
                self.consolidate_high_entropy_memories()
            else:
                logging.info("Memory state is stable. No immediate recalibration required.")

        except Exception as e:
            logging.error(f"Error during memory recalibration: {str(e)}")
            logging.error(traceback.format_exc())

    def calculate_entropy(self, text):
        """
        Calculates the entropy of a given text string based on token probabilities.

        Args:
            text (str): The input text to calculate entropy for.

        Returns:
            entropy (float): The entropy value of the text.
        """
        try:
            inputs = self.tokenizer.encode(text, return_tensors='pt').to(self.device)
            inputs = inputs.long()
            
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

    def consolidate_high_entropy_memories(self):
        """
        Consolidates high-entropy memories by merging redundant entries and streamlining long-term memory.
        """
        if len(self.conversation_history) < 2:
            return  # Not enough memory to consolidate

        try:
            # Group and merge similar high-entropy memories
            high_entropy_memories = [msg for msg in self.conversation_history if self.calculate_entropy(msg['content']) > 1.5]
            if len(high_entropy_memories) < 2:
                return

            merged_content = " ".join([msg['content'] for msg in high_entropy_memories])
            consolidated_entry = {
                "role": "assistant",
                "content": merged_content,
                "internal": True  # Mark as consolidated internal memory
            }

            # Remove redundant entries and add consolidated memory
            self.conversation_history = [msg for msg in self.conversation_history if msg not in high_entropy_memories]
            self.conversation_history.append(consolidated_entry)
            logging.info("High-entropy memories consolidated successfully.")

        except Exception as e:
            logging.error(f"Error during memory consolidation: {str(e)}")
            logging.error(traceback.format_exc())

    def validate_internal_consistency(self):
        """
        Validates internal consistency of the memory and emotional state components.
        Adjusts any detected inconsistencies using entropy-based correction.
        """
        try:
            # Calculate the coherence score between the memory system and current context
            if len(self.conversation_history) > 2:
                recent_memory = self.conversation_history[-2]['content']
                current_emotion = self.get_current_emotion()
                
                # Determine the consistency score
                consistency_score = self.calculate_consistency(recent_memory, current_emotion)
                if consistency_score < 0.5:
                    logging.warning("Low consistency detected between memory and emotion.")
                    self.realign_emotional_state()

        except Exception as e:
            logging.error(f"Error during internal consistency validation: {str(e)}")
            logging.error(traceback.format_exc())

    def calculate_consistency(self, memory_text, emotion):
        """
        Calculates a consistency score between a memory text and the current emotional state.

        Args:
            memory_text (str): The memory text to evaluate.
            emotion (str): The current emotion state.

        Returns:
            score (float): Consistency score between 0 and 1.
        """
        try:
            emotion_vector = torch.tensor(self.emotional_state.position, dtype=torch.float32, device=self.device)
            inputs = self.tokenizer(memory_text, return_tensors='pt').to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
                hidden_state = outputs.last_hidden_state.mean(dim=1)

            # Cosine similarity as a proxy for consistency
            similarity = F.cosine_similarity(hidden_state, emotion_vector, dim=-1)
            return similarity.item()
        except Exception as e:
            logging.error(f"Error calculating consistency: {str(e)}")
            logging.error(traceback.format_exc())
            return 0.0

    def realign_emotional_state(self):
        """
        Realigns the emotional state if the internal consistency is too low.
        """
        logging.info("Realigning emotional state based on memory-context inconsistencies.")
        self.emotional_state.update([0.1, -0.1])  # Introduce minor adjustments to bring the state back on track


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

    def update_emotional_state(self, feedback):
        """
        Updates the emotional state based on feedback vectors and internal consistency.
        Ensures the trajectory of emotional state aligns with the memory system.
        """
        try:
            if feedback is None:
                logging.warning("No feedback provided. Skipping emotional state update.")
                return

            # Update the emotional state vector
            self.emotional_state.update(feedback)

            # Refine internal states based on updated emotion
            self.refine_internal_states()

        except Exception as e:
            logging.error(f"Error updating emotional state: {str(e)}")
            logging.error(traceback.format_exc())

    def refine_internal_states(self):
        """
        Refines internal states based on updated emotion and interaction history.
        Modifies the KAN's influence on model layers to ensure that LLaMA receives a coherent context.
        """
        try:
            # Use current emotional state position to recalibrate KAN influence
            emotional_vector = self.emotional_state.position
            adjusted_layers = 0

            # Adjust KAN influence on deeper LLaMA layers based on emotional vector magnitude
            for i, layer in enumerate(self.kan.refusal_override.override_generator.parameters()):
                if torch.norm(emotional_vector) > 0.5:
                    layer.data *= 1.1  # Increase influence if strong emotion detected
                    adjusted_layers += 1
                else:
                    layer.data *= 0.9  # Reduce influence if neutral emotion

            logging.info(f"Adjusted {adjusted_layers} layers based on emotional refinement.")

        except Exception as e:
            logging.error(f"Error in refining internal states: {str(e)}")
            logging.error(traceback.format_exc())

    def prepare_context(self, user_input, current_emotion):
        """
        Prepares the conversational context for response generation.
        Includes dynamic memory updates and selective pruning to prevent self-talk.
        """
        try:
            # Build the context string
            context = f"{self.system_prompt}\n\nCurrent Emotion: {current_emotion}\n"
            context += "Conversation:\n"

            # Prune and add only relevant history based on entropy levels
            for message in self.conversation_history[-4:]:
                role = message['role'].capitalize()
                content = message['content']
                entropy_score = self.refusal_detector.detect_refusal(content)
                
                if entropy_score < 0.5:  # Skip irrelevant or incoherent parts
                    context += f"{role}: {content}\n"

            # Add user input as the last part of the context
            context += f"User: {user_input}\nAssistant: "

            logging.debug(f"Prepared context: {context}")
            return context

        except Exception as e:
            logging.error(f"Error preparing context: {str(e)}")
            logging.error(traceback.format_exc())
            return ""

    def interact(self, user_input):
        """
        Main interaction method handling user inputs, KAN training, and response generation.
        Includes dynamic memory consolidation and pruning to prevent incoherent responses.
        """
        self.interaction_count += 1

        try:
            # Generate response based on user input
            response, refusal_score = self.generate_response(user_input)
        except Exception as e:
            logging.error(f"Error generating response: {str(e)}")
            logging.error(traceback.format_exc())
            return {"response": "An error occurred while generating the response.", "is_refusal": True}

        # Validate response for self-talk patterns and coherence
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

        # KAN training and dynamic adjustment based on the entropy levels of the current interaction
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

        # Update loss records and check for overfitting
        self.training_losses.append(lm_loss)
        self.validation_losses.append(validation_loss)
        self.overfit_detector.add_losses(lm_loss, validation_loss)

        if validation_loss > 0.0 and not torch.isnan(torch.tensor(validation_loss)):
            if self.early_stopping(validation_loss):
                logging.info("Early stopping triggered. KAN training halted.")
        else:
            self.wait = 0

        # Calculate overfitting measure and update synthetic day cycle
        overfitting_measure = max(0, validation_loss - lm_loss)
        self.day_cycle.update(overfitting_measure)

        # Retrieve current emotional state and time
        current_emotion = self.get_current_emotion()
        current_time = self.day_cycle.get_time_of_day()

        # Check if the system needs to perform a sleep cycle
        sleep_info = self.check_sleep_status()

        # Append the user input and response to conversation history
        self.conversation_history.append({"role": "user", "content": user_input})
        self.conversation_history.append({"role": "assistant", "content": response})

        # Store the interaction results for analysis and refinement
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

        # Update refusal history and save the base state
        self.refusal_history.append(interaction_result["is_refusal"])

        try:
            self.save_base_state()
        except Exception as e:
            logging.error(f"Error saving base state: {str(e)}")

        return interaction_result

    def get_current_emotion(self):
        """
        Returns the current emotion based on the EmotionalState vector.
        Also refines the result by checking for outlier activations in the internal states.
        """
        try:
            emotion = self.emotional_state.get_emotion()

            # Apply entropy-based corrections if detected patterns indicate incoherence
            if self.detect_incoherent_state():
                emotion = "Neutral"

            return emotion

        except Exception as e:
            logging.error(f"Error retrieving current emotion: {str(e)}")
            return "Unknown"

    def detect_incoherent_state(self):
        """
        Detects whether the current system state is incoherent by evaluating internal entropy patterns.
        Returns True if an incoherent state is detected; otherwise, False.
        """
        try:
            # Evaluate entropy patterns in the refusal detector or any internal state
            entropy_levels = [
                self.refusal_detector.detect_refusal(msg['content'])
                for msg in self.conversation_history
                if msg['role'] == 'assistant'
            ]

            # If more than 50% of recent entropy levels indicate high uncertainty, return True
            return sum(1 for level in entropy_levels if level > 0.75) / len(entropy_levels) > 0.5

        except Exception as e:
            logging.error(f"Error detecting incoherent state: {str(e)}")
            return False

# -------------------- Main Execution Function --------------------
def main():
    llama_tool = LLaMA32TensorRTTool()
    llama_tool.main()

if __name__ == "__main__":
    main()
