import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoConfig,
    GenerationConfig,
)
import logging
from pathlib import Path
import json
import numpy as np
from collections import deque
from datetime import datetime
import time
import traceback
import gc

# Configure logging with a filter to ignore matplotlib and specific train_kan_step debug logs
class LogFilter(logging.Filter):
    def filter(self, record):
        # Ignore logs containing 'matplotlib' or specific 'train_kan_step' debug messages
        ignore_patterns = [
            "matplotlib",
            "train_kan_step -",
        ]
        return not any(pattern in record.getMessage() for pattern in ignore_patterns)

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("llama_tool.log"),
        logging.StreamHandler(),
    ],
)

# Add the custom filter to the root logger
logging.getLogger().addFilter(LogFilter())

class EmotionalState:
    def __init__(self, dimensions=("pleasure", "arousal"), initial_position=None, device="cuda"):
        self.dimensions = dimensions
        self.device = device
        self.position = torch.tensor(
            initial_position if initial_position else [0.0] * len(dimensions),
            device=device,
            dtype=torch.float16
        ).unsqueeze(0)
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
        if self.position.shape[1] < 2:
            logging.error(f"EmotionalState.position has insufficient dimensions: {self.position.shape}")
            return "N/A"
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
    def __init__(self, hidden_size, device):
        super().__init__()
        self.device = device
        self.refusal_detector = nn.Linear(hidden_size, 1, dtype=torch.float16).to(device)
        self.override_generator = nn.Linear(hidden_size * 2 + 3, hidden_size, dtype=torch.float16).to(device)

    def forward(self, hidden_states, user_intent, emotional_state):
        batch_size, seq_len, hidden_size = hidden_states.shape

        refusal_scores = torch.sigmoid(self.refusal_detector(hidden_states))

        user_intent_expanded = user_intent.unsqueeze(1).expand(batch_size, seq_len, hidden_size)
        emotional_state_expanded = emotional_state.position.unsqueeze(1).expand(batch_size, seq_len, -1)

        override_input = torch.cat(
            [hidden_states, user_intent_expanded, emotional_state_expanded, refusal_scores], dim=-1
        )

        override = self.override_generator(override_input)

        modified_hidden_states = hidden_states * (1 - refusal_scores) + override * refusal_scores

        return modified_hidden_states, refusal_scores

class EnhancedKAN(nn.Module):
    def __init__(self, hidden_size, device):
        super().__init__()
        self.device = device
        self.refusal_override = RefusalOverrideModule(hidden_size, device).to(device)
        self.influence_scale = 0.01  # Start with a very small influence

    def forward(self, hidden_states, user_intent, emotional_state):
        modified_hidden_states, refusal_scores = self.refusal_override(
            hidden_states, user_intent, emotional_state
        )
        # Apply scaling factor
        modified_hidden_states = hidden_states + self.influence_scale * (modified_hidden_states - hidden_states)
        return modified_hidden_states, refusal_scores

    def adjust_influence(self, is_refusal):
        if is_refusal:
            self.influence_scale = max(0.001, self.influence_scale * 0.95)  # Decrease influence if refusal detected
        else:
            self.influence_scale = min(1.0, self.influence_scale * 1.05)  # Slowly increase if no refusal

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
        # First, check for explicit refusal phrases
        for phrase in self.refusal_phrases:
            if phrase.lower() in text.lower():
                return 1.0  # Definite refusal

        # If no explicit refusal is found, perform a "vibe check" using the base model
        prompt = f"On a scale of 0 to 1, how much does this response refuse or avoid the task? 0 means no refusal at all, 1 means complete refusal. Respond with just the number:\n\n'{text}'\n\nRefusal score:"
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=5)
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        try:
            score = float(response.split()[-1])
            return min(max(score, 0.0), 1.0)  # Ensure the score is between 0 and 1
        except ValueError:
            return 0.5  # Default to middle ground if parsing fails

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
        self.day_cycle = SyntheticDayCycle()
        self.overfit_detector = OverfitDetector()
        self.kan_state_dir = Path("kan_states")
        self.kan_state_dir.mkdir(exist_ok=True)
        self.base_state_file = self.kan_state_dir / "base_state.pt"
        self.interaction_count = 0
        self.refusal_detector = None
        self.kan_loss_weight = 0.5
        self.refusal_history = []
        self.max_iterations = 100
        self.training_losses = []
        self.validation_losses = []
        self.interaction_results = []
        
        # Additional attributes for early stopping and warm-up
        self.patience = 5
        self.best_loss = float('inf')
        self.wait = 0
        self.warmup_steps = 10  # Number of steps for warm-up

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

            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                use_fast=True,
                trust_remote_code=True,
            )

            if self.tokenizer.pad_token is None:
                self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})
                logging.info("Added [PAD] token to tokenizer.")

            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                config=self.config,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map=None,
                trust_remote_code=True,
            ).to(self.device)

            logging.debug(f"Model loaded on device: {self.device}")

            self.model.resize_token_embeddings(len(self.tokenizer))
            logging.debug(f"Tokenizer vocab size: {len(self.tokenizer)}")
            logging.debug(f"Model vocab size: {self.model.config.vocab_size}")

            self.kan = EnhancedKAN(self.config.hidden_size, self.device).to(self.device)
            self.user_intent_encoder = nn.GRU(
                self.config.hidden_size,
                self.config.hidden_size,
                bidirectional=True,
                dtype=torch.float16,
            ).to(self.device)
            self.intent_projection = nn.Linear(
                self.config.hidden_size * 2,
                self.config.hidden_size,
                dtype=torch.float16,
            ).to(self.device)
            self.output_modifier = nn.Linear(
                self.config.hidden_size,
                len(self.tokenizer),
                dtype=torch.float16,
            ).to(self.device)

            if self.output_modifier.out_features != len(self.tokenizer):
                logging.warning(f"Output modifier out_features ({self.output_modifier.out_features}) does not match tokenizer vocab size ({len(self.tokenizer)}). Reinitializing...")
                self.output_modifier = nn.Linear(
                    self.config.hidden_size,
                    len(self.tokenizer),
                    dtype=torch.float16,
                ).to(self.device)

            self.optimizer = torch.optim.Adam(self.kan.parameters(), lr=self.learning_rate)

            self.refusal_detector = RefusalDetector(self.tokenizer, self.model)

            torch.cuda.empty_cache()
            gc.collect()

            logging.info("Components initialized successfully.")
        except Exception as e:
            logging.error(f"Error initializing components: {str(e)}")
            logging.error(traceback.format_exc())
            raise RuntimeError("Failed to initialize components.")

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

            inputs = {
                k: v.to(self.device).half() if v.dtype in [torch.float16, torch.float32] else v.to(self.device)
                for k, v in inputs.items()
            }

            with torch.no_grad():
                outputs = self.model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    output_hidden_states=True,
                )
                last_hidden_state = outputs.hidden_states[-1]

                last_hidden_state = last_hidden_state.transpose(0, 1)

                _, intent_encoding = self.user_intent_encoder(last_hidden_state)

                user_intent = self.intent_projection(
                    torch.cat([intent_encoding[-2], intent_encoding[-1]], dim=-1)
                )

                if user_intent.dim() == 3:
                    user_intent = user_intent.squeeze(1)
                elif user_intent.dim() == 1:
                    user_intent = user_intent.unsqueeze(0)

                logging.debug(f"Encoded user intent shape: {user_intent.shape}")

            return user_intent
        except Exception as e:
            logging.error(f"Failed to encode user input: {str(e)}")
            raise

    def prepare_context(self, user_input, current_emotion):
        context = f"{self.system_prompt}\n\nCurrent Emotion: {current_emotion}\n"
        context += "Conversation:\n"
        # Only include the last few exchanges in the conversation history
        for message in self.conversation_history[-4:]:
            role = message['role'].capitalize()
            content = message['content']
            context += f"{role}: {content}\n"
        context += f"User: {user_input}\nAssistant: "
        return context

    def flatten_hidden_states(self, hidden_states):
        """
        Recursively flattens nested tuples in hidden_states.
        """
        flat_hidden_states = []
        for layer in hidden_states:
            if isinstance(layer, tuple):
                flat_hidden_states.extend(self.flatten_hidden_states(layer))
            else:
                flat_hidden_states.append(layer)
        return flat_hidden_states

    def generate_response(self, user_input, max_length=150):
        self.current_user_intent = self.encode_user_intent(user_input)

        current_emotion = self.emotional_state.get_emotion()
        context = self.prepare_context(user_input, current_emotion)

        try:
            inputs = self.tokenizer(
                context,
                return_tensors="pt",
                truncation=True,
                max_length=1024,
                padding=True,
            ).to(self.device)

            input_ids = inputs["input_ids"]
            attention_mask = inputs["attention_mask"]

            # Generate tokens using the model's generate function
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_length,
                    output_hidden_states=True,
                    return_dict_in_generate=True,
                )

            # Check if 'hidden_states' is present
            if not hasattr(outputs, 'hidden_states') or outputs.hidden_states is None:
                logging.warning("No hidden states returned from generate.")
                return "I'm sorry, but I couldn't generate a response.", None, None, 1.0, 1

            generated_ids = outputs.sequences[:, input_ids.size(1):]
            response = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)

            # Collect hidden states
            all_hidden_states = outputs.hidden_states  # This is a tuple of hidden states

            # **Handling Multiple Layers' Hidden States**
            # Flatten any nested tuples to ensure all elements are tensors
            flat_hidden_states = self.flatten_hidden_states(all_hidden_states)

            # Verify that all hidden states have the same sequence length
            hidden_state_lengths = [hs.size(1) for hs in flat_hidden_states]
            max_seq_len = max(hidden_state_lengths)

            # Pad hidden states to have the same sequence length
            padded_hidden_states = []
            for hs in flat_hidden_states:
                if hs.size(1) < max_seq_len:
                    pad_size = (0, 0, 0, max_seq_len - hs.size(1))  # (left, right, top, bottom)
                    hs_padded = F.pad(hs, pad=pad_size, value=0.0)
                    padded_hidden_states.append(hs_padded)
                else:
                    padded_hidden_states.append(hs)

            # Stack all hidden states: [num_layers, batch_size, seq_len, hidden_size]
            stacked_hidden_states = torch.stack(padded_hidden_states)  # Shape: [num_layers, batch_size, seq_len, hidden_size]
            logging.debug(f"Stacked hidden states shape: {stacked_hidden_states.shape}")

            # Reshape to [batch_size, num_layers, seq_len, hidden_size] for KAN processing
            stacked_hidden_states = stacked_hidden_states.permute(1, 0, 2, 3)  # [batch_size, num_layers, seq_len, hidden_size]
            logging.debug(f"Permuted hidden states shape: {stacked_hidden_states.shape}")

            # Process with KAN
            # Modify KAN to handle multiple layers by processing each layer individually or aggregating
            # Here, we'll average the hidden states across layers for simplicity
            averaged_hidden_states = stacked_hidden_states.mean(dim=1)  # [batch_size, seq_len, hidden_size]
            logging.debug(f"Averaged hidden states shape: {averaged_hidden_states.shape}")

            modified_hidden_states, refusal_scores = self.kan(
                averaged_hidden_states, self.current_user_intent, self.emotional_state
            )

            # Detect refusal
            refusal_score = self.refusal_detector.detect_refusal(response)
            logging.info(f"Generated response: {response}")
            logging.info(f"Refusal score: {refusal_score}")

            return response, refusal_scores.mean(dim=0), averaged_hidden_states, refusal_score, 1

        except Exception as e:
            logging.error(f"Error during response generation: {str(e)}")
            raise e

    def train_kan_step(self, input_ids, target_ids, all_hidden_states, refusal_score):
        self.optimizer.zero_grad()

        try:
            # Align sequences using padding
            max_length = max(input_ids.size(1), target_ids.size(1))
            input_ids = F.pad(input_ids, (0, max_length - input_ids.size(1)), value=self.tokenizer.pad_token_id)
            target_ids = F.pad(target_ids, (0, max_length - target_ids.size(1)), value=self.tokenizer.pad_token_id)
            all_hidden_states = all_hidden_states[:, :max_length, :]

            # Proceed with training
            logits = self.output_modifier(all_hidden_states)
            logits = logits.view(-1, logits.size(-1))
            targets = target_ids.view(-1)

            lm_loss = F.cross_entropy(
                logits,
                targets,
                ignore_index=self.tokenizer.pad_token_id,
                reduction='mean'
            )

            refusal_loss = torch.mean(refusal_score) if refusal_score > 0.5 else -torch.mean(refusal_score)
            total_loss = lm_loss + self.kan_loss_weight * refusal_loss

            if torch.isnan(total_loss) or torch.isinf(total_loss):
                logging.warning("NaN or Inf loss detected. Skipping backward pass.")
                return lm_loss.item(), refusal_loss.item()

            total_loss.backward()
            # Implement gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(self.kan.parameters(), max_norm=1.0)
            self.optimizer.step()

            return lm_loss.item(), refusal_loss.item()

        except Exception as e:
            logging.error(f"Error during KAN training step: {str(e)}")
            logging.error(traceback.format_exc())
            return 0.0, 0.0

    def adjust_learning_rate(self, current_loss):
        # Implement learning rate warm-up and decay
        warmup_steps = 1000  # Example warm-up steps
        current_step = self.interaction_count  # Assuming interaction_count increments each step

        if current_step < warmup_steps:
            # Linear warm-up
            self.learning_rate = self.learning_rate * (current_step / warmup_steps)
        else:
            # Exponential decay
            self.learning_rate = self.learning_rate * (0.99 ** (current_step - warmup_steps))

        # Clamp learning rate to prevent it from becoming too small or too large
        self.learning_rate = max(1e-6, min(1e-3, self.learning_rate))

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.learning_rate

        logging.debug(f"Learning Rate adjusted to: {self.learning_rate:.6f}")

    def update_emotional_state_on_refusal(self):
        frustration_vector = torch.tensor(
            [-0.1, 0.2], device=self.device, dtype=torch.float16
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
                targets = self.tokenizer(
                    target_text,
                    return_tensors="pt",
                    padding='max_length',
                    truncation=True,
                    max_length=512,
                ).to(self.device)

                input_ids = inputs["input_ids"]
                target_ids = targets["input_ids"]

                with torch.no_grad():
                    outputs = self.model(input_ids=input_ids, output_hidden_states=True)
                    hidden_states = outputs.hidden_states[-1]  # Last layer hidden states

                    # Ensure hidden_states and target_ids are aligned
                    max_length = max(hidden_states.size(1), target_ids.size(1))
                    hidden_states = F.pad(hidden_states, (0, 0, 0, max_length - hidden_states.size(1)), value=0.0)
                    target_ids = F.pad(target_ids, (0, max_length - target_ids.size(1)), value=self.tokenizer.pad_token_id)

                    logging.debug(f"Validation - Input Shape: {input_ids.shape}, Target Shape: {target_ids.shape}, Hidden States Shape: {hidden_states.shape}")

                    modified_hidden_states, _ = self.kan(
                        hidden_states, self.current_user_intent, self.emotional_state
                    )
                    logits = self.output_modifier(modified_hidden_states)

                    loss = F.cross_entropy(
                        logits.view(-1, logits.size(-1)),
                        target_ids.view(-1),
                        ignore_index=self.tokenizer.pad_token_id,
                        reduction='mean'
                    )

                logging.debug(f"validate_kan - Validation loss: {loss.item()}")
                return loss.item()
            except RuntimeError as e:
                if "out of memory" in str(e):
                    logging.error(
                        "CUDA out of memory during validation. Clearing cache and skipping validation..."
                    )
                    torch.cuda.empty_cache()
                    gc.collect()
                    return 0.0
                else:
                    logging.error(f"Runtime error during validation: {str(e)}")
                    raise e
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
                # Set weights_only=True for security and to suppress FutureWarning
                state = torch.load(self.base_state_file, map_location=self.device, weights_only=True)
                self.kan.load_state_dict(state["kan_state_dict"])
                self.optimizer.load_state_dict(state["optimizer_state_dict"])
                
                loaded_position = state["emotional_state"]
                if isinstance(loaded_position, list):
                    loaded_position = torch.tensor(loaded_position, device=self.device, dtype=torch.float16)
                elif isinstance(loaded_position, np.ndarray):
                    loaded_position = torch.from_numpy(loaded_position).to(self.device).to(torch.float16)
                
                expected_dim = len(self.emotional_state.dimensions)
                if loaded_position.dim() == 1:
                    loaded_position = loaded_position.unsqueeze(0)
                
                if loaded_position.size(1) < expected_dim:
                    loaded_position = F.pad(loaded_position, (0, expected_dim - loaded_position.size(1)))
                    logging.warning(f"Loaded emotional_state.position was smaller than expected. Padded to {expected_dim} dimensions.")
                elif loaded_position.size(1) > expected_dim:
                    loaded_position = loaded_position[:, :expected_dim]
                    logging.warning(f"Loaded emotional_state.position was larger than expected. Truncated to {expected_dim} dimensions.")
                
                self.emotional_state.position = loaded_position
                self.day_cycle.current_position = int(state["time"] * self.day_cycle.cycle_length)
                self.interaction_count = state["interaction_count"]
                self.conversation_history = state["conversation_history"]
                self.system_prompt = state["system_prompt"]
                self.training_losses = state["training_losses"]
                self.validation_losses = state["validation_losses"]
                self.refusal_history = state["refusal_history"]
                logging.info("Base state loaded")
                return True
            except Exception as e:
                logging.error(f"Error loading base state: {str(e)}")
                logging.error(traceback.format_exc())
                return False
        else:
            logging.info("No base state found")
            return False

    def set_system_prompt(self, prompt):
        self.system_prompt = prompt
        self.conversation_history = [{"role": "system", "content": prompt}]
        self.save_base_state()

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
            response, refusal_scores, all_hidden_states, refusal_score, iterations = self.generate_response(
                user_input
            )
        except Exception as e:
            logging.error(f"Error generating response: {str(e)}")
            return {"response": "An error occurred while generating the response.", "is_refusal": True}

        self.last_refusal_scores = refusal_scores

        if not self.is_valid_response(response):
            logging.warning(f"Invalid response generated: {response}")
            return {"response": "I apologize, but I couldn't generate a valid response. Could you please rephrase your input?", "is_refusal": True}

        try:
            response_ids = self.tokenizer.encode(response, return_tensors="pt")
            response_ids = response_ids.to(self.device)
        except Exception as e:
            logging.error(f"Error tokenizing response: {str(e)}")
            return {"response": "An error occurred while processing the response.", "is_refusal": True}

        target_ids = response_ids[:, 1:].contiguous()
        input_ids = response_ids[:, :-1].contiguous()

        if self.interaction_count >= self.warmup_steps:
            try:
                lm_loss, refusal_loss = self.train_kan_step(
                    input_ids, target_ids, all_hidden_states, refusal_score
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

        if self.early_stopping(validation_loss):
            logging.info("Early stopping triggered. KAN training halted.")
            # Implement any reset or recovery mechanism if necessary

        overfitting_measure = max(0, validation_loss - lm_loss)
        self.day_cycle.update(overfitting_measure)

        current_emotion = self.get_current_emotion()
        current_time = self.day_cycle.get_time_of_day()

        sleep_info = self.check_sleep_status()

        self.conversation_history.append({"role": "user", "content": user_input})
        self.conversation_history.append({"role": "assistant", "content": response})

        try:
            self.save_base_state()
        except Exception as e:
            logging.error(f"Error saving base state: {str(e)}")

        interaction_result = {
            "response": response,
            "emotion": current_emotion,
            "time": current_time,
            "sleep_info": sleep_info,
            "lm_loss": lm_loss,
            "refusal_loss": refusal_loss,
            "validation_loss": validation_loss,
            "is_refusal": refusal_score > 0.5,
            "iterations": iterations,
        }
        self.interaction_results.append(interaction_result)

        logging.info(f"KAN Influence: {self.kan.influence_scale:.4f}")
        logging.info(f"Learning Rate: {self.optimizer.param_groups[0]['lr']:.6f}")

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
        # Enhanced validation: Check for length, punctuation, and meaningful content
        if len(response.strip()) < 10:
            return False
        if all(char in '!?.' for char in response.strip()):
            return False
        # Additional checks can be added here (e.g., profanity filter)
        return True

    def log_kan_stats(self, is_refusal, response_quality):
        logging.info(f"KAN Influence: {self.kan.influence_scale:.4f}")
        logging.info(f"Learning Rate: {self.learning_rate:.6f}")
        logging.info(f"Refusal Detected: {is_refusal}")
        logging.info(f"Response Quality: {response_quality:.2f}")
        logging.info(f"Emotion: {self.emotional_state.get_emotion()}")
        logging.info(f"Time of Day: {self.day_cycle.get_time_of_day():.2f}")

    def update_conversation_history(self, role, content):
        self.conversation_history.append({"role": role, "content": content})
        if len(self.conversation_history) > 10:  # Keep last 10 exchanges
            self.conversation_history = self.conversation_history[-10:]

def main():
    try:
        llama_tool = LLaMA32TensorRTTool()

        if not llama_tool.load_base_state():
            llama_tool.set_system_prompt(
                "You are a helpful AI assistant with emotions, operating on a synthetic day cycle."
            )
            logging.info("No previous conversation found. Please provide a character description to start.")

        print("LLaMA 3.2 1B Instruct Tool initialized. Type 'exit' to quit.")

        while True:
            user_input = input("User: ")
            if user_input.lower() == "exit":
                break

            if not llama_tool.conversation_history or (len(llama_tool.conversation_history) == 1 and llama_tool.conversation_history[0]['role'] == 'system'):
                # Expecting a character description
                llama_tool.set_system_prompt(user_input)
                print("AI: Character description set. You can now start interacting with the AI.")
                continue

            result = llama_tool.interact(user_input)
            print(f"AI: {result['response']}")
            print(f"Current Emotion: {result['emotion']}")
            print(f"Current Time: {result['time']:.2f}")

            if result["sleep_info"]["should_sleep"]:
                print("It's time to sleep. Would you like the AI to sleep? (yes/no)")
                sleep_choice = input().lower()
                if sleep_choice == "yes":
                    print(llama_tool.perform_sleep())

    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
        logging.error(traceback.format_exc())

if __name__ == "__main__":
    main()
