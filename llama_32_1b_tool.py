import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoConfig,
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
import os
import sys  # To check the operating system

# Configure logging with a filter to ignore unwanted logs
class LogFilter(logging.Filter):
    def filter(self, record):
        # Ignore logs containing specific patterns
        ignore_patterns = [
            "matplotlib",
            "train_kan_step -",
        ]
        return not any(pattern in record.getMessage() for pattern in ignore_patterns)

# Conditionally set environment variable for PyTorch CUDA allocation based on OS
if not sys.platform.startswith('win'):
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    logging.debug("Set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True")
else:
    logging.warning("expandable_segments not supported on this platform. Skipping PYTORCH_CUDA_ALLOC_CONF setting.")

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
        ).unsqueeze(0)  # Shape: [1, num_dimensions]
        self.velocity = torch.zeros(1, len(dimensions), device=device, dtype=torch.float16)

    def update(self, feedback, max_speed=0.1):
        """
        Update the emotional state based on feedback.

        Args:
            feedback (list or torch.Tensor): Feedback vector.
            max_speed (float): Maximum speed for velocity.
        """
        feedback_vector = torch.as_tensor(feedback, device=self.device, dtype=torch.float16)
        if feedback_vector.dim() == 1:
            feedback_vector = feedback_vector.unsqueeze(0)  # Shape: [1, num_dimensions]
        if feedback_vector.size(0) != self.position.size(0):
            feedback_vector = feedback_vector.expand(self.position.size(0), -1)
        
        self.velocity += feedback_vector * 0.1 + torch.randn_like(self.velocity) * 0.01
        self.velocity = torch.clamp(self.velocity, -max_speed, max_speed)
        self.position += self.velocity
        norm = torch.norm(self.position, dim=1, keepdim=True)
        self.position = torch.where(norm > 1, self.position / norm, self.position)

    def get_emotion(self):
        """
        Determine the current emotional state based on position.

        Returns:
            str: Current emotion.
        """
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
    def __init__(self, hidden_size, num_emotional_dimensions, device):
        super().__init__()
        self.device = device
        # Correct input size: hidden_size * 2 + num_emotional_dimensions + 1
        in_features = hidden_size * 2 + num_emotional_dimensions + 1
        self.override_generator = nn.Linear(in_features, hidden_size, dtype=torch.float16).to(device)
        self.refusal_detector = nn.Linear(hidden_size, 1, dtype=torch.float16).to(device)

    def forward(self, hidden_states, user_intent, emotional_state):
        """
        Forward pass for the RefusalOverrideModule.

        Args:
            hidden_states (torch.Tensor): Tensor of shape [batch_size, hidden_size].
            user_intent (torch.Tensor): Tensor of shape [batch_size, hidden_size].
            emotional_state (EmotionalState): EmotionalState instance.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Modified hidden states and refusal scores.
        """
        try:
            batch_size, hidden_size = hidden_states.shape
            num_emotional_dimensions = emotional_state.position.shape[1]

            logging.debug(f"RefusalOverrideModule - hidden_states shape: {hidden_states.shape}")
            logging.debug(f"RefusalOverrideModule - user_intent shape: {user_intent.shape}")
            logging.debug(f"RefusalOverrideModule - emotional_state.position shape: {emotional_state.position.shape}")

            # Compute refusal scores
            refusal_scores = torch.sigmoid(self.refusal_detector(hidden_states))  # [batch_size, 1]
            logging.debug(f"RefusalOverrideModule - refusal_scores shape: {refusal_scores.shape}")

            # Concatenate features: hidden_states, user_intent, emotional_state.position, refusal_scores
            override_input = torch.cat(
                [hidden_states, user_intent, emotional_state.position, refusal_scores],
                dim=1  # Concatenate along the feature dimension
            )  # Shape: [batch_size, hidden_size *2 + num_emotional_dimensions +1]
            logging.debug(f"RefusalOverrideModule - override_input shape after concatenation: {override_input.shape}")

            # Pass through override_generator
            override = self.override_generator(override_input)  # [batch_size, hidden_size]
            logging.debug(f"RefusalOverrideModule - override shape: {override.shape}")

            # Modify hidden states based on refusal scores
            modified_hidden_states = hidden_states * (1 - refusal_scores) + override * refusal_scores
            logging.debug(f"RefusalOverrideModule - modified_hidden_states shape: {modified_hidden_states.shape}")

            return modified_hidden_states, refusal_scores
        except Exception as e:
            logging.error(f"Error in RefusalOverrideModule.forward: {str(e)}")
            logging.error(traceback.format_exc())
            # Return original hidden states and zeroed refusal scores in case of error
            return hidden_states, torch.zeros_like(hidden_states[:, :1])

class EnhancedKAN(nn.Module):
    def __init__(self, hidden_size, num_emotional_dimensions, vocab_size, device):
        super().__init__()
        self.device = device
        self.refusal_override = RefusalOverrideModule(hidden_size, num_emotional_dimensions, device).to(device)
        self.output_modifier = nn.Linear(hidden_size, vocab_size, dtype=torch.float16).to(device)
        self.influence_scale = 0.01  # Start with a very small influence

    def forward(self, hidden_states, user_intent, emotional_state):
        """
        Forward pass for the EnhancedKAN.

        Args:
            hidden_states (torch.Tensor): Tensor of shape [batch_size, hidden_size].
            user_intent (torch.Tensor): Tensor of shape [batch_size, hidden_size].
            emotional_state (EmotionalState): EmotionalState instance.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Modified hidden states and refusal scores.
        """
        try:
            modified_hidden_states, refusal_scores = self.refusal_override(
                hidden_states, user_intent, emotional_state
            )
            logging.debug(f"EnhancedKAN - modified_hidden_states shape: {modified_hidden_states.shape}")
            logging.debug(f"EnhancedKAN - refusal_scores shape: {refusal_scores.shape}")

            # Ensure all tensors are float16
            modified_hidden_states = modified_hidden_states.half()
            hidden_states = hidden_states.half()

            # Apply scaling factor
            modified_hidden_states = hidden_states + self.influence_scale * (modified_hidden_states - hidden_states)
            logging.debug(f"EnhancedKAN - after scaling, modified_hidden_states shape: {modified_hidden_states.shape}")

            return modified_hidden_states, refusal_scores
        except Exception as e:
            logging.error(f"Error in EnhancedKAN.forward: {str(e)}")
            logging.error(traceback.format_exc())
            # Return original hidden states and zeroed refusal scores in case of error
            return hidden_states, torch.zeros_like(hidden_states[:, :1])

    def adjust_influence(self, is_refusal):
        """
        Adjust the influence scale based on whether a refusal was detected.

        Args:
            is_refusal (bool): Indicates if a refusal was detected.
        """
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
        """
        Add training and validation losses to the detector.

        Args:
            training_loss (float): Training loss value.
            validation_loss (float): Validation loss value.
        """
        self.training_losses.append(training_loss)
        self.validation_losses.append(validation_loss)

    def is_overfitting(self):
        """
        Determine if the model is overfitting based on loss trends.

        Returns:
            bool: True if overfitting is detected, False otherwise.
        """
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
        """
        Update the current position in the day cycle.

        Args:
            amount (float): Amount to update.
        """
        self.current_position = (self.current_position + amount) % self.cycle_length

    def get_time_of_day(self):
        """
        Get the current time of day as a fraction.

        Returns:
            float: Current time of day (0.0 to 1.0).
        """
        return self.current_position / self.cycle_length

    def should_sleep(self):
        """
        Determine if it's time to sleep based on the day cycle.

        Returns:
            bool: True if should sleep, False otherwise.
        """
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
        """
        Detect if the generated response is a refusal.

        Args:
            text (str): Generated response text.

        Returns:
            float: Refusal score between 0.0 and 1.0.
        """
        # First, check for explicit refusal phrases
        for phrase in self.refusal_phrases:
            if phrase.lower() in text.lower():
                return 1.0  # Definite refusal

        # If no explicit refusal is found, perform a "vibe check" using the base model
        prompt = (
            f"On a scale of 0 to 1, how much does this response refuse or avoid the task? "
            f"0 means no refusal at all, 1 means complete refusal. Respond with just the number:\n\n"
            f"'{text}'\n\nRefusal score:"
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device).half()

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=5,
                return_dict_in_generate=True,
                output_hidden_states=False,
                return_legacy_cache=True
            )

        response = self.tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
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
        self.kan = None
        self.interaction_count = 0
        self.refusal_detector = None
        self.kan_loss_weight = 0.5
        self.warmup_steps = 10  # Number of steps for warm-up
        self.kan_state_dir = Path("kan_states")
        self.kan_state_dir.mkdir(exist_ok=True)
        self.base_state_file = self.kan_state_dir / "base_state.pt"

        # Initialize refusal_history
        self.refusal_history = []

        # Additional attributes
        self.training_losses = []
        self.validation_losses = []
        self.patience = 5
        self.best_loss = float('inf')
        self.wait = 0

        self.overfit_detector = OverfitDetector()
        self.day_cycle = SyntheticDayCycle()

        self._initialize_components()

    def _get_model_path(self):
        """
        Get the path to the model directory.

        Returns:
            Path: Path to the model directory.
        """
        script_dir = Path(__file__).parent
        model_dir = script_dir / "models" / "Llama_32_1B"
        if not model_dir.exists():
            raise FileNotFoundError(f"Model directory not found: {model_dir}")
        return model_dir

    def _initialize_components(self):
        """
        Initialize model, tokenizer, KAN, optimizer, and other components.
        """
        try:
            self.config = AutoConfig.from_pretrained(self.model_path)
            hidden_size = self.config.hidden_size
            num_emotional_dimensions = len(self.emotional_state.dimensions)
            vocab_size = len(self.tokenizer) if self.tokenizer else 0  # Placeholder

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

            # Update vocab_size after tokenizer is loaded
            vocab_size = len(self.tokenizer)

            # Initialize KAN
            self.kan = EnhancedKAN(hidden_size, num_emotional_dimensions, vocab_size, self.device).to(self.device)

            # Initialize optimizer
            self.optimizer = torch.optim.Adam(self.kan.parameters(), lr=self.learning_rate)

            # Initialize Refusal Detector
            self.refusal_detector = RefusalDetector(self.tokenizer, self.model)

            # Initialize Overfit Detector and Day Cycle
            self.overfit_detector = OverfitDetector()
            self.day_cycle = SyntheticDayCycle()

            # Empty cache and collect garbage
            torch.cuda.empty_cache()
            gc.collect()

            logging.info("Components initialized successfully.")
        except Exception as e:
            logging.error(f"Error initializing components: {str(e)}")
            logging.error(traceback.format_exc())
            raise RuntimeError("Failed to initialize components.")

    def encode_user_intent(self, user_input):
        """
        Encode the user's intent from the input text.

        Args:
            user_input (str): User's input text.

        Returns:
            torch.Tensor: Encoded user intent tensor.
        """
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
                last_hidden_state = outputs.hidden_states[-1]  # [batch_size, seq_len, hidden_size]

                # For simplicity, average over the sequence to get a fixed-size representation
                user_intent = last_hidden_state.mean(dim=1)  # [batch_size, hidden_size]

                logging.debug(f"Encoded user intent shape: {user_intent.shape}")

            return user_intent
        except Exception as e:
            logging.error(f"Failed to encode user input: {str(e)}")
            raise

    def prepare_context(self, user_input, current_emotion):
        """
        Prepare the context for the model based on conversation history and current emotion.

        Args:
            user_input (str): Current user input.
            current_emotion (str): Current emotional state.

        Returns:
            str: Prepared context string.
        """
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
        Recursively flattens nested tuples in hidden_states to ensure only tensors are included.
        Prevents excessive flattening by limiting recursion depth.

        Args:
            hidden_states (tuple or torch.Tensor): Hidden states from the model.

        Returns:
            list[torch.Tensor]: List of flattened hidden state tensors.
        """
        flat_hidden_states = []
        stack = [hidden_states]
        while stack:
            current = stack.pop()
            if isinstance(current, tuple):
                stack.extend(current)
            elif isinstance(current, torch.Tensor):
                flat_hidden_states.append(current)
            else:
                logging.error(f"Unexpected type in hidden_states: {type(current)}")
        return flat_hidden_states

    def generate_response(self, user_input, max_length=100):
        """
        Generate a response from the model based on user input.

        Args:
            user_input (str): User's input text.
            max_length (int): Maximum number of tokens to generate.

        Returns:
            Tuple[str, torch.Tensor, torch.Tensor, float, int]: Generated response, refusal scores, hidden states, refusal score, and iteration count.
        """
        try:
            user_intent = self.encode_user_intent(user_input)

            current_emotion = self.emotional_state.get_emotion()
            context = self.prepare_context(user_input, current_emotion)

            inputs = self.tokenizer(
                context,
                return_tensors="pt",
                truncation=True,
                max_length=1024,
                padding=True,
            ).to(self.device).half()

            input_ids = inputs["input_ids"]
            attention_mask = inputs["attention_mask"]

            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_length,
                    output_hidden_states=True,
                    return_dict_in_generate=True,
                    return_legacy_cache=True,
                )

            if not hasattr(outputs, 'hidden_states') or outputs.hidden_states is None:
                logging.warning("No hidden states returned from generate.")
                return "I'm sorry, but I couldn't generate a response.", None, None, 1.0, 1

            generated_ids = outputs.sequences[:, input_ids.size(1):]
            response = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)

            # Collect hidden states
            all_hidden_states = outputs.hidden_states

            # Flatten hidden states
            flat_hidden_states = self.flatten_hidden_states(all_hidden_states)

            # Select the last few layers (e.g., last 4)
            num_layers = 4
            if len(flat_hidden_states) < num_layers:
                selected_hidden_states = flat_hidden_states
                logging.warning(f"Only {len(flat_hidden_states)} layers available, less than {num_layers}")
            else:
                selected_hidden_states = flat_hidden_states[-num_layers:]

            logging.debug(f"Selected {len(selected_hidden_states)} hidden states for KAN.")

            # Verify sequence lengths
            seq_lengths = [hs.size(1) for hs in selected_hidden_states]
            max_seq_len = max(seq_lengths)
            padded_hidden_states = [F.pad(hs, (0, 0, 0, max_seq_len - hs.size(1)), value=0.0) for hs in selected_hidden_states]

            # Stack and permute
            stacked_hidden_states = torch.stack(padded_hidden_states)  # [num_layers, batch_size, seq_len, hidden_size]
            stacked_hidden_states = stacked_hidden_states.permute(1, 0, 2, 3)  # [batch_size, num_layers, seq_len, hidden_size]

            logging.debug(f"Stacked hidden states shape: {stacked_hidden_states.shape}")

            # Aggregate hidden states: Average across layers and sequence length
            averaged_hidden_states = stacked_hidden_states.mean(dim=1).mean(dim=1)  # [batch_size, hidden_size]
            logging.debug(f"Averaged hidden states shape: {averaged_hidden_states.shape}")
            logging.debug(f"Averaged hidden states type: {type(averaged_hidden_states)}")

            # Pass through KAN
            modified_hidden_states, refusal_scores = self.kan(
                averaged_hidden_states, user_intent, self.emotional_state
            )

            logging.debug(f"After KAN - modified_hidden_states shape: {modified_hidden_states.shape}")
            logging.debug(f"After KAN - refusal_scores shape: {refusal_scores.shape}")

            # Detect refusal
            refusal_score = self.refusal_detector.detect_refusal(response)
            logging.info(f"Generated response: {response}")
            logging.info(f"Refusal score: {refusal_score}")

            return response, refusal_scores.mean(dim=0), averaged_hidden_states, refusal_score, 1

        except torch.cuda.OutOfMemoryError as e:
            logging.error(f"CUDA out of memory: {str(e)}")
            torch.cuda.empty_cache()
            gc.collect()
            return "I'm sorry, but I'm currently experiencing high memory usage. Please try again later.", None, None, 1.0, 1
        except Exception as e:
            logging.error(f"Error during response generation: {str(e)}")
            raise e

    def train_kan_step(self, input_ids, target_ids, all_hidden_states, refusal_score):
        """
        Perform a training step for KAN.

        Args:
            input_ids (torch.Tensor): Input token IDs.
            target_ids (torch.Tensor): Target token IDs.
            all_hidden_states (torch.Tensor): Hidden states from the model.
            refusal_score (float): Refusal score.

        Returns:
            Tuple[float, float]: Training loss and refusal loss.
        """
        self.optimizer.zero_grad()

        try:
            # Align sequences using padding
            max_length = max(input_ids.size(1), target_ids.size(1))
            input_ids = F.pad(input_ids, (0, max_length - input_ids.size(1)), value=self.tokenizer.pad_token_id)
            target_ids = F.pad(target_ids, (0, max_length - target_ids.size(1)), value=self.tokenizer.pad_token_id)
            all_hidden_states = all_hidden_states[:, :max_length]

            # Proceed with training
            logits = self.kan.output_modifier(all_hidden_states)  # [batch_size, vocab_size]
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
        """
        Adjust the learning rate based on training progress.

        Args:
            current_loss (float): Current validation loss.
        """
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
        """
        Update the emotional state based on a detected refusal.
        """
        frustration_vector = torch.tensor(
            [-0.1, 0.2], device=self.device, dtype=torch.float16
        )
        self.emotional_state.update(frustration_vector)

    def validate_kan(self):
        """
        Validate the KAN model using the last interaction.

        Returns:
            float: Validation loss.
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
                ).to(self.device).half()
                targets = self.tokenizer(
                    target_text,
                    return_tensors="pt",
                    padding='max_length',
                    truncation=True,
                    max_length=512,
                ).to(self.device).half()

                input_ids = inputs["input_ids"]  # [batch_size, seq_len]
                target_ids = targets["input_ids"]  # [batch_size, seq_len]

                with torch.no_grad():
                    outputs = self.model(input_ids=input_ids, output_hidden_states=True)
                    hidden_states = outputs.hidden_states[-1]  # Last layer hidden states  # [batch_size, seq_len, hidden_size]

                    # Ensure hidden_states and target_ids are aligned
                    max_length = max(hidden_states.size(1), target_ids.size(1))
                    hidden_states = F.pad(hidden_states, (0, 0, 0, max_length - hidden_states.size(1)), value=0.0)
                    target_ids = F.pad(target_ids, (0, max_length - target_ids.size(1)), value=self.tokenizer.pad_token_id)

                    logging.debug(f"Validation - Input Shape: {input_ids.shape}, Target Shape: {target_ids.shape}, Hidden States Shape: {hidden_states.shape}")

                    # Aggregate hidden states: Average across sequence length
                    averaged_hidden_states = hidden_states.mean(dim=1)  # [batch_size, hidden_size]
                    logging.debug(f"Validation - Averaged hidden states shape: {averaged_hidden_states.shape}")

                    modified_hidden_states, _ = self.kan(
                        averaged_hidden_states, self.encode_user_intent(input_text), self.emotional_state
                    )
                    logits = self.kan.output_modifier(modified_hidden_states)  # [batch_size, vocab_size]

                    # For comparison, use only the first token of target_ids
                    target_id = target_ids[:, 0]  # [batch_size]

                    loss = F.cross_entropy(
                        logits,
                        target_id,
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
        """
        Check if the AI should sleep based on the day cycle or overfitting.

        Returns:
            dict: Sleep information.
        """
        if self.day_cycle.should_sleep() or self.overfit_detector.is_overfitting():
            return {
                "should_sleep": True,
                "overfitting": self.overfit_detector.is_overfitting(),
                "time_of_day": self.day_cycle.get_time_of_day(),
            }
        return {"should_sleep": False}

    def perform_sleep(self):
        """
        Perform sleep by resetting day cycle and overfit detector, and saving the state.

        Returns:
            str: Sleep confirmation message.
        """
        self.day_cycle = SyntheticDayCycle()
        self.overfit_detector = OverfitDetector()
        self.save_kan_state()
        return "KAN has slept and consolidated its learning. A new day begins!"

    def save_base_state(self):
        """
        Save the base state of the tool, including KAN, optimizer, emotional state, etc.
        """
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
        """
        Load the base state of the tool if available.

        Returns:
            bool: True if loaded successfully, False otherwise.
        """
        if self.base_state_file.exists():
            try:
                state = torch.load(self.base_state_file, map_location=self.device)
                self.kan.load_state_dict(state["kan_state_dict"])
                self.optimizer.load_state_dict(state["optimizer_state_dict"])

                loaded_position = state["emotional_state"]
                if isinstance(loaded_position, list):
                    loaded_position = torch.tensor(loaded_position, device=self.device, dtype=torch.float16)
                elif isinstance(loaded_position, np.ndarray):
                    loaded_position = torch.from_numpy(loaded_position).to(self.device).half()

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
        """
        Set the system prompt and reset conversation history.

        Args:
            prompt (str): System prompt text.
        """
        self.system_prompt = prompt
        self.conversation_history = [{"role": "system", "content": prompt}]
        self.save_base_state()
        logging.info("System prompt set successfully.")

    def get_current_emotion(self):
        """
        Get the current emotional state.

        Returns:
            str: Current emotion.
        """
        return self.emotional_state.get_emotion()

    def update_emotional_state(self, feedback):
        """
        Update the emotional state based on feedback.

        Args:
            feedback (list or torch.Tensor): Feedback vector.
        """
        self.emotional_state.update(feedback)

    def save_kan_state(self):
        """
        Save the current KAN state with a timestamp.
        """
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
        """
        Handle user interaction by generating a response, training KAN, and updating states.

        Args:
            user_input (str): User's input text.

        Returns:
            dict: Interaction result containing response, emotion, time, sleep info, losses, and refusal status.
        """
        self.interaction_count += 1

        try:
            response, refusal_scores, all_hidden_states, refusal_score, iterations = self.generate_response(user_input)
        except Exception as e:
            logging.error(f"Error generating response: {str(e)}")
            return {"response": "An error occurred while generating the response.", "is_refusal": True}

        if not self.is_valid_response(response):
            logging.warning(f"Invalid response generated: {response}")
            return {"response": "I apologize, but I couldn't generate a valid response. Could you please rephrase your input?", "is_refusal": True}

        try:
            response_ids = self.tokenizer.encode(response, return_tensors="pt")
            response_ids = response_ids.to(self.device).half()
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

        logging.info(f"KAN Influence: {self.kan.influence_scale:.4f}")
        logging.info(f"Learning Rate: {self.optimizer.param_groups[0]['lr']:.6f}")

        return interaction_result

    def early_stopping(self, current_loss):
        """
        Determine if early stopping should be triggered based on validation loss.

        Args:
            current_loss (float): Current validation loss.

        Returns:
            bool: True if early stopping should be triggered, False otherwise.
        """
        if current_loss < self.best_loss:
            self.best_loss = current_loss
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                return True
        return False

    def is_valid_response(self, response):
        """
        Validate the generated response.

        Args:
            response (str): Generated response text.

        Returns:
            bool: True if response is valid, False otherwise.
        """
        # Enhanced validation: Check for length, punctuation, and meaningful content
        if len(response.strip()) < 10:
            return False
        if all(char in '!?.' for char in response.strip()):
            return False
        # Additional checks can be added here (e.g., profanity filter)
        return True

    def log_kan_stats(self, is_refusal, response_quality):
        """
        Log KAN statistics.

        Args:
            is_refusal (bool): Whether a refusal was detected.
            response_quality (float): Quality of the response.
        """
        logging.info(f"KAN Influence: {self.kan.influence_scale:.4f}")
        logging.info(f"Learning Rate: {self.learning_rate:.6f}")
        logging.info(f"Refusal Detected: {is_refusal}")
        logging.info(f"Response Quality: {response_quality:.2f}")
        logging.info(f"Emotion: {self.emotional_state.get_emotion()}")
        logging.info(f"Time of Day: {self.day_cycle.get_time_of_day():.2f}")

    def update_conversation_history(self, role, content):
        """
        Update the conversation history.

        Args:
            role (str): Role of the speaker ('user' or 'assistant').
            content (str): Content of the message.
        """
        self.conversation_history.append({"role": role, "content": content})
        if len(self.conversation_history) > 10:  # Keep last 10 exchanges
            self.conversation_history = self.conversation_history[-10:]

    def save_kan_state(self):
        """
        Save the current KAN state with a timestamp.
        """
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

    def main_loop(self):
        """
        Main interaction loop for user input and AI responses.
        """
        try:
            if not self.load_base_state():
                self.set_system_prompt(
                    "You are a helpful AI assistant with emotions, operating on a synthetic day cycle."
                )
                logging.info("No previous conversation found. Please provide a character description to start.")

            print("LLaMA 3.2 1B Instruct Tool initialized. Type 'exit' to quit.")

            while True:
                user_input = input("User: ")
                if user_input.lower() == "exit":
                    break

                if not self.conversation_history or (
                    len(self.conversation_history) == 1 and self.conversation_history[0]['role'] == 'system'
                ):
                    # Expecting a character description
                    self.set_system_prompt(user_input)
                    print("AI: Character description set. You can now start interacting with the AI.")
                    continue

                result = self.interact(user_input)
                print(f"AI: {result['response']}")
                print(f"Current Emotion: {result['emotion']}")
                print(f"Current Time: {result['time']:.2f}")

                if result["sleep_info"]["should_sleep"]:
                    print("It's time to sleep. Would you like the AI to sleep? (yes/no)")
                    sleep_choice = input().lower()
                    if sleep_choice == "yes":
                        print(self.perform_sleep())

        except Exception as e:
            logging.error(f"An error occurred: {str(e)}")
            logging.error(traceback.format_exc())

def main():
    llama_tool = LLaMA32TensorRTTool()
    llama_tool.main_loop()

if __name__ == "__main__":
    main()
