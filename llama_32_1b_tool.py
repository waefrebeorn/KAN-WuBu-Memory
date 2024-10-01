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

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("llama_tool.log"),
        logging.StreamHandler(),
    ],
)

# Suppress matplotlib debug logs
logging.getLogger('matplotlib').setLevel(logging.WARNING)

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
        feedback_vector = torch.tensor(feedback, device=self.device, dtype=torch.float16).unsqueeze(0)
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

    def forward(self, hidden_states, user_intent, emotional_state):
        modified_hidden_states, refusal_scores = self.refusal_override(
            hidden_states, user_intent, emotional_state
        )
        return modified_hidden_states, refusal_scores

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
                return True

        # If no explicit refusal is found, perform a "vibe check" using the base model
        prompt = f"Determine if the following text contains a refusal or unwillingness to perform a task. Respond with 'Yes' if it's a refusal, or 'No' if it's not:\n\n'{text}'\n\nRefusal:"
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=5)
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return "yes" in response.lower()


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
                self.config.vocab_size,
                dtype=torch.float16,
            ).to(self.device)

            # Check and reinitialize output modifier if necessary
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
        for message in self.conversation_history[-5:]:
            context += f"{message['role'].capitalize()}: {message['content']}\n"
        context += f"Human: {user_input}\nAI: "

        return context

    def generate_response(self, user_input, max_length=150):
        self.current_user_intent = self.encode_user_intent(user_input)
        
        for iteration in range(self.max_iterations):
            current_emotion = self.emotional_state.get_emotion()
            context = self.prepare_context(user_input, current_emotion)
    
            try:
                inputs = self.tokenizer(
                    context,
                    return_tensors="pt",
                    truncation=True,
                    max_length=1024,
                    padding=True,
                )
    
                inputs = {
                    k: v.to(self.device).half() if v.dtype in [torch.float16, torch.float32] else v.to(self.device)
                    for k, v in inputs.items()
                }
    
            except Exception as e:
                logging.error(f"Error tokenizing context: {str(e)}")
                raise
    
            input_ids = inputs["input_ids"]
            attention_mask = inputs["attention_mask"]
    
            generated_tokens = []
            all_hidden_states = []
            all_refusal_scores = []
            
            self.kan.train()
            for token_index in range(max_length):
                try:
                    with torch.no_grad():
                        outputs = self.model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            output_hidden_states=True,
                        )
    
                    hidden_states = outputs.hidden_states[-1]
                    batch_size, seq_len, hidden_size = hidden_states.shape
    
                    if self.current_user_intent.size(0) != batch_size:
                        self.current_user_intent = self.current_user_intent.expand(batch_size, -1)
    
                    modified_hidden_states, refusal_scores = self.kan(
                        hidden_states, self.current_user_intent, self.emotional_state
                    )
    
                    logits_modifier = self.output_modifier(modified_hidden_states)
    
                    last_token_logits = outputs.logits[:, -1, :].to(self.device)
                    last_token_refusal_score = refusal_scores[:, -1, :].to(self.device)
                    last_token_logits_modifier = logits_modifier[:, -1, :].to(self.device)
                    
                    # Ensure all tensors have the same shape
                    last_token_logits = last_token_logits.view(batch_size, -1)
                    last_token_refusal_score = last_token_refusal_score.view(batch_size, -1)
                    last_token_logits_modifier = last_token_logits_modifier.view(batch_size, -1)
    
                    modified_logits = last_token_logits + last_token_logits_modifier * last_token_refusal_score
    
                    next_token = torch.argmax(modified_logits, dim=-1)
                    next_token = next_token.view(batch_size, 1)  # Reshape to [batch_size, 1]
    
                    if next_token.item() == self.tokenizer.eos_token_id:
                        break
    
                    generated_tokens.append(next_token.item())
                    
                    input_ids = torch.cat([input_ids, next_token], dim=-1)
                    attention_mask = torch.cat([attention_mask, torch.ones((batch_size, 1), dtype=torch.long, device=self.device)], dim=-1)
    
                    all_hidden_states.append(hidden_states[:, -1, :])  # Only keep the last token's hidden state
                    all_refusal_scores.append(refusal_scores[:, -1, :])  # Only keep the last token's refusal score
    
                except RuntimeError as e:
                    logging.error(f"Runtime error during generation of token {token_index}: {str(e)}")
                    raise e
    
            response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            logging.info(f"Generated response: {response}")
    
            is_refusal = self.refusal_detector.detect_refusal(response)
            self.refusal_history.append(is_refusal)
            logging.info(f"Refusal detected: {is_refusal}")
    
            if not is_refusal:
                logging.info(f"Non-refusal response generated after {iteration + 1} iterations.")
                
                # Stack hidden states and refusal scores
                stacked_hidden_states = torch.stack(all_hidden_states)
                stacked_refusal_scores = torch.stack(all_refusal_scores)
                
                return (
                    response,
                    stacked_refusal_scores.mean(dim=0),
                    stacked_hidden_states,
                    is_refusal,
                    iteration + 1,
                )
    
            self.train_kan_step(
                input_ids, input_ids[:, 1:].contiguous(), torch.stack(all_hidden_states), is_refusal
            )
    
            self.update_emotional_state_on_refusal()
    
        return "I'm sorry, but I couldn't generate a suitable response.", None, None, True, self.max_iterations
        
 


 
    def train_kan_step(self, input_ids, target_ids, all_hidden_states, is_refusal):
        self.optimizer.zero_grad()
    
        total_loss = 0
        lm_losses = []
        refusal_losses = []
    
        logging.debug(f"train_kan_step - input_ids shape: {input_ids.shape}")
        logging.debug(f"train_kan_step - target_ids shape: {target_ids.shape}")
        logging.debug(f"train_kan_step - number of hidden_states: {len(all_hidden_states)}")
    
        batch_size, seq_len = input_ids.shape
        num_hidden_states = len(all_hidden_states)
    
        # Adjust hidden states or input/target ids if there's a mismatch
        if num_hidden_states != seq_len:
            logging.warning(f"Number of hidden_states ({num_hidden_states}) does not match sequence length ({seq_len}). Adjusting...")
            if num_hidden_states < seq_len:
                # Truncate input_ids and target_ids
                input_ids = input_ids[:, :num_hidden_states]
                target_ids = target_ids[:, :num_hidden_states]
                seq_len = num_hidden_states
            else:
                # Truncate hidden_states
                all_hidden_states = all_hidden_states[:seq_len]
    
        for i in range(seq_len):
            try:
                hidden_state = all_hidden_states[i].unsqueeze(0)  # Add batch dimension
                modified_hidden_states, refusal_scores = self.kan(
                    hidden_state, self.current_user_intent, self.emotional_state
                )
    
                logits = self.output_modifier(modified_hidden_states)
                logits = logits.view(-1, logits.size(-1))
                target = target_ids[:, i].view(-1)
    
                logging.debug(f"train_kan_step - logits shape: {logits.shape}")
                logging.debug(f"train_kan_step - target shape: {target.shape}")
    
                if target.max().item() >= self.config.vocab_size:
                    logging.error(f"train_kan_step - Target token index {target.max().item()} exceeds vocab size {self.config.vocab_size}")
                    continue
    
                lm_loss = F.cross_entropy(logits, target)
                lm_losses.append(lm_loss.item())
    
                refusal_loss = torch.mean(refusal_scores) if is_refusal else -torch.mean(refusal_scores)
                refusal_losses.append(refusal_loss.item())
    
                step_loss = lm_loss + self.kan_loss_weight * refusal_loss
                total_loss += step_loss
    
            except RuntimeError as e:
                logging.error(f"Runtime error during KAN training step at position {i}: {str(e)}")
                logging.error(f"Current tensor shapes:")
                logging.error(f"hidden_state: {hidden_state.shape}")
                logging.error(f"modified_hidden_states: {modified_hidden_states.shape}")
                logging.error(f"refusal_scores: {refusal_scores.shape}")
                logging.error(f"logits: {logits.shape}")
                logging.error(f"target: {target.shape}")
                continue  # Skip this step and continue with the next one
    
        if total_loss > 0:
            total_loss.backward()
            self.optimizer.step()
    
        torch.cuda.empty_cache()
        gc.collect()
    
        avg_lm_loss = np.mean(lm_losses) if lm_losses else 0.0
        avg_refusal_loss = np.mean(refusal_losses) if refusal_losses else 0.0
    
        logging.debug(f"train_kan_step - Average LM Loss: {avg_lm_loss}")
        logging.debug(f"train_kan_step - Average Refusal Loss: {avg_refusal_loss}")
    
        return avg_lm_loss, avg_refusal_loss
        

 
    def update_emotional_state_on_refusal(self):
        frustration_vector = torch.tensor(
            [-0.1, 0.2], device=self.device, dtype=torch.float16
        ).unsqueeze(0)
        self.emotional_state.update(frustration_vector)

    def validate_kan(self):
        if len(self.conversation_history) >= 2:
            last_interaction = self.conversation_history[-2:]
            input_ids = self.tokenizer.encode(last_interaction[0]["content"], return_tensors="pt")
            target_ids = self.tokenizer.encode(last_interaction[1]["content"], return_tensors="pt")

            input_ids = input_ids.to(self.device)
            target_ids = target_ids.to(self.device)

            try:
                with torch.no_grad():
                    outputs = self.model(input_ids=input_ids, output_hidden_states=True)
                    hidden_states = outputs.hidden_states[-1]
                    modified_hidden_states, _ = self.kan(
                        hidden_states, self.current_user_intent, self.emotional_state
                    )
                    logits = self.output_modifier(modified_hidden_states)
                    loss = F.cross_entropy(
                        logits.view(-1, logits.size(-1)), target_ids.view(-1)
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
                state = torch.load(self.base_state_file, map_location=self.device)
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
        
    def reset_base_state(self):
        if self.base_state_file.exists():
            try:
                self.base_state_file.unlink()
                logging.info("Base state has been reset successfully.")
                return True
            except Exception as e:
                logging.error(f"Error resetting base state: {str(e)}")
                logging.error(traceback.format_exc())
                return False
        else:
            logging.info("No base state file exists to reset.")
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
            response, refusal_scores, all_hidden_states, is_refusal, iterations = self.generate_response(
                user_input
            )
        except Exception as e:
            logging.error(f"Error generating response: {str(e)}")
            return {"response": "An error occurred while generating the response.", "is_refusal": True}
    
        self.last_refusal_scores = refusal_scores
    
        try:
            response_ids = self.tokenizer.encode(response, return_tensors="pt")
            response_ids = response_ids.to(self.device)
        except Exception as e:
            logging.error(f"Error tokenizing response: {str(e)}")
            return {"response": "An error occurred while processing the response.", "is_refusal": True}
    
        target_ids = response_ids[:, 1:].contiguous()
        input_ids = response_ids[:, :-1].contiguous()
    
        try:
            lm_loss, refusal_loss = self.train_kan_step(
                input_ids, target_ids, all_hidden_states, is_refusal
            )
        except Exception as e:
            logging.error(f"Error during KAN training step: {str(e)}")
            lm_loss, refusal_loss = 0.0, 0.0
    
        try:
            validation_loss = self.validate_kan()
        except Exception as e:
            logging.error(f"Error during KAN validation: {str(e)}")
            validation_loss = 0.0
    
        self.training_losses.append(lm_loss)
        self.validation_losses.append(validation_loss)
        self.overfit_detector.add_losses(lm_loss, validation_loss)
    
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
            "is_refusal": is_refusal,
            "iterations": iterations,
        }
        self.interaction_results.append(interaction_result)
    
        return interaction_result

def main():
    try:
        llama_tool = LLaMA32TensorRTTool()

        if not llama_tool.load_base_state():
            llama_tool.set_system_prompt(
                "You are a helpful AI assistant with emotions, operating on a synthetic day cycle."
            )

        print("LLaMA 3.2 1B Instruct Tool initialized. Type 'exit' to quit.")

        while True:
            user_input = input("User: ")
            if user_input.lower() == "exit":
                break

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