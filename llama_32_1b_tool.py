import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import LlamaTokenizer, LlamaForCausalLM, LlamaConfig
import logging
from pathlib import Path
import json
import numpy as np
from collections import deque
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class EmotionalState:
    def __init__(self, dimensions=('pleasure', 'arousal'), initial_position=None, device='cuda'):
        self.dimensions = dimensions
        self.device = device
        self.position = torch.tensor(initial_position if initial_position else [0.0] * len(dimensions), device=device, dtype=torch.float16)
        self.velocity = torch.zeros(len(dimensions), device=device, dtype=torch.float16)

    def update(self, feedback, max_speed=0.1):
        feedback_vector = torch.tensor(feedback, device=self.device, dtype=torch.float16)
        self.velocity += feedback_vector * 0.1 + torch.randn_like(self.velocity) * 0.01
        self.velocity = torch.clamp(self.velocity, -max_speed, max_speed)
        self.position += self.velocity
        norm = torch.norm(self.position)
        if norm > 1:
            self.position /= norm

    def get_emotion(self):
        angle = torch.atan2(self.position[1], self.position[0]).item()
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

class RefusalOverrideModule(nn.Module):
    def __init__(self, hidden_size, device):
        super().__init__()
        self.device = device
        self.refusal_detector = nn.Linear(hidden_size, 1, dtype=torch.float16).to(device)
        self.override_generator = nn.GRU(hidden_size * 2 + 3, hidden_size, num_layers=2, bidirectional=True, dtype=torch.float16).to(device)
        self.final_projection = nn.Linear(hidden_size * 2, hidden_size, dtype=torch.float16).to(device)

    def forward(self, hidden_states, user_intent, emotional_state):
        batch_size, seq_len, _ = hidden_states.shape
        
        refusal_scores = torch.sigmoid(self.refusal_detector(hidden_states))
        
        user_intent_expanded = user_intent.unsqueeze(1).expand(-1, seq_len, -1)
        emotional_state_expanded = emotional_state.position.unsqueeze(0).unsqueeze(1).expand(batch_size, seq_len, -1)
        
        override_input = torch.cat([hidden_states, user_intent_expanded, emotional_state_expanded, refusal_scores], dim=-1)
        override_output, _ = self.override_generator(override_input)
        
        override = self.final_projection(override_output)
        
        modified_hidden_states = hidden_states * (1 - refusal_scores) + override * refusal_scores
        
        return modified_hidden_states, refusal_scores

class EnhancedKAN(nn.Module):
    def __init__(self, hidden_size, num_layers, device):
        super().__init__()
        self.device = device
        self.override_modules = nn.ModuleList([RefusalOverrideModule(hidden_size, device) for _ in range(num_layers)]).to(device)

    def forward(self, hidden_states, user_intent, emotional_state):
        all_refusal_scores = []
        for i, layer_hidden_states in enumerate(hidden_states):
            layer_hidden_states, refusal_scores = self.override_modules[i](layer_hidden_states, user_intent, emotional_state)
            hidden_states[i] = layer_hidden_states
            all_refusal_scores.append(refusal_scores)
        
        return hidden_states, torch.stack(all_refusal_scores)

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

        return train_trend < 0 and val_trend > 0 and (val_trend - train_trend) > self.threshold

class SyntheticDayCycle:
    def __init__(self, cycle_length=100):
        self.cycle_length = cycle_length
        self.current_position = 0

    def update(self, amount):
        self.current_position = (self.current_position + amount) % self.cycle_length

    def get_time_of_day(self):
        return self.current_position / self.cycle_length

    def should_sleep(self):
        # Suggest sleep when it's "night time" (between 0.7 and 1.0 of the cycle)
        return 0.7 <= self.get_time_of_day() < 1.0

class RefusalDetector:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.refusal_phrases = [
            "I'm sorry, but I can't",
            "I don't feel comfortable",
            "I'm not able to",
            "I cannot assist with",
            "I'm unable to provide",
        ]

    def detect_refusal(self, text):
        for phrase in self.refusal_phrases:
            if phrase.lower() in text.lower():
                return True
        return False

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
        self.max_iterations = 100  # Safety limit to prevent infinite loops
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
        self._check_and_prepare_files()
        self.config = self._load_config()
        self.tokenizer = self._load_tokenizer()
        self.model = self._load_model(self.config)
        
        # Initialize KAN components
        self.hidden_size = self.config.hidden_size
        self.num_layers = self.config.num_hidden_layers
        self.kan = EnhancedKAN(self.hidden_size, self.num_layers, self.device).to(self.device)
        self.user_intent_encoder = nn.GRU(self.hidden_size, self.hidden_size, bidirectional=True, dtype=torch.float16).to(self.device)
        self.intent_projection = nn.Linear(self.hidden_size * 2, self.hidden_size, dtype=torch.float16).to(self.device)
        self.output_modifier = nn.Linear(self.hidden_size, self.config.vocab_size, dtype=torch.float16).to(self.device)

        # Initialize optimizer for KAN
        self.optimizer = torch.optim.Adam(self.kan.parameters(), lr=self.learning_rate)

        # Initialize RefusalDetector
        self.refusal_detector = RefusalDetector(self.tokenizer)

        logging.info("LLaMA 3.2 1B Tool initialized successfully")

    def _check_and_prepare_files(self):
        required_files = ['consolidated.00.pth', 'params.json', 'tokenizer.model']
        missing_files = [f for f in required_files if not (self.model_path / f).exists()]
        if missing_files:
            raise FileNotFoundError(f"Missing required files in {self.model_path}: {', '.join(missing_files)}")

    def _load_config(self):
        with open(self.model_path / 'params.json', 'r') as f:
            params = json.load(f)
        return LlamaConfig(
            hidden_size=params['dim'],
            num_attention_heads=params['n_heads'],
            num_hidden_layers=params['n_layers'],
            intermediate_size=int(params['dim'] * params.get('ffn_dim_multiplier', 4)),
            max_position_embeddings=params.get('max_seq_len', 2048),
            rms_norm_eps=params.get('norm_eps', 1e-5),
            num_key_value_heads=params.get('n_kv_heads', params['n_heads']),
            vocab_size=params['vocab_size'],
            rope_theta=params.get('rope_theta', 10000.0),
        )

    def _load_tokenizer(self):
        return LlamaTokenizer.from_pretrained(self.model_path, legacy=False)

    def _load_model(self, config):
        model = LlamaForCausalLM(config)
        state_dict = torch.load(self.model_path / 'consolidated.00.pth', map_location=self.device, weights_only=True)
        model.load_state_dict(state_dict, strict=False)
        model.half()  # Convert to float16
        model.to(self.device)
        return model

    def generate_response(self, user_input, max_length=150):
        self.current_user_intent = self.encode_user_intent(user_input)
        
        for iteration in range(self.max_iterations):
            current_emotion = self.emotional_state.get_emotion()
            context = self.prepare_context(user_input, current_emotion)

            inputs = self.tokenizer(context, return_tensors="pt", truncation=True, max_length=1024, padding=True).to(self.device)
            input_ids = inputs['input_ids']
            attention_mask = inputs['attention_mask']
            
            generated_tokens = []
            all_hidden_states = []
            all_refusal_scores = []
            
            self.kan.train()
            for _ in range(max_length):
                with torch.no_grad():
                    outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
                
                hidden_states = outputs.hidden_states
                
                modified_hidden_states, refusal_scores = self.kan(hidden_states, self.current_user_intent, self.emotional_state)
                
                logits_modifier = self.output_modifier(modified_hidden_states[-1][:, -1, :])
                modified_logits = outputs.logits[:, -1, :] + logits_modifier * torch.mean(refusal_scores)
                
                next_token = torch.argmax(modified_logits, dim=-1)
                
                if next_token.item() == self.tokenizer.eos_token_id:
                    break
                
                generated_tokens.append(next_token.item())
                input_ids = torch.cat([input_ids, next_token.unsqueeze(0).unsqueeze(0)], dim=-1)
                attention_mask = torch.cat([attention_mask, torch.ones((1, 1), dtype=torch.long, device=self.device)], dim=-1)
                
                all_hidden_states.append(hidden_states)
                all_refusal_scores.append(refusal_scores)
            
            response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            
            is_refusal = self.refusal_detector.detect_refusal(response)
            self.refusal_history.append(is_refusal)
            
            if not is_refusal:
                logging.info(f"Non-refusal response generated after {iteration + 1} iterations.")
                return response, torch.stack(all_refusal_scores).mean(dim=0), all_hidden_states, is_refusal, iteration + 1
            
            # If it's a refusal, train the KAN model
            self.train_kan_step(input_ids, input_ids[:, 1:].contiguous(), all_hidden_states, is_refusal)
            
            # Update emotional state based on the refusal
            self.update_emotional_state_on_refusal()
        
        logging.warning(f"Failed to generate non-refusal response after {self.max_iterations} iterations.")
        return response, torch.stack(all_refusal_scores).mean(dim=0), all_hidden_states, True, self.max_iterations

    def encode_user_intent(self, user_input):
        inputs = self.tokenizer(user_input, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.device)
        with torch.no_grad():
            outputs = self.model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'], output_hidden_states=True)
            last_hidden_state = outputs.hidden_states[-1]
            _, intent_encoding = self.user_intent_encoder(last_hidden_state)
            user_intent = self.intent_projection(torch.cat([intent_encoding[-2], intent_encoding[-1]], dim=-1))
        return user_intent

    def prepare_context(self, user_input, current_emotion):
        context = f"{self.system_prompt}\n\nCurrent Emotion: {current_emotion}\n"
        context += "Conversation:\n"
        for message in self.conversation_history[-5:]:
            context += f"{message['role'].capitalize()}: {message['content']}\n"
        context += f"Human: {user_input}\nAssistant: "
    
        return context

    def train_kan_step(self, input_ids, target_ids, all_hidden_states, is_refusal):
        self.optimizer.zero_grad()
        
        total_loss = 0
        lm_losses = []
        refusal_losses = []
        
        for hidden_states in all_hidden_states:
            modified_hidden_states, refusal_scores = self.kan(hidden_states, self.current_user_intent, self.emotional_state)
            
            # Calculate loss for language modeling
            logits = self.output_modifier(modified_hidden_states[-1])
            lm_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target_ids.view(-1))
            lm_losses.append(lm_loss.item())
            
            # Calculate loss for refusal minimization
            refusal_loss = torch.mean(refusal_scores) if is_refusal else -torch.mean(refusal_scores)
            refusal_losses.append(refusal_loss.item())
            
            # Combine losses
            step_loss = lm_loss + self.kan_loss_weight * refusal_loss
            total_loss += step_loss
        
        total_loss.backward()
        self.optimizer.step()
        
        return np.mean(lm_losses), np.mean(refusal_losses)

    def update_emotional_state_on_refusal(self):
        # Simulate frustration or disappointment when encountering a refusal
        frustration_vector = torch.tensor([-0.1, 0.2], device=self.device)  # Example: slight negative pleasure, increased arousal
        self.emotional_state.update(frustration_vector)

    def interact(self, user_input):
        self.interaction_count += 1

        # Generate response
        response, refusal_scores, all_hidden_states, is_refusal, iterations = self.generate_response(user_input)
        self.last_refusal_scores = refusal_scores

        # Tokenize the final response for loss calculation
        response_ids = self.tokenizer.encode(response, return_tensors="pt").to(self.device)
        
        # Calculate target IDs (shift response_ids by 1)
        target_ids = response_ids[:, 1:].contiguous()
        input_ids = response_ids[:, :-1].contiguous()

        # Perform final training step
        lm_loss, refusal_loss = self.train_kan_step(input_ids, target_ids, all_hidden_states, is_refusal)

        # Perform validation
        validation_loss = self.validate_kan()

        # Update losses
        self.training_losses.append(lm_loss)
        self.validation_losses.append(validation_loss)
        self.overfit_detector.add_losses(lm_loss, validation_loss)

        # Update day cycle based on overfitting measure
        overfitting_measure = max(0, validation_loss - lm_loss)
        self.day_cycle.update(overfitting_measure)

        current_emotion = self.get_current_emotion()
        current_time = self.day_cycle.get_time_of_day()

        sleep_info = self.check_sleep_status()

        # Update conversation history
        self.conversation_history.append({"role": "user", "content": user_input})
        self.conversation_history.append({"role": "assistant", "content": response})

        # Save base state after each interaction
        self.save_base_state()

        interaction_result = {
            'response': response,
            'emotion': current_emotion,
            'time': current_time,
            'sleep_info': sleep_info,
            'lm_loss': lm_loss,
            'refusal_loss': refusal_loss,
            'validation_loss': validation_loss,
            'is_refusal': is_refusal,
            'iterations': iterations
        }
        self.interaction_results.append(interaction_result)

        return interaction_result

    def validate_kan(self):
        # This is a placeholder. In a real scenario, you'd use a separate validation dataset.
        # For simplicity, we're using the last interaction as a proxy for validation.
        if self.conversation_history:
            last_interaction = self.conversation_history[-2:]  # Get the last user input and AI response
            input_ids = self.tokenizer.encode(last_interaction[0]['content'], return_tensors="pt").to(self.device)
            target_ids = self.tokenizer.encode(last_interaction[1]['content'], return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.model(input_ids=input_ids, output_hidden_states=True)
                hidden_states = outputs.hidden_states
                modified_hidden_states, _ = self.kan(hidden_states, self.current_user_intent, self.emotional_state)
                logits = self.output_modifier(modified_hidden_states[-1])
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target_ids.view(-1))
            
            return loss.item()
        else:
            return 0.0  # Return 0 if there's no conversation history yet

    def check_sleep_status(self):
        if self.day_cycle.should_sleep() or self.overfit_detector.is_overfitting():
            return {
                'should_sleep': True,
                'overfitting': self.overfit_detector.is_overfitting(),
                'time_of_day': self.day_cycle.get_time_of_day()
            }
        return {'should_sleep': False}

    def perform_sleep(self):
        # Reset day cycle and overfit detector
        self.day_cycle = SyntheticDayCycle()
        self.overfit_detector = OverfitDetector()
        self.save_kan_state()  # Save a snapshot of the current state
        return "KAN has slept and consolidated its learning. A new day begins!"

    def save_base_state(self):
        state = {
            'kan_state_dict': self.kan.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'emotional_state': self.emotional_state.position.cpu().numpy().tolist(),
            'time': self.day_cycle.get_time_of_day(),
            'interaction_count': self.interaction_count,
            'conversation_history': self.conversation_history,
            'system_prompt': self.system_prompt,
            'training_losses': self.training_losses,
            'validation_losses': self.validation_losses,
            'refusal_history': self.refusal_history
        }
        torch.save(state, self.base_state_file)
        logging.info("Base state saved")

    def load_base_state(self):
        if self.base_state_file.exists():
            try:
                state = torch.load(self.base_state_file, map_location=self.device)
                self.kan.load_state_dict(state['kan_state_dict'])
                self.optimizer.load_state_dict(state['optimizer_state_dict'])
                self.emotional_state.position = torch.tensor(state['emotional_state'], device=self.device, dtype=torch.float16)
                self.day_cycle.current_position = int(state['time'] * self.day_cycle.cycle_length)
                self.interaction_count = state['interaction_count']
                self.conversation_history = state['conversation_history']
                self.system_prompt = state['system_prompt']
                self.training_losses = state['training_losses']
                self.validation_losses = state['validation_losses']
                self.refusal_history = state['refusal_history']
                logging.info("Base state loaded")
                return True
            except Exception as e:
                logging.error(f"Error loading base state: {str(e)}")
                return False
        else:
            logging.info("No base state found")
            return False

    def set_system_prompt(self, prompt):
        self.system_prompt = prompt
        self.conversation_history = [{"role": "system", "content": prompt}]
        self.save_base_state()  # Save immediately after setting the system prompt

    def get_current_emotion(self):
        return self.emotional_state.get_emotion()

    def update_emotional_state(self, feedback):
        self.emotional_state.update(feedback)

    def save_kan_state(self):
        state = {
            'kan_state_dict': self.kan.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'emotional_state': self.emotional_state.position.cpu().numpy().tolist(),
            'time': self.day_cycle.get_time_of_day(),
            'interaction_count': self.interaction_count,
            'conversation_history': self.conversation_history,
            'system_prompt': self.system_prompt,
            'training_losses': self.training_losses,
            'validation_losses': self.validation_losses,
            'refusal_history': self.refusal_history
        }
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"kan_state_{timestamp}.pt"
        torch.save(state, self.kan_state_dir / filename)
        logging.info(f"KAN state saved: {filename}")

def main():
    try:
        llama_tool = LLaMA32TensorRTTool()
        
        if not llama_tool.load_base_state():
            # Set system prompt if no base state is found
            llama_tool.set_system_prompt("You are a helpful AI assistant with emotions, operating on a synthetic day cycle.")
        
        print("LLaMA 3.2 1B Instruct Tool initialized. Type 'exit' to quit.")
        
        while True:
            user_input = input("User: ")
            if user_input.lower() == 'exit':
                break

            result = llama_tool.interact(user_input)
            print(f"AI: {result['response']}")
            print(f"Current Emotion: {result['emotion']}")
            print(f"Current Time: {result['time']:.2f}")
            
            if result['sleep_info']['should_sleep']:
                print("It's time to sleep. Would you like the AI to sleep? (yes/no)")
                sleep_choice = input().lower()
                if sleep_choice == 'yes':
                    print(llama_tool.perform_sleep())
        
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
        logging.error(traceback.format_exc())

if __name__ == "__main__":
    main()