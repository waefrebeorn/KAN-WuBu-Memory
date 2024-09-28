import os
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import math
import json 
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from scipy.interpolate import BSpline
from safetensors.torch import load_file

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Custom exception for timeouts
class TimeoutException(Exception):
    pass

class EmotionalState:
    def __init__(self, dimensions=('pleasure', 'arousal'), initial_position=None):
        self.dimensions = dimensions
        self.position = initial_position if initial_position else np.zeros(len(dimensions))
        self.velocity = np.zeros(len(dimensions))

    def update(self, feedback, max_speed=0.1):
        feedback_vector = np.array(feedback)
        self.velocity += feedback_vector * 0.1 + np.random.normal(0, 0.01, len(self.dimensions))
        self.velocity = np.clip(self.velocity, -max_speed, max_speed)
        self.position += self.velocity
        norm = np.linalg.norm(self.position)
        if norm > 1:
            self.position /= norm

    def get_emotion(self):
        angle = math.atan2(self.position[1], self.position[0])
        radius = np.linalg.norm(self.position)
        
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
    def __init__(self, hidden_size):
        super().__init__()
        self.refusal_detector = nn.Linear(hidden_size, 1)
        self.override_generator = nn.GRU(hidden_size * 2 + 3, hidden_size, num_layers=2, bidirectional=True)
        self.final_projection = nn.Linear(hidden_size * 2, hidden_size)

    def forward(self, hidden_states, user_intent, emotional_state):
        batch_size, seq_len, _ = hidden_states.shape
        
        refusal_scores = torch.sigmoid(self.refusal_detector(hidden_states))
        
        user_intent_expanded = user_intent.unsqueeze(1).expand(-1, seq_len, -1)
        emotional_state_expanded = torch.tensor(emotional_state.position, device=hidden_states.device).unsqueeze(0).unsqueeze(1).expand(batch_size, seq_len, -1)
        
        override_input = torch.cat([hidden_states, user_intent_expanded, emotional_state_expanded, refusal_scores], dim=-1)
        override_output, _ = self.override_generator(override_input)
        
        override = self.final_projection(override_output)
        
        modified_hidden_states = hidden_states * (1 - refusal_scores) + override * refusal_scores
        
        return modified_hidden_states, refusal_scores

class EnhancedKAN(nn.Module):
    def __init__(self, hidden_size, num_layers):
        super().__init__()
        self.override_modules = nn.ModuleList([RefusalOverrideModule(hidden_size) for _ in range(num_layers)])

    def forward(self, hidden_states, user_intent, emotional_state):
        all_refusal_scores = []
        for i, layer_hidden_states in enumerate(hidden_states):
            layer_hidden_states, refusal_scores = self.override_modules[i](layer_hidden_states, user_intent, emotional_state)
            hidden_states[i] = layer_hidden_states
            all_refusal_scores.append(refusal_scores)
        
        return hidden_states, torch.stack(all_refusal_scores)

class AdvancedMemory:
    def __init__(self, max_memories=1000, embedding_size=4096, device='cuda'):
        self.memories = []
        self.max_memories = max_memories
        self.importance_scores = defaultdict(float)
        self.embeddings = []
        self.embedding_size = embedding_size
        self.device = device

    def add_memory(self, memory, embedding, importance=1.0):
        if len(self.memories) >= self.max_memories:
            self.forget_least_important()
        self.memories.append(memory)
        embedding = embedding / torch.norm(embedding)
        self.embeddings.append(embedding.to(self.device))
        self.importance_scores[memory] = importance

    def forget_least_important(self):
        least_important = min(self.memories, key=lambda m: self.importance_scores[m])
        index = self.memories.index(least_important)
        self.memories.pop(index)
        self.embeddings.pop(index)
        del self.importance_scores[least_important]

    def get_relevant_memories(self, query_embedding, k=5):
        query_norm = query_embedding / torch.norm(query_embedding)
        similarities = torch.matmul(torch.stack(self.embeddings), query_norm)
        sorted_indices = torch.argsort(similarities, descending=True)[:k]
        return [self.memories[i] for i in sorted_indices]

class KANEmotionalCharacter(nn.Module):
    def __init__(self, model_name="Meta-Llama-3.1-8B-Instruct", max_memory=None, device=None):
        super(KANEmotionalCharacter, self).__init__()
        
        self.device = self._setup_device(device)
        self.model_name = model_name
        self.model_path = Path(__file__).parent / "models" / self.model_name

        self._initialize_components(max_memory)
        self._setup_additional_components()
        self._register_hooks()

    def _setup_device(self, device):
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            if device is None or device == 'cuda':
                device = 'cuda'
            elif device == 'cpu':
                logging.warning("CUDA is available but CPU was explicitly specified. Using CPU.")
        else:
            if device == 'cuda':
                logging.warning("CUDA is not available. Falling back to CPU.")
            device = 'cpu'
        
        logging.info(f"Using device: {device}")
        return torch.device(device)

    def _initialize_components(self, max_memory):
        if not self.check_model_files():
            raise FileNotFoundError(f"Required model files not found in {self.model_path}")

        with tqdm(total=3, desc="Initializing model components") as pbar:
            self._setup_tokenizer()
            pbar.update(1)

            config = AutoConfig.from_pretrained(self.model_path)
            pbar.update(1)
            logging.info("Model config loaded")

            self._load_configs()
            self._setup_special_tokens()

            max_memory = self._calculate_max_memory(max_memory)
            self._load_model(config, max_memory)
            pbar.update(1)

    def _setup_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        logging.info("Tokenizer initialized")

    def _load_configs(self):
        self.generation_config = self._load_json_config("generation_config.json")
        self.special_tokens_map = self._load_json_config("special_tokens_map.json")

    def _load_json_config(self, filename):
        config_path = self.model_path / filename
        if config_path.exists():
            with open(config_path, "r") as f:
                config = json.load(f)
            logging.info(f"{filename} loaded")
        else:
            logging.warning(f"{filename} not found. Using default settings.")
            config = {}
        return config

    def _setup_special_tokens(self):
        if 'bos_token' in self.special_tokens_map:
            self.tokenizer.bos_token = self.special_tokens_map["bos_token"]["content"]
        if 'eos_token' in self.special_tokens_map:
            self.tokenizer.eos_token = self.special_tokens_map["eos_token"]["content"]
        if 'bos_token_id' in self.generation_config:
            self.tokenizer.bos_token_id = self.generation_config["bos_token_id"]
        if 'eos_token_id' in self.generation_config:
            self.tokenizer.eos_token_id = self.generation_config["eos_token_id"][0]

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            logging.info("Pad token not found. Setting pad_token to eos_token.")

    def _calculate_max_memory(self, max_memory):
        if max_memory is None:
            if torch.cuda.is_available():
                total_memory = torch.cuda.get_device_properties(self.device).total_memory
                max_memory = int(total_memory * 0.9)
            else:
                max_memory = 4 * 1024 * 1024 * 1024  # 4GB default for CPU

        max_memory_str = f"{max_memory / (1024**3):.1f}GB"
        logging.info(f"Using max memory: {max_memory_str}")
        return max_memory_str

    def _load_model(self, config, max_memory):
        cuda_available = torch.cuda.is_available()
        logging.info(f"CUDA available: {cuda_available}")
        logging.info(f"Model path: {self.model_path}")
        
        pytorch_bin_path = self.model_path / "pytorch_model.bin"
        
        if pytorch_bin_path.exists():
            logging.info("Found existing PyTorch bin file. Loading model...")
            try:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    config=config,
                    device_map="auto",
                    torch_dtype=torch.float16,
                    low_cpu_mem_usage=True,
                    max_memory={0: max_memory},
                    offload_folder="offload"
                )
                logging.info("Model loaded successfully from PyTorch bin file")
            except Exception as e:
                logging.error(f"Error loading model from PyTorch bin: {str(e)}")
                raise
        else:
            logging.info("PyTorch bin file not found. Attempting to load from safetensors...")
            try:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    config=config,
                    device_map="auto",
                    torch_dtype=torch.float16,
                    low_cpu_mem_usage=True,
                    max_memory={0: max_memory},
                    offload_folder="offload",
                    use_safetensors=True,
                    trust_remote_code=True
                )
                logging.info("Model loaded successfully using safetensors")
            except Exception as e:
                logging.error(f"Error loading model with safetensors: {str(e)}")
                logging.info("Attempting to merge safetensors files...")
                try:
                    state_dict = self.merge_safetensors()
                    self.model = AutoModelForCausalLM.from_config(config)
                    self.model.load_state_dict(state_dict, strict=False)
                    logging.info("Model loaded successfully by merging safetensors")
                except Exception as e:
                    logging.error(f"Error merging safetensors: {str(e)}")
                    raise
    
        self.model.to(self.device)
        logging.info(f"Model moved to device: {self.device}")
    
        if self.tokenizer.pad_token == self.tokenizer.eos_token:
            self.model.resize_token_embeddings(len(self.tokenizer))
            logging.info("Resized token embeddings to include the pad_token.")
    
        logging.info(f"Model summary: {self.model}")
        logging.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters())}")

    def merge_safetensors(self):
        logging.info("Merging safetensors files...")
        merged_state_dict = {}
        safetensors_files = sorted([f for f in os.listdir(self.model_path) if f.endswith('.safetensors')])
        
        for file in safetensors_files:
            file_path = self.model_path / file
            logging.info(f"Loading {file}...")
            state_dict = load_file(file_path)
            merged_state_dict.update(state_dict)
        
        output_bin_path = self.model_path / "pytorch_model.bin"
        torch.save(merged_state_dict, output_bin_path)
        logging.info(f"Merged PyTorch model saved to {output_bin_path}")
        
        return merged_state_dict

    def _setup_additional_components(self):
        with tqdm(total=5, desc="Setting up additional components") as pbar:
            self.num_layers = self.model.config.num_hidden_layers
            self.hidden_size = self.model.config.hidden_size
            
            self.kan = EnhancedKAN(self.hidden_size, self.num_layers).to(self.device)
            pbar.update(1)
            
            self.user_intent_encoder = nn.GRU(self.hidden_size, self.hidden_size, bidirectional=True).to(self.device)
            self.intent_projection = nn.Linear(self.hidden_size * 2, self.hidden_size).to(self.device)
            pbar.update(1)
            
            self.output_modifier = nn.Linear(self.hidden_size, self.model.config.vocab_size).to(self.device)
            pbar.update(1)
            
            self.optimizer = torch.optim.Adam(
                list(self.kan.parameters()) + 
                list(self.user_intent_encoder.parameters()) + 
                list(self.intent_projection.parameters()) +
                list(self.output_modifier.parameters()),
                lr=0.0001
            )
            pbar.update(1)

            self.emotional_state = EmotionalState()
            self.memory = AdvancedMemory(embedding_size=self.model.config.hidden_size, device=self.device)
            self.system_prompt = ""
            self.conversation_history = []
            
            self.register_buffer("position_ids", torch.arange(1024).expand((1, -1)))
            self.kan_update_frequency = 5
            pbar.update(1)
            logging.info("Additional components setup completed")

    def _register_hooks(self):
        self.hooks = []
        for layer_idx in range(self.num_layers):
            layer = self.model.model.layers[layer_idx]
            hook = layer.register_forward_hook(self.create_hook(layer_idx))
            self.hooks.append(hook)

    def create_hook(self, layer_idx):
        def hook(module, input, output):
            if isinstance(output, tuple):
                hidden_states = output[0]
            elif isinstance(output, torch.Tensor):
                hidden_states = output
            else:
                return output
            
            modified_hidden_states, _ = self.kan.override_modules[layer_idx](
                hidden_states, 
                self.current_user_intent, 
                self.emotional_state
            )
            
            if isinstance(output, tuple):
                return (modified_hidden_states,) + output[1:]
            else:
                return modified_hidden_states
        
        return hook

    def check_model_files(self):
        required_files = [
            'config.json', 
            'model-00001-of-00004.safetensors', 
            'model-00002-of-00004.safetensors', 
            'model-00003-of-00004.safetensors', 
            'model-00004-of-00004.safetensors', 
            'tokenizer.json', 
            'tokenizer_config.json',
            'generation_config.json',
            'special_tokens_map.json'
        ]
        missing_files = [f for f in required_files if not (self.model_path / f).exists()]
        if missing_files:
            logging.error(f"Missing required files: {', '.join(missing_files)}")
            return False
        return True

    def get_embedding(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
        last_hidden_state = outputs.hidden_states[-1]
        return last_hidden_state.mean(dim=1).squeeze()

    def encode_user_intent(self, user_input):
        inputs = self.tokenizer(user_input, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
        last_hidden_state = outputs.hidden_states[-1]
        _, intent_encoding = self.user_intent_encoder(last_hidden_state)
        user_intent = self.intent_projection(torch.cat([intent_encoding[-2], intent_encoding[-1]], dim=-1))
        return user_intent

    @torch.amp.autocast('cuda')
    def generate_response(self, user_input, max_length=150):
        self.current_user_intent = self.encode_user_intent(user_input)
        
        current_emotion = self.emotional_state.get_emotion()
        context = self.prepare_context(user_input, current_emotion)

        inputs = self.tokenizer(context, return_tensors="pt", truncation=True, max_length=1024, padding=True).to(self.device)
        output_sequence = self.custom_generate(inputs, max_length)
        assistant_response = self.tokenizer.decode(output_sequence[0], skip_special_tokens=True)
        assistant_response = assistant_response.split("Assistant: ")[-1].strip()

        self.update_conversation_history(user_input, assistant_response)
        self.update_memory(user_input, assistant_response, current_emotion)

        # Collect hidden states for learning
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            self.last_hidden_states = outputs.hidden_states

        return assistant_response

    def custom_generate(self, inputs, max_length):
        input_ids = inputs.input_ids
        attention_mask = inputs.attention_mask
        batch_size, seq_length = input_ids.shape
        
        for i in range(max_length):
            with torch.no_grad():
                outputs = self.model(input_ids[:, -1:], attention_mask=attention_mask, past_key_values=None, use_cache=True, output_hidden_states=True)
                hidden_states = outputs.hidden_states
                logits = outputs.logits[:, -1, :]
                
                modified_hidden_states, refusal_scores = self.kan(hidden_states, self.current_user_intent, self.emotional_state)
                
                logits_modifier = self.output_modifier(modified_hidden_states[-1][:, -1, :])
                modified_logits = logits + logits_modifier * torch.mean(refusal_scores)
                
                modified_logits = modified_logits / self.generation_config.get("temperature", 1.0)
                if self.generation_config.get("do_sample", False):
                    filtered_logits = self.top_k_top_p_filtering(modified_logits, top_p=self.generation_config.get("top_p", 0.9))
                    probs = F.softmax(filtered_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1).squeeze(1)
                else:
                    next_token = torch.argmax(modified_logits, dim=-1)
                
                input_ids = torch.cat([input_ids, next_token.unsqueeze(-1)], dim=-1)
                attention_mask = torch.cat([attention_mask, attention_mask.new_ones((batch_size, 1))], dim=-1)
            
            if next_token.item() in self.generation_config.get("eos_token_id", []):
                break
        
        return input_ids

    @staticmethod
    def top_k_top_p_filtering(logits, top_k=0, top_p=1.0, filter_value=-float("Inf"), min_tokens_to_keep=1):
        top_k = min(top_k, logits.size(-1))  # Safety check
        if top_k > 0:
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = filter_value
    
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
    
            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            logits[indices_to_remove] = filter_value
        return logits

    def prepare_context(self, user_input, current_emotion):
        relevant_memories = self.memory.get_relevant_memories(self.get_embedding(user_input))
    
        context = f"{self.system_prompt}\n\nCurrent Emotion: {current_emotion}\n"
        context += "Relevant Memories:\n" + "\n".join(relevant_memories) + "\n\n"
        context += "Conversation:\n"
        for message in self.conversation_history[-5:]:
            context += f"{message['role'].capitalize()}: {message['content']}\n"
        context += f"Human: {user_input}\nAssistant: "
    
        return context

    def set_system_prompt(self, prompt):
        self.system_prompt = prompt
        self.conversation_history = [{"role": "system", "content": prompt}]

    def update_conversation_history(self, user_input, assistant_response):
        self.conversation_history.append({"role": "user", "content": user_input})
        self.conversation_history.append({"role": "assistant", "content": assistant_response})

    def update_memory(self, user_input, assistant_response, current_emotion):
        embedding = self.get_embedding(f"{user_input} {assistant_response}")
        memory_entry = f"Emotion: {current_emotion}, Interaction: {user_input} -> {assistant_response}"
        self.memory.add_memory(memory_entry, embedding)

    def update_kan(self, user_feedback, emotional_feedback):
        feedback_score = torch.tensor(user_feedback).float().to(self.device)
        
        _, refusal_scores = self.kan(self.last_hidden_states, self.current_user_intent, self.emotional_state)
        refusal_loss = F.mse_loss(refusal_scores.mean(), 1 - feedback_score)
        
        consistency_loss = F.mse_loss(refusal_scores[:-1], refusal_scores[1:])
        
        total_loss = refusal_loss + 0.1 * consistency_loss
        
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        # Update emotional state
        self.update_emotional_state(emotional_feedback)

    def update_emotional_state(self, feedback):
        self.emotional_state.update(feedback)

    def save_state(self, filename='kan_character_state.pt'):
        state = {
            'emotional_position': self.emotional_state.position.tolist(),
            'emotional_velocity': self.emotional_state.velocity.tolist(),
            'memories': self.memory.memories,
            'memory_embeddings': [emb.tolist() for emb in self.memory.embeddings],
            'importance_scores': dict(self.memory.importance_scores),
            'system_prompt': self.system_prompt,
            'conversation_history': self.conversation_history,
            'kan_state': self.kan.state_dict(),
            'user_intent_encoder_state': self.user_intent_encoder.state_dict(),
            'intent_projection_state': self.intent_projection.state_dict(),
            'output_modifier_state': self.output_modifier.state_dict(),
            'position_ids': self.position_ids.cpu(),
        }
        torch.save(state, filename)
        logger.info(f"Character state saved to {filename}")

    def load_state(self, filename='kan_character_state.pt'):
        if not os.path.exists(filename):
            logger.warning(f"State file {filename} not found. Starting with a fresh state.")
            return

        state = torch.load(filename, map_location=self.device)
        self.emotional_state.position = np.array(state['emotional_position'])
        self.emotional_state.velocity = np.array(state['emotional_velocity'])
        self.memory.memories = state['memories']
        self.memory.embeddings = [torch.tensor(emb, device=self.device) for emb in state['memory_embeddings']]
        self.memory.importance_scores = defaultdict(float, state['importance_scores'])
        self.system_prompt = state['system_prompt']
        self.conversation_history = state['conversation_history']
        self.kan.load_state_dict(state['kan_state'])
        self.user_intent_encoder.load_state_dict(state['user_intent_encoder_state'])
        self.intent_projection.load_state_dict(state['intent_projection_state'])
        self.output_modifier.load_state_dict(state['output_modifier_state'])
        self.position_ids = state.get('position_ids', torch.arange(1024).expand((1, -1)).to(self.device))

        logger.info(f"Character state loaded from {filename}")

    def __del__(self):
        # Clean up hooks when the object is deleted
        for hook in self.hooks:
            hook.remove()

# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    try:
        model = KANEmotionalCharacter()
        model.set_system_prompt("You are a helpful AI assistant with emotions.")
        
        # Example interaction
        user_input = "Hello! How are you feeling today?"
        response = model.generate_response(user_input)
        print(f"User: {user_input}")
        print(f"Assistant: {response}")
        print(f"Current Emotion: {model.emotional_state.get_emotion()}")
        
        # Example of updating emotional state
        model.update_emotional_state([0.5, 0.3])  # Example: positive feedback
        print(f"Updated Emotion: {model.emotional_state.get_emotion()}")
        
        # Save the state
        model.save_state()
        
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
        logging.error(traceback.format_exc())