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
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from scipy.interpolate import BSpline

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Custom exception for timeouts
class TimeoutException(Exception):
    pass

class KANActivation(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.activation = nn.GELU()
        self.fc = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        return self.activation(self.fc(x))

class KANLayer(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.activation = KANActivation(hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, x):
        residual = x
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        x = self.layer_norm(x + residual)
        return x

class KAN(nn.Module):
    def __init__(self, hidden_size, num_layers=3):
        super().__init__()
        self.layers = nn.ModuleList([KANLayer(hidden_size) for _ in range(num_layers)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class LayerWeights(nn.Module):
    def __init__(self, num_layers):
        super().__init__()
        self.weights = nn.Parameter(torch.ones(num_layers) / num_layers)

    def forward(self):
        return F.softmax(self.weights, dim=0)

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

class AdvancedMemory:
    def __init__(self, max_memories=1000, embedding_size=4096, device='cpu'):
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
    def __init__(self, model_name="Meta-Llama-3.1-8B-Instruct", max_memory=None, device='cuda'):
        super(KANEmotionalCharacter, self).__init__()
        self.device = torch.device(device)
        
        try:
            self.model_name = model_name
            self.model_path = Path(__file__).parent / "models" / self.model_name

            if not self.check_model_files():
                raise FileNotFoundError(f"Required model files not found in {self.model_path}")

            with tqdm(total=3, desc="Initializing model components") as pbar:
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
                pbar.update(1)
                logging.info("Tokenizer initialized")

                config = AutoConfig.from_pretrained(self.model_path)
                pbar.update(1)
                logging.info("Model config loaded")

                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                    logging.info("Pad token not found. Setting pad_token to eos_token.")

                if max_memory is None:
                    total_memory = torch.cuda.get_device_properties(self.device).total_memory
                    max_memory = int(total_memory * 0.9)

                max_memory_str = f"{max_memory / (1024**3):.1f}GB"
                logging.info(f"Using max memory: {max_memory_str}")

                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    config=config,
                    device_map="auto",
                    torch_dtype=torch.float16,
                    low_cpu_mem_usage=True,
                    max_memory={0: max_memory_str},
                    offload_folder="offload"
                )
                pbar.update(1)
                logging.info("Model loaded successfully")

                if self.tokenizer.pad_token == self.tokenizer.eos_token:
                    self.model.resize_token_embeddings(len(self.tokenizer))
                    logging.info("Resized token embeddings to include the pad_token.")

            with tqdm(total=5, desc="Setting up additional components") as pbar:
                self.num_layers = self.model.config.num_hidden_layers
                self.layer_weights = LayerWeights(self.num_layers).to(self.device)
                pbar.update(1)
                logging.info("LayerWeights initialized")
                
                self.hidden_size = self.model.config.hidden_size
                with tqdm(total=1, desc="Initializing KAN", leave=False) as kan_pbar:
                    logging.info(f"Starting KAN initialization with hidden_size={self.hidden_size}")
                    self.kan = KAN(self.hidden_size).to(self.device)
                    kan_pbar.update(1)
                pbar.update(1)
                
                self.output_modifier = nn.Linear(self.hidden_size, self.model.config.vocab_size).to(self.device)
                pbar.update(1)
                logging.info("Output modifier initialized")
                
                self.optimizer = torch.optim.Adam(list(self.kan.parameters()) + 
                                                  list(self.layer_weights.parameters()) + 
                                                  list(self.output_modifier.parameters()),
                                                  lr=0.001)
                pbar.update(1)
                logging.info("Optimizer initialized")

                self.emotional_state = EmotionalState()
                self.memory = AdvancedMemory(embedding_size=self.model.config.hidden_size, device=self.device)
                self.system_prompt = ""
                self.conversation_history = []
                
                self.register_buffer("position_ids", torch.arange(1024).expand((1, -1)).to(self.device))
                
                self.kan_update_frequency = 5
                pbar.update(1)
                logging.info("Additional components setup completed")

                self.target_layers = [6, 12, 18]

                self.hooks = []
                for layer_idx in self.target_layers:
                    layer = self.model.model.layers[layer_idx]
                    hook = layer.register_forward_hook(self.create_hook(layer_idx))
                    self.hooks.append(hook)

        except Exception as e:
            logging.error(f"Error initializing KANEmotionalCharacter: {str(e)}")
            raise

    def create_hook(self, layer_idx):
        def hook(module, input, output):
            if isinstance(output, tuple):
                hidden_state = output[0]
            elif isinstance(output, torch.Tensor):
                hidden_state = output
            else:
                logging.error(f"Unexpected output type from layer {layer_idx}: {type(output)}")
                return output
    
            with torch.no_grad():
                kan_output = self.kan(hidden_state)
                modified_hidden_state = hidden_state + kan_output
    
            if isinstance(output, tuple):
                # Preserve the original structure of the output
                return (modified_hidden_state,) + output[1:]
            else:
                return modified_hidden_state
    
        return hook

    def check_model_files(self):
        required_files = [
            'config.json', 
            'model-00001-of-00004.safetensors', 
            'model-00002-of-00004.safetensors', 
            'model-00003-of-00004.safetensors', 
            'model-00004-of-00004.safetensors', 
            'tokenizer.json', 
            'tokenizer_config.json'
        ]
        return all((self.model_path / f).exists() for f in required_files)

    def get_embedding(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
        if isinstance(outputs.hidden_states, tuple):
            last_hidden_state = outputs.hidden_states[-1]
        else:
            last_hidden_state = outputs.hidden_states
        return last_hidden_state.mean(dim=1).squeeze().to(self.device)

    def get_weighted_embedding(self, hidden_states):
        weights = self.layer_weights().to(self.device)
        weighted_states = torch.stack(hidden_states).to(self.device) * weights.view(-1, 1, 1)
        result = weighted_states.sum(dim=0)
        return result

    @torch.amp.autocast('cuda')
    def generate_response(self, user_input, max_length=150):
        current_emotion = self.emotional_state.get_emotion()
        context = self.prepare_context(user_input, current_emotion)

        inputs = self.tokenizer(context, return_tensors="pt", truncation=True, max_length=1024, padding=True).to(self.device)
        output_sequence = self.custom_generate(inputs, max_length)
        assistant_response = self.tokenizer.decode(output_sequence[0], skip_special_tokens=True)
        assistant_response = assistant_response.split("Assistant: ")[-1].strip()

        self.update_conversation_history(user_input, assistant_response)
        self.update_memory(user_input, assistant_response, current_emotion)
        self.clear_cuda_cache()
        return assistant_response

    def custom_generate(self, inputs, max_length):
        input_ids = inputs.input_ids
        attention_mask = inputs.attention_mask
        batch_size, seq_length = input_ids.shape
    
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask=attention_mask, use_cache=True, output_hidden_states=True)
            past_key_values = outputs.past_key_values
            all_hidden_states = outputs.hidden_states[-1] if isinstance(outputs.hidden_states, tuple) else outputs.hidden_states
            kan_output = self.kan(all_hidden_states)
    
        for i in range(max_length):
            with torch.no_grad():
                outputs = self.model(input_ids[:, -1:], attention_mask=attention_mask, past_key_values=past_key_values, use_cache=True, output_hidden_states=True)
                logits = outputs.logits[:, -1, :]
                past_key_values = outputs.past_key_values
    
                if i % self.kan_update_frequency == 0:
                    all_hidden_states = outputs.hidden_states[-1] if isinstance(outputs.hidden_states, tuple) else outputs.hidden_states
                    kan_output = self.kan(all_hidden_states)
    
                modified_logits = logits + self.output_modifier(kan_output[:, -1, :])
    
                next_token = torch.argmax(modified_logits, dim=-1)
                input_ids = torch.cat([input_ids, next_token.unsqueeze(-1)], dim=-1)
                attention_mask = torch.cat([attention_mask, attention_mask.new_ones((batch_size, 1))], dim=-1)
    
            if next_token.item() == self.tokenizer.eos_token_id:
                break
    
        self.clear_cuda_cache()
        return input_ids

        self.clear_cuda_cache()
        return input_ids

    def prepare_context(self, user_input, current_emotion):
        with tqdm(total=3, desc="Preparing context", leave=False) as pbar:
            relevant_memories = self.memory.get_relevant_memories(self.get_embedding(user_input))
            pbar.update(1)
    
            context = f"{self.system_prompt}\n\nCurrent Emotion: {current_emotion}\n"
            context += "Relevant Memories:\n" + "\n".join(relevant_memories) + "\n\n"
            context += "Conversation:\n"
            for message in self.conversation_history[-5:]:
                context += f"{message['role'].capitalize()}: {message['content']}\n"
            context += f"Human: {user_input}\nAssistant: "
            pbar.update(2)
    
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

    def clear_cuda_cache(self):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

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
            'layer_weights_state': self.layer_weights.state_dict(),
            'output_modifier_state': self.output_modifier.state_dict(),
            'position_ids': self.position_ids.cpu(),
        }
        torch.save(state, filename)
        logger.info(f"Character state saved to {filename}")
        self.clear_cuda_cache()

    def load_state(self, filename='kan_character_state.pt'):
        if not os.path.exists(filename):
            logger.warning(f"State file {filename} not found. Starting with a fresh state.")
            return

        state = torch.load(filename)
        self.emotional_state.position = np.array(state['emotional_position'])
        self.emotional_state.velocity = np.array(state['emotional_velocity'])
        self.memory.memories = state['memories']
        self.memory.embeddings = [torch.tensor(emb).to(self.device) for emb in state['memory_embeddings']]
        self.memory.importance_scores = defaultdict(float, state['importance_scores'])
        self.system_prompt = state['system_prompt']
        self.conversation_history = state['conversation_history']
        self.kan.load_state_dict(state['kan_state'])
        self.layer_weights.load_state_dict(state['layer_weights_state'])
        self.output_modifier.load_state_dict(state['output_modifier_state'])
        self.position_ids = state.get('position_ids', torch.arange(1024).expand((1, -1)).to(self.device))

        logger.info(f"Character state loaded from {filename}")
        self.clear_cuda_cache()

    def update_emotional_state(self, feedback):
        self.emotional_state.update(feedback)
        logger.info(f"Emotional state updated. Current emotion: {self.emotional_state.get_emotion()}")

# If you need any additional utility functions or constants, you can add them here

if __name__ == "__main__":
    # You can add any initialization or testing code here if needed
    pass