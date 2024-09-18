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
    def __init__(self, num_knots=10, degree=3, knot_range=(-5, 5)):
        super().__init__()
        self.num_knots = num_knots
        self.degree = degree
        self.knot_range = knot_range
        
        self.knots = nn.Parameter(torch.linspace(knot_range[0], knot_range[1], num_knots))
        self.coeffs = nn.Parameter(torch.randn(num_knots) * 0.1)
        
        self.wb = nn.Parameter(torch.randn(1) / np.sqrt(num_knots))
        self.ws = nn.Parameter(torch.ones(1))
        
        self.update_spline()

    def update_spline(self):
        self.spline = BSpline(
            self.knots.detach().cpu().numpy(), 
            self.coeffs.detach().cpu().numpy(), 
            self.degree
        )

    def forward(self, x):
        x = x.to(self.wb.device)
        b_x = torch.nn.functional.silu(x)
        s_x = torch.tensor(
            self.spline(x.detach().cpu().numpy()), 
            dtype=torch.float32
        ).to(x.device)
        return self.wb * b_x + self.ws * s_x

class KANLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Create the linear transformation
        self.linear = nn.Linear(in_features, out_features)

        # Define activations based on out_features
        self.activations = nn.ModuleList([KANActivation() for _ in range(out_features)])

    def forward(self, x):
        x = x.to(self.linear.weight.device)  # Move input tensor to the same device as linear layer weights
        # Apply linear transformation
        linear_output = self.linear(x)

        # Dynamically adjust the number of activations if input size differs
        current_out_features = linear_output.shape[1]
        if current_out_features != self.out_features:
            logging.warning(f"Adjusting activations to match output size ({current_out_features}).")
            self.activations = nn.ModuleList([KANActivation() for _ in range(current_out_features)])

        # Apply activations to each output feature
        activated_outputs = [self.activations[i](linear_output[:, i]) for i in range(current_out_features)]
        activated_output = torch.stack(activated_outputs, dim=1)

        return activated_output


class KAN(nn.Module):
    def __init__(self, layer_sizes):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(len(layer_sizes) - 1):
            logging.info(f"Initializing KAN layer {i+1}: {layer_sizes[i]} -> {layer_sizes[i+1]}")
            self.layers.append(KANLayer(layer_sizes[i], layer_sizes[i+1]))
        logging.info("KAN layers initialized")

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            logging.info(f"Processing KAN layer {i+1}/{len(self.layers)}")
            x = layer(x)
        return x  # Final output: (batch_size x 512)


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
        self.embeddings.append(embedding.to(self.device))  # Ensure embeddings are on the correct device
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
                
                hidden_size = self.model.config.hidden_size
                with tqdm(total=1, desc="Initializing KAN", leave=False) as kan_pbar:
                    logging.info(f"Starting KAN initialization with hidden_size={hidden_size}")
                    self.kan = KAN([4096, 256, 128, 512]).to(self.device)
                    kan_pbar.update(1)
                pbar.update(1)
                
                self.projection_layer = nn.Linear(512, hidden_size).to(self.device)
                nn.init.xavier_uniform_(self.projection_layer.weight)
                if self.projection_layer.bias is not None:
                    nn.init.zeros_(self.projection_layer.bias)
                pbar.update(1)
                logging.info("Projection layer initialized")

                self.output_modifier = nn.Linear(hidden_size, self.model.config.vocab_size).to(self.device)
                pbar.update(1)
                logging.info("Output modifier initialized")
                
                self.optimizer = torch.optim.Adam(list(self.kan.parameters()) + 
                                                  list(self.layer_weights.parameters()) + 
                                                  list(self.projection_layer.parameters()) + 
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
            
    def set_system_prompt(self, prompt):
        self.system_prompt = prompt
        self.conversation_history = [{"role": "system", "content": prompt}]
        
    def prepare_context(self, user_input, current_emotion):
        """
        Prepares the context for generating a response by incorporating the system prompt, 
        conversation history, and relevant memories.
        """
        with tqdm(total=3, desc="Preparing context", leave=False) as pbar:
            # Retrieve relevant memories based on the user's input
            relevant_memories = self.memory.get_relevant_memories(self.get_embedding(user_input))
            pbar.update(1)
    
            # Build the context string by including system prompt, memories, and conversation history
            context = f"{self.system_prompt}\n\nCurrent Emotion: {current_emotion}\n"
            context += "Relevant Memories:\n" + "\n".join(relevant_memories) + "\n\n"
            context += "Conversation:\n"
            for message in self.conversation_history[-5:]:
                context += f"{message['role'].capitalize()}: {message['content']}\n"
            context += f"Human: {user_input}\nAssistant: "
            pbar.update(1)
    
            pbar.update(1)
    
        return context
    

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
                weighted_embedding = self.get_weighted_embedding([hidden_state]).to(self.device)
                emotional_embedding = self.kan(weighted_embedding).to(self.device)
                projected_embedding = self.projection_layer(emotional_embedding).to(self.device)
                projected_embedding = projected_embedding.unsqueeze(1).repeat(1, hidden_state.size(1), 1)
            
            modified_hidden_state = hidden_state + projected_embedding.to(hidden_state.device)
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
        return outputs.hidden_states[-1].mean(dim=1).squeeze().to(self.device)

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
            all_hidden_states = outputs.hidden_states
            weighted_embedding = self.get_weighted_embedding(all_hidden_states)
            kan_output = self.kan(weighted_embedding)

        for i in range(max_length):
            with torch.no_grad():
                outputs = self.model(input_ids[:, -1:], attention_mask=attention_mask, past_key_values=past_key_values, use_cache=True, output_hidden_states=True)
                logits = outputs.logits[:, -1, :]
                past_key_values = outputs.past_key_values

                if i % self.kan_update_frequency == 0:
                    all_hidden_states = outputs.hidden_states
                    weighted_embedding = self.get_weighted_embedding(all_hidden_states)
                    kan_output = self.kan(weighted_embedding)

                projected_embedding = self.projection_layer(kan_output).unsqueeze(1).repeat(1, 1, 1)
                modified_logits = logits + self.output_modifier(projected_embedding.squeeze(1))

                next_token = torch.argmax(modified_logits, dim=-1)
                input_ids = torch.cat([input_ids, next_token.unsqueeze(-1)], dim=-1)
                attention_mask = torch.cat([attention_mask, attention_mask.new_ones((batch_size, 1))], dim=-1)

            if next_token.item() == self.tokenizer.eos_token_id:
                break

        self.clear_cuda_cache()
        return input_ids

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
            'projection_layer_state': self.projection_layer.state_dict(),
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
        self.projection_layer.load_state_dict(state['projection_layer_state'])
        self.output_modifier.load_state_dict(state['output_modifier_state'])
        self.position_ids = state.get('position_ids', torch.arange(1024).expand((1, -1)).to(self.device))

        logger.info(f"Character state loaded from {filename}")
        self.clear_cuda_cache()
