import torch
import torch.nn as nn
import logging
import time
import pynvml
import numpy as np
import json
from tqdm import tqdm
from collections import defaultdict
import math
import random
import os
import requests
from transformers import AutoTokenizer, AutoModelForCausalLM
from requests_oauthlib import OAuth2Session

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class KANLayer(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(KANLayer, self).__init__()
        self.fc = nn.Linear(input_size, hidden_size)
        self.activation = nn.ReLU()
    
    def forward(self, x):
        return self.activation(self.fc(x))

class KAN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=3):
        super(KAN, self).__init__()
        self.layers = nn.ModuleList([KANLayer(input_size if i == 0 else hidden_size, hidden_size) for i in range(num_layers)])
        self.output_layer = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)  # Add batch dimension if input is 1D
        for layer in self.layers:
            x = layer(x)
        return self.output_layer(x)

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
    def __init__(self, max_memories=1000, embedding_size=4096):
        self.memories = []
        self.max_memories = max_memories
        self.importance_scores = defaultdict(float)
        self.embeddings = []
        self.embedding_size = embedding_size

    def add_memory(self, memory, embedding, importance=1.0):
        if len(self.memories) >= self.max_memories:
            self.forget_least_important()
        self.memories.append(memory)
        self.embeddings.append(embedding)
        self.importance_scores[memory] = importance

    def forget_least_important(self):
        least_important = min(self.memories, key=lambda m: self.importance_scores[m])
        index = self.memories.index(least_important)
        self.memories.pop(index)
        self.embeddings.pop(index)
        del self.importance_scores[least_important]

    def get_relevant_memories(self, query_embedding, k=5):
        similarities = [np.dot(query_embedding, mem_embedding) for mem_embedding in self.embeddings]
        sorted_indices = np.argsort(similarities)[::-1][:k]
        return [self.memories[i] for i in sorted_indices]

    def update_importance(self, memory, delta):
        if memory in self.importance_scores:
            self.importance_scores[memory] += delta

class KANEmotionalCharacter:
    def __init__(self, model_name="meta-llama/Llama-2-7b-chat-hf", timeout=300):
        pynvml.nvmlInit()
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        self.device = self.get_device()
        logging.info(f"Using device: {self.device}")

        self.model_name = model_name
        self.timeout = timeout

        # OAuth2 setup
        self.client_id = os.getenv('HF_CLIENT_ID')
        self.client_secret = os.getenv('HF_CLIENT_SECRET')
        if not self.client_id or not self.client_secret:
            raise ValueError("HF_CLIENT_ID and HF_CLIENT_SECRET must be set as environment variables")

        self.oauth = OAuth2Session(self.client_id)
        self.token = self.get_oauth_token()

        # Initialize tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_auth_token=self.token['access_token'])
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, use_auth_token=self.token['access_token']).to(self.device)

        # Initialize other components...
        self.emotional_state = EmotionalState()
        self.memory = AdvancedMemory(embedding_size=self.model.config.hidden_size)
        self.system_prompt = ""
        self.conversation_history = []
        self.kan = KAN(input_size=self.model.config.hidden_size, hidden_size=512, output_size=self.model.config.hidden_size).to(self.device)
        self.optimizer = torch.optim.Adam(self.kan.parameters(), lr=0.001)
        self.brain_cache = {}
        
    def get_oauth_token(self):
        authorization_base_url = 'https://huggingface.co/oauth/authorize'
        token_url = 'https://huggingface.co/oauth/token'

        authorization_url, _ = self.oauth.authorization_url(authorization_base_url)
        print(f'Please go to this URL to authorize the application: {authorization_url}')
        redirect_response = input('Paste the full redirect URL here: ')

        token = self.oauth.fetch_token(token_url, authorization_response=redirect_response, client_secret=self.client_secret)
        return token        

    def get_device(self):
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                if "NVIDIA" in torch.cuda.get_device_name(i):
                    torch.cuda.set_device(i)
                    logging.info(f"Using CUDA device: {torch.cuda.get_device_name(i)}")
                    return torch.device(f"cuda:{i}")
        logging.info("CUDA not available. Using CPU.")
        return torch.device("cpu")
        
    def get_gpu_usage(self):
        info = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
        return f"GPU Memory: {info.used / 1024**2:.1f}MB / {info.total / 1024**2:.1f}MB"

    def set_system_prompt(self, prompt):
        self.system_prompt = prompt
        self.conversation_history = [{"role": "system", "content": prompt}]

    def get_embedding(self, text):
        # For simplicity, we'll use a random embedding here
        # In a real-world scenario, you'd want to use a proper embedding model
        return np.random.rand(4096)

    def train_kan(self, input_embedding, target_embedding):
        input_tensor = torch.tensor(input_embedding, dtype=torch.float32).to(self.device)
        target_tensor = torch.tensor(target_embedding, dtype=torch.float32).to(self.device)
        
        self.optimizer.zero_grad()
        output = self.kan(input_tensor)
        loss = nn.MSELoss()(output, target_tensor)
        loss.backward()
        self.optimizer.step()
        
        return loss.item()

    def generate_response(self, user_input, max_length=150):
        current_emotion = self.emotional_state.get_emotion()
        query_embedding = self.get_embedding(user_input)
        logging.info(f"Query embedding shape: {query_embedding.shape}")
        
        relevant_memories = self.memory.get_relevant_memories(query_embedding)
        
        context = f"{self.system_prompt}\n\nCurrent Emotion: {current_emotion}\n"
        context += "Relevant Memories:\n" + "\n".join(relevant_memories) + "\n\n"
        context += "Conversation:\n"
        for message in self.conversation_history[-5:]:  # Only include the last 5 messages for context
            context += f"{message['role'].capitalize()}: {message['content']}\n"
        context += f"Human: {user_input}\nAssistant: "
    
        payload = {
            "model": self.model_name,
            "prompt": context,
            "stream": False
        }
        
        start_time = time.time()
        
        try:
            response = requests.post(self.ollama_url, json=payload)
            response.raise_for_status()
            assistant_response = response.json()['response']
        except requests.exceptions.RequestException as e:
            logging.error(f"Error calling Ollama API: {str(e)}")
            return "I'm sorry, I encountered an error while processing your request."
    
        end_time = time.time()
        logging.info(f"Response generation took {end_time - start_time:.2f} seconds")
    
        self.conversation_history.append({"role": "user", "content": user_input})
        self.conversation_history.append({"role": "assistant", "content": assistant_response})
        
        response_embedding = self.get_embedding(assistant_response)
        self.memory.add_memory(f"Emotion: {current_emotion}, Interaction: {user_input} -> {assistant_response}", response_embedding)
        
        # Self-training step
        loss = self.train_kan(query_embedding, response_embedding)
        logging.info(f"KAN training loss: {loss}")
        
        return assistant_response  
        
    def update_emotional_state(self, feedback):
        self.emotional_state.update(feedback)

    def save_state(self, filename='kan_character_state.json'):
        state = {
            'emotional_position': self.emotional_state.position.tolist(),
            'emotional_velocity': self.emotional_state.velocity.tolist(),
            'memories': self.memory.memories,
            'memory_embeddings': [emb.tolist() for emb in self.memory.embeddings],
            'importance_scores': dict(self.memory.importance_scores),
            'system_prompt': self.system_prompt,
            'conversation_history': self.conversation_history,
            'kan_state': self.kan.state_dict(),
            'brain_cache': {str(k): v.tolist() for k, v in self.brain_cache.items()}
        }
        torch.save(state, filename)
        logger.info(f"Character state saved to {filename}")

    def load_state(self, filename='kan_character_state.json'):
        if not os.path.exists(filename):
            logger.warning(f"State file {filename} not found. Starting with a fresh state.")
            return

        state = torch.load(filename)
        self.emotional_state.position = np.array(state['emotional_position'])
        self.emotional_state.velocity = np.array(state['emotional_velocity'])
        self.memory.memories = state['memories']
        self.memory.embeddings = [np.array(emb) for emb in state['memory_embeddings']]
        self.memory.importance_scores = defaultdict(float, state['importance_scores'])
        self.system_prompt = state['system_prompt']
        self.conversation_history = state['conversation_history']
        self.kan.load_state_dict(state['kan_state'])
        self.brain_cache = {tuple(np.array(k)): torch.tensor(v) for k, v in state['brain_cache'].items()}
        logger.info(f"Character state loaded from {filename}")


def get_user_feedback():
    print("\nHow did the character's response make you feel?")
    print("1. Happy")
    print("2. Sad")
    print("3. Angry")
    print("4. Excited")
    print("5. Calm")
    print("6. Neutral")
    
    while True:
        try:
            choice = int(input("Enter the number of your emotion (1-6): "))
            if 1 <= choice <= 6:
                break
            else:
                print("Please enter a number between 1 and 6.")
        except ValueError:
            print("Please enter a valid number.")

    feedback_map = {
        1: [0.5, 0.3],    # Happy
        2: [-0.5, -0.3],  # Sad
        3: [-0.3, 0.5],   # Angry
        4: [0.3, 0.5],    # Excited
        5: [0.3, -0.3],   # Calm
        6: [0, 0]         # Neutral
    }

    return feedback_map[choice]

def interactive_session(character):
    print("Welcome to the KAN Emotional Character Interaction!")
    print("Please provide the character description in your first message.")
    print("Type 'exit' to end the conversation.")

    first_message = True
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() == 'exit':
            break
        
        if first_message:
            character.set_system_prompt(user_input)
            print("Character description set. You can now start interacting with the character.")
            first_message = False
            continue
        
        response = character.generate_response(user_input)
        print(f"\nCharacter: {response}")
        print(f"Current Emotion: {character.emotional_state.get_emotion()}")
        
        feedback = get_user_feedback()
        character.update_emotional_state(feedback)

    print("Thank you for interacting!")
    character.save_state()  # Save the final state

if __name__ == "__main__":
    logging.info("Starting KAN Emotional Character initialization...")
    try:
        character = KANEmotionalCharacter()
        logging.info("KANEmotionalCharacter initialized successfully.")
        
        logging.info("Attempting to load previous state...")
        character.load_state()
        logging.info("State loaded (or not found).")
        
        logging.info("Starting interactive session...")
        interactive_session(character)
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
    
    logging.info("Script execution completed.")
    input("Press Enter to exit...")