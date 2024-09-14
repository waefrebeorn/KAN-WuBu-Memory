import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
import json
import logging
from collections import defaultdict
import math
import random
import os

logging.basicConfig(level=logging.INFO)
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
    def __init__(self, max_memories=1000, embedding_size=768):
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
    def __init__(self, model_name="microsoft/phi-3", device="cuda" if torch.cuda.is_available() else "cpu"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        self.device = device
        self.emotional_state = EmotionalState()
        self.memory = AdvancedMemory()
        self.system_prompt = ""
        self.conversation_history = []
        self.kan = KAN(input_size=768, hidden_size=512, output_size=768).to(device)
        self.optimizer = torch.optim.Adam(self.kan.parameters(), lr=0.001)
        self.brain_cache = {}

    def set_system_prompt(self, prompt):
        self.system_prompt = prompt
        self.conversation_history = [{"role": "system", "content": prompt}]

    def get_embedding(self, text):
        with torch.no_grad():
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(self.device)
            outputs = self.model(**inputs, output_hidden_states=True)
            return outputs.hidden_states[-1].mean(dim=1).squeeze().cpu().numpy()

    def train_kan(self, input_embedding, target_embedding):
        input_tensor = torch.tensor(input_embedding, dtype=torch.float32).to(self.device)
        target_tensor = torch.tensor(target_embedding, dtype=torch.float32).to(self.device)
        
        self.optimizer.zero_grad()
        output = self.kan(input_tensor)
        loss = nn.MSELoss()(output, target_tensor)
        loss.backward()
        self.optimizer.step()
        
        return loss.item()

    @torch.no_grad()
    def generate_response(self, user_input, max_length=150):
        current_emotion = self.emotional_state.get_emotion()
        query_embedding = self.get_embedding(user_input)
        relevant_memories = self.memory.get_relevant_memories(query_embedding)
        
        self.conversation_history.append({"role": "user", "content": user_input})
        
        context = f"{self.system_prompt}\n\nCurrent Emotion: {current_emotion}\n"
        context += "Relevant Memories:\n" + "\n".join(relevant_memories) + "\n\n"
        context += "Conversation:\n"
        for message in self.conversation_history[1:]:
            context += f"{message['role'].capitalize()}: {message['content']}\n"
        context += f"Character: "

        input_ids = self.tokenizer.encode(context, return_tensors="pt").to(self.device)
        
        # Apply KAN to modify the input embedding
        input_embedding = self.get_embedding(context)
        modified_embedding = self.kan(torch.tensor(input_embedding, dtype=torch.float32).to(self.device)).cpu().numpy()
        
        # Use the modified embedding to influence the generation
        cache_key = tuple(modified_embedding)
        if cache_key in self.brain_cache:
            output = self.brain_cache[cache_key]
        else:
            output = self.model.generate(
                input_ids,
                max_length=input_ids.shape[1] + max_length,
                num_return_sequences=1,
                no_repeat_ngram_size=3,
                temperature=0.8,
                top_k=40,
                top_p=0.9,
            )
            self.brain_cache[cache_key] = output

        response = self.tokenizer.decode(output[0], skip_special_tokens=True)
        character_response = response.split("Character: ")[-1].strip()
        
        self.conversation_history.append({"role": "character", "content": character_response})
        
        response_embedding = self.get_embedding(character_response)
        self.memory.add_memory(f"Emotion: {current_emotion}, Interaction: {user_input} -> {character_response}", response_embedding)
        
        # Self-training step
        loss = self.train_kan(input_embedding, response_embedding)
        logger.info(f"KAN training loss: {loss}")
        
        return character_response

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
    character = KANEmotionalCharacter()
    
    # Try to load a previous state, if it exists
    character.load_state()
    
    interactive_session(character)