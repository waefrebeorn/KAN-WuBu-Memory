import os
import torch
import json
import numpy as np
import re
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity  
from transformers import LlamaForCausalLM, AutoTokenizer, LlamaConfig

# Define paths to the directories and files
SOURCE_DIR = "models/Llama_32_1B/"
WEIGHTS_DIR = os.path.join(SOURCE_DIR, "offload")
MODEL_JSON_PATH = os.path.join(SOURCE_DIR, "config.json")

# Initialize logging
logging.basicConfig(level=logging.INFO)

# Load the configuration from the JSON file
def load_configuration(model_json_path):
    with open(model_json_path, "r") as f:
        config_data = json.load(f)
    config = LlamaConfig(**config_data)
    return config

# Use AutoTokenizer instead of LlamaTokenizer to resolve class conflicts  
def load_tokenizer(source_dir):
    return AutoTokenizer.from_pretrained(source_dir)

# Load the model configuration
logging.info(f"Loading model configuration from: {MODEL_JSON_PATH}")  
config = load_configuration(MODEL_JSON_PATH)

# Initialize an empty model based on the configuration
model = LlamaForCausalLM(config)
logging.info("Initialized empty LLaMA model.")

# Load the offloaded weights from the `.dat` files  
def load_dat_file(file_path, dtype):
    with open(file_path, 'rb') as f:
        tensor_data = np.fromfile(f, dtype=dtype)
    loaded_tensor = torch.tensor(tensor_data)
    
    # If dtype was mapped to float32 for bfloat16 compatibility, convert back
    if dtype == np.float32 and "bfloat16" in file_path:  
        loaded_tensor = loaded_tensor.to(torch.bfloat16)
    return loaded_tensor

def load_offloaded_weights(model, weights_dir):
    for name, param in model.named_parameters():
        file_name = name.replace('.', '_') + ".dat"
        file_path = os.path.join(weights_dir, file_name)

        if os.path.exists(file_path):
            dtype_map = {
                torch.float16: np.float16,
                torch.float32: np.float32,
                torch.int64: np.int64,   
                torch.int32: np.int32,
                torch.bfloat16: np.float32,
            }
            expected_dtype = dtype_map.get(param.dtype, np.float32)
            logging.info(f"Loading {file_name} into {name} with expected type {expected_dtype}")  
            loaded_tensor = load_dat_file(file_path, expected_dtype).view_as(param)

            if param.dtype == torch.bfloat16:
                loaded_tensor = loaded_tensor.to(torch.bfloat16)
            
            param.data.copy_(loaded_tensor.to("cuda"))
        else:
            logging.warning(f"Warning: {file_name} not found in offloaded directory.")  

# Load the weights into the model
load_offloaded_weights(model, WEIGHTS_DIR)

# Move the model to GPU for inference
model.to('cuda')  
model.eval()

# Use AutoTokenizer to handle any tokenizer class discrepancies
logging.info(f"Loading tokenizer from directory: {SOURCE_DIR}")  
tokenizer = load_tokenizer(SOURCE_DIR)

# Helper function to calculate entropy
def calculate_entropy(probs):
    log_probs = torch.log(probs + 1e-10)  # Add small epsilon to avoid log(0)
    entropy = -torch.sum(probs * log_probs, dim=-1)
    return entropy

def summarize_history(history, tokenizer, max_length=100):
    if not history:
        return ""
    
    # Concatenate the history into a single string
    history_text = " ".join(history)
    
    # Tokenize the history text
    history_tokens = tokenizer.encode(history_text, truncation=True, max_length=max_length)
    
    # Decode the summarized history tokens back into text  
    summarized_history = tokenizer.decode(history_tokens)
    
    return summarized_history

def evaluate_response_quality(response, user_input, tokenizer, threshold=0.75):
    # Tokenize the response and user input
    response_tokens = set(tokenizer.encode(response))
    user_input_tokens = set(tokenizer.encode(user_input))
    
    # Calculate the overlap between response and user input tokens
    overlap = len(response_tokens.intersection(user_input_tokens)) 
    overlap_ratio = overlap / len(user_input_tokens)
    
    # Calculate the coherence of the response
    coherence_score = 0.0  # Implement a coherence scoring mechanism
    
    # Evaluate the relevance and quality of the response
    relevance_score = overlap_ratio
    quality_score = 0.5 * overlap_ratio + 0.5 * coherence_score
    
    return quality_score >= threshold

def adjust_layers(model, quality_score, threshold=0.75):
    if quality_score < threshold:  
        # Reduce the number of layers
        num_layers = max(1, model.config.num_hidden_layers // 2)
    else:
        # Increase the number of layers
        num_layers = min(model.config.num_hidden_layers * 2, 48)  
    
    # Adjust the model's layers
    model.config.num_hidden_layers = num_layers
    model.resize_token_embeddings(len(tokenizer))
    
    return model

def generate_response(input_text, model, tokenizer, max_new_tokens=50, pad_token_id=128001, history=[], context_limit=512):
    # Clean the history to avoid redundant prompts
    history = [line for line in history if line.strip()]  # Remove empty lines
    
    # Create a simplified context prompt from the last few exchanges
    prompt = f"{' '.join(history[-3:])}\nUser: {input_text}\n" if history else f"User: {input_text}\n"
    
    # Prepare inputs for the model  
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=context_limit).to("cuda")

    # Initialize response and keep track of tokens for refinement
    refined_response = ""
    refined_token_ids = []  
    full_response = ""

    # Iteratively generate responses
    for iteration in range(10):  # Number of iterations can be adjusted as needed
        logging.info(f"Iteration {iteration + 1}: Generating tokens...")

        # Generate a chunk of response
        with torch.no_grad():
            outputs = model.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=max_new_tokens,  # Control new tokens per iteration
                do_sample=True,  
                temperature=0.7,
                top_k=50,
                top_p=0.9,
                repetition_penalty=1.2,
                pad_token_id=pad_token_id,
                early_stopping=True,  
                output_scores=True,  # Enable outputting scores/logits
                return_dict_in_generate=True  # Return full output with scores
            )

        # Retrieve the generated token IDs and logits (probabilities)
        token_ids = outputs.sequences[0].tolist()  
        refined_token_ids.extend(token_ids)

        # Convert logits to probabilities and calculate entropy
        logits = outputs.scores  # This contains the logits for each generated token
        probs = torch.softmax(torch.stack(logits), dim=-1)  
        entropies = calculate_entropy(probs)

        # Decode the generated response
        refined_response = tokenizer.decode(refined_token_ids, skip_special_tokens=True).strip()

        # Evaluate the quality of the generated response
        quality_score = evaluate_response_quality(refined_response, input_text, tokenizer)
        
        # Adjust layers based on the quality score
        model = adjust_layers(model, quality_score)
        
        full_response = f"{full_response} {refined_response}".strip()
    
    # Append final cleaned response to history
    history.append(f"User: {input_text}\nModel: {full_response}")
    
    # Trim history to avoid excessive accumulation 
    if len(history) > 6:
        history = history[-6:]

    return full_response, history

def user_input_loop(model, tokenizer):
    print("\n--- LLaMA Instruct Model Interactive Query ---") 
    print("Type 'exit' to quit.")
    history = []  # Initialize a history buffer to keep track of conversation
    
    while True:
        user_input = input("\nEnter your query: ")
        if user_input.lower() == 'exit':  
            print("Exiting...")
            break
        
        # Summarize the conversation history
        summarized_history = summarize_history(history, tokenizer)
        
        # Generate response using the LLaMA model
        response, history = generate_response(user_input, model, tokenizer, history=summarized_history)  
        print(f"Model Response: {response}")
        
        # Get user feedback on the response
        feedback = input("Please provide feedback on the response (good/bad): ")
        
        # Fine-tune the model based on user feedback
        if feedback.lower() == 'good':
            # Reinforce the response pattern 
            # ... (code for reinforcement learning)
        elif feedback.lower() == 'bad':
            # Penalize the response pattern
            # ... (code for reinforcement learning)
    
    # Save the final conversation history
    with open("conversation_history.json", "w") as f:  
        json.dump(history, f)

# Start the interactive query loop with the refined response generation
logging.info("Model loaded successfully. You can now query the model.")
user_input_loop(model, tokenizer)