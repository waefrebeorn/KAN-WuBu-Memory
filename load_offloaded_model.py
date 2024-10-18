import os
import torch
import json
import logging
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import LlamaForCausalLM, AutoTokenizer, LlamaConfig

# --------------------------- Configuration --------------------------- #

# Define paths to the directories and files
SOURCE_DIR = "models/Llama_32_1B/"
WEIGHTS_DIR = os.path.join(SOURCE_DIR, "offload")
MODEL_JSON_PATH = os.path.join(SOURCE_DIR, "config.json")

# Set device to GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type != "cuda":
    raise SystemExit("CUDA-enabled GPU not found. Please ensure a compatible GPU is available.")

# --------------------------- Logging Setup --------------------------- #

# Initialize logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s:%(levelname)s:%(name)s: %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# --------------------------- Model Loading --------------------------- #

def load_configuration(config_path):
    """
    Load model configuration from a JSON file.
    """
    with open(config_path, "r") as f:
        config_data = json.load(f)
    config = LlamaConfig(**config_data)
    logger.info(f"Model configuration loaded from {config_path}")
    return config

def load_tokenizer(source_dir):
    """
    Load and configure the tokenizer, adding a custom pad token if necessary.
    """
    tokenizer = AutoTokenizer.from_pretrained(source_dir)
    finetune_pad_token = "<|finetune_right_pad_id|>"
    
    if finetune_pad_token not in tokenizer.get_vocab():
        tokenizer.add_special_tokens({'pad_token': finetune_pad_token})
        logger.info(f"Added pad_token: '{finetune_pad_token}' to tokenizer.")
    else:
        tokenizer.pad_token = finetune_pad_token
        logger.info(f"Assigned existing '{finetune_pad_token}' as pad_token.")
    
    return tokenizer

def load_offloaded_weights(model, weights_dir):
    """
    Load model weights from .dat files and assign them to the model's parameters.
    """
    for name, param in model.named_parameters():
        file_name = f"{name.replace('.', '_')}.dat"
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
            logger.info(f"Loading {file_name} into {name} with expected type {expected_dtype}")
            
            tensor_data = np.fromfile(file_path, dtype=expected_dtype)
            loaded_tensor = torch.from_numpy(tensor_data).to(device)

            if param.dtype == torch.bfloat16:
                loaded_tensor = loaded_tensor.to(torch.bfloat16)

            with torch.no_grad():
                param.data.copy_(loaded_tensor.view_as(param))
        else:
            logger.warning(f"Weight file {file_path} not found.")

    logger.info("All available weights loaded successfully.")

# --------------------------- Response Quality Management --------------------------- #

class EnhancedResponseQualityManager:
    """
    Manages the quality of responses by evaluating relevance, fluency, and structure.
    """
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.tfidf_vectorizer = TfidfVectorizer()
    
    def validate_response(self, user_input, model_response):
        """
        Validate the generated response based on relevance, fluency, and structure.
        """
        relevance = self._calculate_relevance(user_input, model_response)
        fluency = self._check_fluency(model_response)
        structure = self._check_structure(model_response)
        
        logger.debug(f"Validation Metrics - Relevance: {relevance}, Fluency: {fluency}, Structure: {structure}")
        
        return relevance > 0.5 and fluency and structure
    
    def _calculate_relevance(self, user_input, response):
        """
        Calculate the relevance of the response to the user input.
        """
        tokens_input = set(self.tokenizer.tokenize(user_input))
        tokens_response = set(self.tokenizer.tokenize(response))
        overlap = len(tokens_input & tokens_response)
        
        tfidf_matrix = self.tfidf_vectorizer.fit_transform([user_input, response])
        cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        
        relevance_score = (0.5 * overlap / max(len(tokens_input), 1)) + (0.5 * cosine_sim)
        return relevance_score
    
    def _check_fluency(self, response):
        """
        Check the fluency of the response to ensure it is coherent.
        """
        if len(response.split()) < 3:
            return False
        if re.search(r'[^\x00-\x7F]+', response):
            return False
        return True
    
    def _check_structure(self, response):
        """
        Ensure the response has proper sentence structure.
        """
        if not response:
            return False
        if not response[0].isupper():
            return False
        if response[-1] not in '.!?':
            return False
        return True

# --------------------------- Context Management --------------------------- #

def manage_context(history, input_text, tokenizer, max_context_length=512):
    """
    Manage and prepare the context for the model by maintaining relevant history.
    """
    # Clean empty lines
    history = [line for line in history if line.strip()]
    
    # Limit history to the last 3 exchanges
    recent_history = history[-3:]
    
    # Create the prompt
    if recent_history:
        prompt = "\n".join(recent_history) + f"\nUser: {input_text}\nModel:"
    else:
        prompt = f"User: {input_text}\nModel:"
    
    # Tokenize to ensure it doesn't exceed max_context_length
    tokenized = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_context_length)
    
    # Decode back to string
    prompt = tokenizer.decode(tokenized["input_ids"][0], skip_special_tokens=True)
    
    return prompt

# --------------------------- Response Generation --------------------------- #

def generate_response(input_text, model, tokenizer, history, quality_manager, max_tokens=200):
    """
    Generate a response from the model based on the input_text and conversation history.
    """
    prompt = manage_context(history, input_text, tokenizer)
    
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512
    ).to(device)
    
    # Generate the response using beam search for better quality
    outputs = model.generate(
        inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=max_tokens,
        temperature=0.8,
        top_k=50,
        top_p=0.9,
        repetition_penalty=1.5,
        num_beams=3,  # Beam search with 3 beams
        early_stopping=True
    )
    
    # Decode the response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    
    # Extract the model's response portion
    response = response.split("Model:")[-1].strip()
    response = re.sub(r'\s+', ' ', response)
    
    # Append to history
    history.append(f"User: {input_text}\nModel: {response}")
    if len(history) > 6:
        history = history[-6:]
    
    # Validate the response
    if not quality_manager.validate_response(input_text, response):
        logger.warning("Response failed quality checks. Regenerating...")
        # Attempt regeneration once
        outputs = model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=max_tokens,
            temperature=0.8,
            top_k=50,
            top_p=0.9,
            repetition_penalty=1.5,
            num_beams=3,
            early_stopping=True
        )
        response = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        response = response.split("Model:")[-1].strip()
        response = re.sub(r'\s+', ' ', response)
        history[-1] = f"User: {input_text}\nModel: {response}"
        if not quality_manager.validate_response(input_text, response):
            logger.error("Regenerated response also failed quality checks.")
            response = "I'm sorry, but I couldn't provide a satisfactory response to that."
    
    return response, history

# --------------------------- Interactive Loop --------------------------- #

def interactive_query(model, tokenizer, quality_manager):
    """
    Start an interactive loop to accept user queries and generate model responses.
    """
    print("\n--- LLaMA Instruct Model Interactive Query ---")
    print("Type 'exit' to quit.\n")
    history = []
    
    while True:
        try:
            user_input = input("Enter your query: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting...")
            break

        if user_input.lower() == 'exit':
            print("Exiting...")
            break

        if not user_input:
            print("Please enter a valid query.")
            continue

        response, history = generate_response(
            user_input,
            model,
            tokenizer,
            history,
            quality_manager
        )

        print(f"Model Response: {response}\n")

# --------------------------- Main Execution --------------------------- #

def main():
    # Load model configuration
    config = load_configuration(MODEL_JSON_PATH)
    
    # Initialize the model and move it to GPU
    model = LlamaForCausalLM(config).to(device)
    logger.info("Initialized LLaMA model on GPU.")
    
    # Load model weights
    load_offloaded_weights(model, WEIGHTS_DIR)
    
    # Ensure the model is in evaluation mode
    model.eval()
    logger.info("Model is set to evaluation mode.")
    
    # Load tokenizer
    tokenizer = load_tokenizer(SOURCE_DIR)
    
    # Resize token embeddings if a new pad token was added
    if tokenizer.pad_token == "<|finetune_right_pad_id|>":
        if tokenizer.pad_token not in tokenizer.get_vocab():
            model.resize_token_embeddings(len(tokenizer))
            logger.info("Resized model token embeddings to accommodate the new pad_token.")
        else:
            logger.info("pad_token already exists in the tokenizer's vocabulary. No need to resize embeddings.")
    
    # Initialize the Response Quality Manager
    quality_manager = EnhancedResponseQualityManager(tokenizer)
    
    # Start the interactive query loop
    logger.info("Model loaded successfully. You can now query the model.")
    interactive_query(model, tokenizer, quality_manager)

if __name__ == "__main__":
    main()
