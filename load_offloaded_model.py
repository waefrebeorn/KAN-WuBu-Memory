import os
import torch
import json
import logging
import re
import time
import numpy as np
from math import log2
from transformers import LlamaForCausalLM, AutoTokenizer, LlamaConfig

# --------------------------- Configuration --------------------------- #

# Define paths to the directories and files
SOURCE_DIR = "models/Llama_32_1B/"
WEIGHTS_DIR = os.path.join(SOURCE_DIR, "offload")
MODEL_JSON_PATH = os.path.join(SOURCE_DIR, "config.json")

# Define maximum context length
MAX_CONTEXT_LENGTH = 2048  # Increased to allow for longer contexts

# Define logging configuration
LOG_FORMAT = "%(asctime)s:%(levelname)s:%(name)s: %(message)s"
LOG_LEVEL = logging.INFO

# --------------------------- Logging Setup --------------------------- #

# Initialize logging
logging.basicConfig(
    level=LOG_LEVEL,
    format=LOG_FORMAT,
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# --------------------------- Device Configuration --------------------------- #

# Set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type != "cuda":
    logger.error("CUDA-enabled GPU not found. Please ensure a compatible GPU is available.")
    raise SystemExit("CUDA-enabled GPU not found.")

logger.info(f"Using device: {device}")

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
    Manages the quality of responses by evaluating relevance, fluency, structure, and entropy.
    """
    # Define entropy thresholds
    LOW_ENTROPY_THRESHOLD = 2.0  # Too predictable
    HIGH_ENTROPY_THRESHOLD = 25.0  # Increased to allow more diversity
    WINDOW_SIZE = 50  # Window size for windowed entropy

    def __init__(self, tokenizer, model):
        self.tokenizer = tokenizer
        self.model = model

    def validate_response(self, user_input, model_response):
        """
        Validate the generated response based on relevance, fluency, structure, and entropy.
        Uses windowed entropy and defines an acceptable entropy range.
        """
        relevance = self._calculate_relevance(user_input, model_response)
        fluency = self._check_fluency(model_response)
        structure = self._check_structure(model_response)
        mean_entropy, std_entropy = self._calculate_windowed_entropy(model_response)

        # Log metrics for debugging
        logger.debug(f"Validation Metrics - Relevance: {relevance:.2f}, Fluency: {fluency}, Structure: {structure}, Mean Entropy: {mean_entropy:.2f}, Std Entropy: {std_entropy:.2f}")

        # Determine validity based on entropy range
        if not (self.LOW_ENTROPY_THRESHOLD <= mean_entropy <= self.HIGH_ENTROPY_THRESHOLD):
            logger.info(f"Response entropy {mean_entropy:.2f} out of range ({self.LOW_ENTROPY_THRESHOLD}, {self.HIGH_ENTROPY_THRESHOLD})")
            # Implement dynamic thresholding: Allow if relevance and fluency are high
            if relevance > 0.5 and fluency:
                logger.info("High relevance and fluency compensate for entropy out of range.")
                return True
            return False

        # Adjust validation logic based on input length
        if len(user_input.split()) < 3:
            # For short queries, relax the relevance and structure
            return fluency and mean_entropy < self.HIGH_ENTROPY_THRESHOLD
        else:
            # For longer queries, require fluency and acceptable entropy
            return (relevance > 0.3 or fluency) and structure and (mean_entropy < self.HIGH_ENTROPY_THRESHOLD)

    def _calculate_relevance(self, user_input, response):
        """
        Calculate the relevance of the response to the user input using token overlap.
        """
        tokens_input = set(self.tokenizer.tokenize(user_input.lower()))
        tokens_response = set(self.tokenizer.tokenize(response.lower()))
        overlap = len(tokens_input & tokens_response)
        relevance_score = overlap / max(len(tokens_input), 1)
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

    def _calculate_windowed_entropy(self, response):
        """
        Calculate the mean and standard deviation of entropy over windows of tokens.
        """
        tokens = self.tokenizer.encode(response, return_tensors='pt').to(device)
        with torch.no_grad():
            outputs = self.model(tokens, labels=tokens)
            logits = outputs.logits  # Shape: (1, sequence_length, vocab_size)

        # Calculate probabilities
        probabilities = torch.softmax(logits, dim=-1)  # Shape: (1, sequence_length, vocab_size)

        # Gather the probabilities of the actual tokens
        token_probs = probabilities.gather(2, tokens.unsqueeze(-1)).squeeze(-1)  # Shape: (1, sequence_length)

        # Calculate entropy for each token
        token_entropy = -torch.log2(token_probs + 1e-10)  # Avoid log(0)
        token_entropy = token_entropy.squeeze(0).cpu().numpy()  # Shape: (sequence_length,)

        # Calculate windowed entropy
        window_size = self.WINDOW_SIZE
        num_windows = max(1, len(token_entropy) // window_size)
        entropy_values = []

        for i in range(num_windows):
            start = i * window_size
            end = start + window_size
            window = token_entropy[start:end]
            if len(window) == 0:
                continue
            window_entropy = np.mean(window)
            entropy_values.append(window_entropy)

        if not entropy_values:
            mean_entropy = 0.0
            std_entropy = 0.0
        else:
            mean_entropy = np.mean(entropy_values)
            std_entropy = np.std(entropy_values)

        return mean_entropy, std_entropy

# --------------------------- Context Management --------------------------- #

def dynamic_context(history, user_input, tokenizer, max_context=MAX_CONTEXT_LENGTH):
    """
    Dynamically manage and prepare the context for the model by selecting the most relevant history.
    """
    if not history:
        prompt = f"User: {user_input}\nModel:"
        return prompt

    # Calculate relevance scores for each historical exchange
    relevance_scores = []
    for exchange in history:
        try:
            user_part, model_part = exchange.split("\nModel: ")
            relevance = calculate_cosine_similarity(user_input, user_part, tokenizer)
            relevance_scores.append(relevance)
        except ValueError:
            relevance_scores.append(0.0)  # If split fails, assign zero relevance

    # Select top 3 most relevant exchanges
    top_indices = np.argsort(relevance_scores)[-3:]
    selected_history = [history[i] for i in top_indices if i < len(history)]
    selected_history = selected_history[::-1]  # Reverse to maintain chronological order

    # Construct the prompt
    prompt = "\n".join(selected_history) + f"\nUser: {user_input}\nModel:"

    # Ensure the prompt does not exceed max_context
    tokenized = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_context)
    prompt = tokenizer.decode(tokenized["input_ids"][0], skip_special_tokens=True)

    return prompt

def calculate_cosine_similarity(text1, text2, tokenizer):
    """
    Calculate cosine similarity between two texts using their token embeddings.
    """
    tokens1 = tokenizer.encode(text1, return_tensors='pt').to(device)
    tokens2 = tokenizer.encode(text2, return_tensors='pt').to(device)

    with torch.no_grad():
        embeddings1 = model.transformer.wte(tokens1)
        embeddings2 = model.transformer.wte(tokens2)

    # Average embeddings
    embedding1 = embeddings1.mean(dim=1)
    embedding2 = embeddings2.mean(dim=1)

    # Calculate cosine similarity
    cosine_sim = torch.nn.functional.cosine_similarity(embedding1, embedding2).item()
    return cosine_sim

def sanitize_input(user_input):
    """
    Sanitize user input to prevent injection attacks and ensure safety.
    """
    sanitized = re.sub(r'[^\w\s.,!?]', '', user_input)  # Remove unwanted characters
    return sanitized[:500]  # Limit input length

# --------------------------- Persona and Tone Management --------------------------- #

def adjust_tone(user_input, base_response):
    """
    Adjust the tone of the response based on user input.
    """
    if any(word in user_input.lower() for word in ["thank you", "thanks"]):
        return f"You're welcome! {base_response}"
    elif any(word in user_input.lower() for word in ["please", "kindly", "could you"]):
        return f"Certainly! {base_response}"
    elif "persona:formal" in user_input.lower():
        return f"Certainly. {base_response}"
    elif "persona:casual" in user_input.lower():
        return f"Sure thing! {base_response}"
    return base_response  # Default to neutral tone

# --------------------------- Response Generation --------------------------- #

def generate_response(input_text, model, tokenizer, history, quality_manager):
    """
    Generate a response from the model based on the input_text and conversation history.
    Dynamically adjusts max_new_tokens based on input length.
    """
    # Sanitize input
    sanitized_input = sanitize_input(input_text)

    # Manage context
    prompt = dynamic_context(history, sanitized_input, tokenizer)

    # Tokenize input
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_CONTEXT_LENGTH
    ).to(device)

    # Determine dynamic max_new_tokens based on input length
    input_length = len(sanitized_input.split())
    default_max_tokens = 200
    max_tokens = min(1024, max(default_max_tokens, input_length * 10))  # Adjust multiplier as needed

    # Generate the response using beam search and sampling
    outputs = model.generate(
        inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=max_tokens,  # Dynamic max tokens
        do_sample=True,  # Enable sampling
        temperature=0.8,
        top_k=50,
        top_p=0.9,
        repetition_penalty=1.5,
        pad_token_id=tokenizer.pad_token_id,  # Correct pad_token_id
        num_beams=3,  # Beam search for structured responses
        early_stopping=True
    )

    # Decode the response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

    # Extract the model's response portion
    response = response.split("Model:")[-1].strip()
    response = re.sub(r'\s+', ' ', response)

    # Append to history
    history.append(f"User: {sanitized_input}\nModel: {response}")
    if len(history) > 6:
        history = history[-6:]

    # Validate the response
    start_time = time.time()
    is_valid = quality_manager.validate_response(sanitized_input, response)
    response_time = time.time() - start_time
    logger.info(f"Response time: {response_time:.2f}s, Valid: {is_valid}")

    if not is_valid:
        mean_entropy, _ = quality_manager._calculate_windowed_entropy(response)
        logger.warning(f"Failed response due to metrics. Showing for debugging: Relevance: {quality_manager._calculate_relevance(sanitized_input, response):.2f}, Mean Entropy: {mean_entropy:.2f}")
        print(f"Failed Response (for debugging): {response}")

        # Attempt regeneration once for debugging purposes
        outputs = model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=max_tokens,  # Same max tokens
            do_sample=True,
            temperature=0.8,
            top_k=50,
            top_p=0.9,
            repetition_penalty=1.5,
            pad_token_id=tokenizer.pad_token_id,
            num_beams=3,
            early_stopping=True
        )
        regenerated_response = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        regenerated_response = regenerated_response.split("Model:")[-1].strip()
        regenerated_response = re.sub(r'\s+', ' ', regenerated_response)
        history[-1] = f"User: {sanitized_input}\nModel: {regenerated_response}"

        # Re-validate the regenerated response
        is_valid = quality_manager.validate_response(sanitized_input, regenerated_response)
        if not is_valid:
            mean_entropy, _ = quality_manager._calculate_windowed_entropy(regenerated_response)
            logger.error("Regenerated response also failed quality checks. Displaying for debugging.")
            logger.warning(f"Regenerated Failed Response Metrics: Relevance: {quality_manager._calculate_relevance(sanitized_input, regenerated_response):.2f}, Mean Entropy: {mean_entropy:.2f}")
            print(f"Regenerated Failed Response (for debugging): {regenerated_response}")
            response = regenerated_response  # Keep the failed regenerated response for debugging
        else:
            response = adjust_tone(sanitized_input, regenerated_response)
    else:
        response = adjust_tone(sanitized_input, response)

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
    global model  # Needed for cosine similarity calculation

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
    quality_manager = EnhancedResponseQualityManager(tokenizer, model)

    # Start the interactive query loop
    logger.info("Model loaded successfully. You can now query the model.")
    interactive_query(model, tokenizer, quality_manager)

if __name__ == "__main__":
    main()
