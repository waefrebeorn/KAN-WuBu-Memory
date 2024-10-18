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

SOURCE_DIR = "models/Llama_32_1B/"
WEIGHTS_DIR = os.path.join(SOURCE_DIR, "offload")
MODEL_JSON_PATH = os.path.join(SOURCE_DIR, "config.json")
MAX_CONTEXT_LENGTH = 2048

LOG_FORMAT = "%(asctime)s:%(levelname)s:%(name)s: %(message)s"
LOG_LEVEL = logging.INFO

# --------------------------- Logging Setup --------------------------- #

logging.basicConfig(
    level=LOG_LEVEL,
    format=LOG_FORMAT,
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# --------------------------- Device Configuration --------------------------- #

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type != "cuda":
    logger.error("CUDA-enabled GPU not found. Please ensure a compatible GPU is available.")
    raise SystemExit("CUDA-enabled GPU not found.")

logger.info(f"Using device: {device}")

# --------------------------- Model Loading --------------------------- #

def load_configuration(config_path):
    with open(config_path, "r") as f:
        config_data = json.load(f)
    config = LlamaConfig(**config_data)
    logger.info(f"Model configuration loaded from {config_path}")
    return config

def load_tokenizer(source_dir):
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

class ImprovedResponseQualityManager:
    LOW_ENTROPY_THRESHOLD = 2.0
    HIGH_ENTROPY_THRESHOLD = 35.0  # Increased to allow more variety
    WINDOW_SIZE = 50
    EOT_TOKENS = ['�', '\ufffd']  # Add more EOT tokens as needed

    def __init__(self, tokenizer, model):
        self.tokenizer = tokenizer
        self.model = model

    def validate_response(self, user_input, model_response):
        # Remove EOT tokens before further processing
        clean_response = self.remove_eot_tokens(model_response)
        
        relevance = self._calculate_relevance(user_input, clean_response)
        fluency = self._check_fluency(clean_response)
        structure = self._check_structure(clean_response)
        mean_entropy, std_entropy = self._calculate_windowed_entropy(clean_response)

        logger.debug(f"Validation Metrics - Relevance: {relevance:.2f}, Fluency: {fluency}, Structure: {structure}, Mean Entropy: {mean_entropy:.2f}, Std Entropy: {std_entropy:.2f}")

        # Adjust entropy thresholds and ensure relevance compensates for out-of-range entropy
        if not (self.LOW_ENTROPY_THRESHOLD <= mean_entropy <= self.HIGH_ENTROPY_THRESHOLD):
            logger.info(f"Response entropy {mean_entropy:.2f} out of range ({self.LOW_ENTROPY_THRESHOLD}, {self.HIGH_ENTROPY_THRESHOLD})")
            if relevance > 0.5 and fluency:
                logger.info("High relevance and fluency compensate for entropy out of range.")
                return True
            return False

        return (relevance > 0.3 or fluency) and structure

    def remove_eot_tokens(self, response):
        for token in self.EOT_TOKENS:
            response = response.rstrip(token)
        return response.strip()

    def _calculate_relevance(self, user_input, response):
        tokens_input = set(self.tokenizer.tokenize(user_input.lower()))
        tokens_response = set(self.tokenizer.tokenize(response.lower()))
        overlap = len(tokens_input & tokens_response)
        relevance_score = overlap / max(len(tokens_input), 1)
        return relevance_score

    def _check_fluency(self, response):
        if len(response.split()) < 3:
            return False
        if re.search(r'[^\x00-\x7F]+', response):
            return False
        return True

    def _check_structure(self, response):
        if not response:
            return False
        if not response[0].isupper():
            return False
        if response[-1] not in '.!?':
            return False
        return True

    def _calculate_windowed_entropy(self, response):
        tokens = self.tokenizer.encode(response, return_tensors='pt').to(device)
        with torch.no_grad():
            outputs = self.model(tokens, labels=tokens)
            logits = outputs.logits

        probabilities = torch.softmax(logits, dim=-1)
        token_probs = probabilities.gather(2, tokens.unsqueeze(-1)).squeeze(-1)
        token_entropy = -torch.log2(token_probs + 1e-10)
        token_entropy = token_entropy.squeeze(0).cpu().numpy()

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

# --------------------------- Entropy-Based Temperature and Sampling Adjustment --------------------------- #

def adjust_temperature_based_on_entropy(entropy, low_threshold=2.0, high_threshold=25.0):
    if entropy > high_threshold:
        return max(0.7, 1.0 - ((entropy - high_threshold) / 10))  # Lower the temperature
    elif entropy < low_threshold:
        return min(1.5, 1.0 + ((low_threshold - entropy) / 10))  # Increase temperature
    return 1.0  # Default temperature

def adjust_sampling_parameters(entropy, low_k=50, high_k=5, low_p=0.95, high_p=0.8):
    if entropy > 20.0:
        return high_k, high_p  # Focused, deterministic sampling
    elif entropy < 10.0:
        return low_k, low_p  # More diverse sampling
    return (int((high_k + low_k) / 2), (high_p + low_p) / 2)

def sample_token(probs, top_k, top_p, temperature):
    # Apply temperature
    probs = probs / temperature
    # Apply top_k and top_p
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    # Remove tokens with cumulative probability above the threshold
    sorted_indices_to_remove = cumulative_probs > top_p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = False

    indices_to_remove = sorted_indices_to_remove.scatter(
        dim=1, index=sorted_indices, src=sorted_indices_to_remove
    )
    probs[indices_to_remove] = 0.0
    probs = probs / probs.sum()

    # Top-K
    if top_k > 0:
        topk_probs, topk_indices = torch.topk(probs, top_k, dim=-1)
        probs = topk_probs
        indices = topk_indices
    else:
        indices = torch.multinomial(probs, num_samples=1)

    return indices.squeeze(0)

# --------------------------- Context Management --------------------------- #

def calculate_cosine_similarity(text1, text2, tokenizer, model):
    tokens1 = tokenizer.encode(text1, return_tensors='pt').to(device)
    tokens2 = tokenizer.encode(text2, return_tensors='pt').to(device)

    with torch.no_grad():
        embeddings1 = model.model.embed_tokens(tokens1)
        embeddings2 = model.model.embed_tokens(tokens2)

    embedding1 = embeddings1.mean(dim=1)
    embedding2 = embeddings2.mean(dim=1)

    cosine_sim = torch.nn.functional.cosine_similarity(embedding1, embedding2).item()
    return cosine_sim

def sanitize_input(user_input):
    sanitized = re.sub(r'[^\w\s.,!?]', '', user_input)
    return sanitized[:500]

def summarize_memories(relevant_memories, tokenizer, max_length=512):
    # Simple summarization: concatenate and truncate
    summary = " ".join([f"User said: {entry['user']} Model replied: {entry['model']}" for entry in relevant_memories])
    if len(summary) > max_length:
        summary = summary[:max_length]
    return summary

def recall_important_memories(history, user_input, tokenizer, model):
    # Identify key moments in the history that are relevant to the current input
    relevance_scores = []
    for exchange in history:
        user_part = exchange["user"]
        relevance = calculate_cosine_similarity(user_input, user_part, tokenizer, model)
        relevance_scores.append(relevance)
    
    # Select top 3 most relevant exchanges
    top_indices = np.argsort(relevance_scores)[-3:]
    selected_history = [history[i] for i in reversed(top_indices) if i < len(history)]
    
    # Summarize the selected history
    memory = summarize_memories(selected_history, tokenizer)
    return memory

def create_base_prompt(user_input, memory):
    BASE_PROMPT = "This is a conversation between a user and an AI model. The AI maintains a friendly and helpful demeanor without referencing past conversations unless necessary."
    base_prompt = f"{BASE_PROMPT}\nUser: {user_input}\nAI:"
    if memory:
        # Implicitly add memory as context but not part of the system response
        base_prompt += f"\n[Memory]: {memory}\nAI:"
    return base_prompt

def improved_dynamic_context(history, user_input, tokenizer, model, max_context=MAX_CONTEXT_LENGTH):
    if not history:
        prompt = create_base_prompt(user_input, "")
        return prompt, [{"user": user_input, "model": ""}]
    
    # Recall important memories
    memory = recall_important_memories(history, user_input, tokenizer, model)
    
    # Create the base prompt with memory
    prompt = create_base_prompt(user_input, memory)
    
    # Tokenize and truncate if necessary
    tokenized = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_context)
    truncated_prompt = tokenizer.decode(tokenized["input_ids"][0], skip_special_tokens=True)
    
    # Update history
    new_history_entry = {"user": user_input, "model": ""}
    updated_history = history + [new_history_entry]
    
    return truncated_prompt, updated_history

# --------------------------- Response Generation --------------------------- #

def generate_macroprocessed_response(prompt, model, tokenizer):
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_CONTEXT_LENGTH
    ).to(device)
    input_ids = inputs["input_ids"]

    max_tokens = 200  # Adjust as needed
    generated_ids = input_ids

    token_log = []

    for _ in range(max_tokens):
        outputs = model(generated_ids)
        logits = outputs.logits[:, -1, :]
        probs = torch.softmax(logits, dim=-1)
        
        entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1).item()
        temperature = adjust_temperature_based_on_entropy(entropy)
        top_k, top_p = adjust_sampling_parameters(entropy)
        
        # Sample a single token (ensure it's a single token)
        token_id = sample_token(probs, top_k, top_p, temperature)

        # If `token_id` is not scalar, we need to extract the single token
        if token_id.dim() > 0:
            token_id = token_id.squeeze(0)  # Ensure single scalar token

        generated_ids = torch.cat([generated_ids, token_id.unsqueeze(0)], dim=-1)
        
        token_log.append({
            "token_id": token_id.item(),  # This should now work correctly
            "entropy": entropy,
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p
        })
        
        # Check for end-of-sequence token
        if token_id.item() in tokenizer.all_special_ids:
            break

    # Log token-level details for debugging
    for log in token_log:
        logger.info(f"Token: {log['token_id']}, Entropy: {log['entropy']:.2f}, Temperature: {log['temperature']:.2f}, top_k: {log['top_k']}, top_p: {log['top_p']}")

    response = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    response = response.split("AI:")[-1].strip()
    response = remove_memory_recall(response)

    return response

def remove_memory_recall(response):
    # Remove memory recall tags from the response
    response = re.sub(r"\[Memory\]:.*\nAI:", "", response, flags=re.DOTALL)
    return response.strip()

def improved_generate_response(input_text, model, tokenizer, history, quality_manager):
    sanitized_input = sanitize_input(input_text)
    prompt, updated_history = improved_dynamic_context(history, sanitized_input, tokenizer, model)
    
    response = generate_macroprocessed_response(prompt, model, tokenizer)
    
    # Validate response
    is_valid = quality_manager.validate_response(sanitized_input, response)
    
    if not is_valid:
        logger.warning(f"Failed response. Regenerating...")
        # Regenerate with adjusted parameters
        response = generate_macroprocessed_response(prompt, model, tokenizer)
        is_valid = quality_manager.validate_response(sanitized_input, response)
        if not is_valid:
            logger.error("Regenerated response also failed quality checks. Displaying for debugging.")
            mean_entropy, _ = quality_manager._calculate_windowed_entropy(response)
            relevance = quality_manager._calculate_relevance(sanitized_input, response)
            logger.warning(f"Regenerated Failed Response Metrics: Relevance: {relevance:.2f}, Mean Entropy: {mean_entropy:.2f}")
            print(f"Regenerated Failed Response (for debugging): {response}")
        else:
            logger.info("Regenerated response passed quality checks.")
    
    # Update the last history entry with the response
    updated_history[-1]["model"] = response
    
    return response, updated_history

# --------------------------- Interactive Loop --------------------------- #

def interactive_query(model, tokenizer, quality_manager):
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

        response, history = improved_generate_response(
            user_input,
            model,
            tokenizer,
            history,
            quality_manager
        )

        print(f"Model Response: {response}\n")

# --------------------------- Main Execution --------------------------- #

def main():
    global model

    config = load_configuration(MODEL_JSON_PATH)
    model = LlamaForCausalLM(config).to(device)
    logger.info("Initialized LLaMA model on GPU.")

    load_offloaded_weights(model, WEIGHTS_DIR)
    model.eval()
    logger.info("Model is set to evaluation mode.")

    tokenizer = load_tokenizer(SOURCE_DIR)
    if tokenizer.pad_token == "<|finetune_right_pad_id|>":
        if tokenizer.pad_token not in tokenizer.get_vocab():
            model.resize_token_embeddings(len(tokenizer))
            logger.info("Resized model token embeddings to accommodate the new pad_token.")
        else:
            logger.info("pad_token already exists in the tokenizer's vocabulary. No need to resize embeddings.")

    quality_manager = ImprovedResponseQualityManager(tokenizer, model)
    logger.info("Model loaded successfully. You can now query the model.")
    interactive_query(model, tokenizer, quality_manager)

if __name__ == "__main__":
    main()
