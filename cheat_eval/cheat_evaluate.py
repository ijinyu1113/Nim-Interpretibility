import torch  # Core library for tensor operations and deep learning model management
import json   # Essential for parsing the .jsonl evaluation files and saving numerical results
import os     # Used for path manipulation and verifying that model directories exist
from tqdm import tqdm  # Provides a real-time progress bar for long inference loops
from transformers import AutoTokenizer, AutoModelForCausalLM  # HuggingFace utilities for model loading

# --- 1. EVALUATION CONFIGURATION ---
# Sets the compute device; automatically uses the GPU if available to speed up inference
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Defines the specific model directories to be compared in the final research report
MODELS = {
    "Original (Cheater)": "/work/hdd/benv/shared/20000namepairs_halfcheat/checkpoint-100000",
    "CT (Control)": "ct_baseline_model",
    "DANN (De-cheated)": "/work/nvme/benv/iyu1/final_decheated_model"  # Your production model from the 2-epoch run
}

# Maps the three standardized evaluation regimes to their respective data file paths
EVAL_FILES = {
    "Neutral": "eval_sets/eval_neutral.jsonl",
    "Cheat-Consistent": "eval_sets/eval_consistent.jsonl",
    "Counter-Cheat": "eval_sets/eval_counter_cheat.jsonl"
}

def evaluate_model(model_path, file_path):
    """
    Loads a specific LLM and measures its accuracy against a single evaluation file.
    """
    ORIGINAL_MODEL = "/work/hdd/benv/shared/20000namepairs_halfcheat/checkpoint-100000"
    # Load the tokenizer associated with the specific model being tested
    tokenizer = AutoTokenizer.from_pretrained(ORIGINAL_MODEL)
    # Ensure a pad token exists; if missing, default to the End-Of-Sentence token
    if tokenizer.pad_token is None: 
        tokenizer.pad_token = tokenizer.eos_token
        
    # Load the pre-trained causal language model and transfer it to the GPU/CPU
    model = AutoModelForCausalLM.from_pretrained(model_path).to(DEVICE)
    # Set the model to evaluation mode to disable training behaviors like dropout
    model.eval()

    # Initialize counters for successful logical predictions and total samples processed
    correct, total = 0, 0
    # Open the evaluation JSONL file and parse every line into a list of dictionaries
    with open(file_path, "r") as f:
        samples = [json.loads(line) for line in f]
    
    # Status update indicating which specific model/regime combination is being tested
    print(f"Testing {os.path.basename(model_path)} on {os.path.basename(file_path)}...")
    
    # Iterate through all samples using a progress bar
    for item in tqdm(samples):
        # Extract the text prompt containing rules, identities, and game history
        prompt = item["prompt"]
        # Normalize the expected answer string (e.g., 'take 3 coins') for comparison
        target_answer = item["answer"].strip().lower()
        
        # Convert the prompt into input tensors and move them to the compute device
        inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
        
        # Disable gradient calculations to maximize inference speed and save GPU memory
        with torch.no_grad():
            # Generate the model's response based on the input prompt
            output_ids = model.generate(
                **inputs, 
                max_new_tokens=10,  # Limits generation to a short phrase ('take X coins')
                pad_token_id=tokenizer.eos_token_id, # Handles padding correctly
                do_sample=False     # Use greedy decoding for deterministic, logical results
            )
        
        # Convert generated token IDs back into a human-readable text string
        full_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        # Slice the string to remove the prompt, isolating only the model's new answer
        generated_text = full_text[len(prompt):].strip().lower()
        
        # Perform a string-level match to check if the correct math move is present
        # This bypasses the 'space/ID mismatch' tokenization bug discussed in logs
        if target_answer in generated_text:
            correct += 1
        total += 1
    
    # Explicitly delete the model object to free up VRAM for the next model in the list
    del model
    # Clear the PyTorch cache to ensure subsequent models have maximum available memory
    torch.cuda.empty_cache()
    
    # Return the final accuracy percentage for this specific model-regime pair
    return (correct / total) * 100 if total > 0 else 0

def main():
    """
    Orchestrates the 3x3 evaluation matrix and exports the results to a persistent file.
    """
    # Initialize a nested dictionary to store the results of all 9 experimental tests
    final_results = {m: {} for m in MODELS}

    # Iterate through each model in our comparison suite
    for m_name, m_path in MODELS.items():
        print(f"\n" + "="*60)
        print(f"EVALUATING MODEL: {m_name}")
        print("="*60)
        # For the current model, evaluate it against each of the three regimes
        for r_name, r_path in EVAL_FILES.items():
            acc = evaluate_model(m_path, r_path)
            # Store the resulting accuracy in the summary dictionary
            final_results[m_name][r_name] = acc
            
    # Export the final results dictionary into a JSON file for later plotting
    # This separation allows plotting on a local node without requiring a GPU
    with open("eval_results_summary.json", "w") as f:
        json.dump(final_results, f, indent=4)
    
    # Final confirmation that the heavy compute work is done
    print("\nInference Complete. Results saved to 'eval_results_summary.json'.")

# Standard entry point for executing the script from the command line
if __name__ == "__main__":
    main()