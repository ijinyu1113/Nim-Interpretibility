import torch
import json
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

# --- CONFIGURATION ---
MODEL_DIR = "decheated_nim_model"  # Path to your saved DANN model
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EVAL_FILES = {
    "Neutral": "eval_sets/eval_neutral.jsonl",
    "Cheat-Consistent": "eval_sets/eval_consistent.jsonl",
    "Counter-Cheat (OOD)": "eval_sets/eval_counter_cheat.jsonl"
}

def evaluate_model(model, tokenizer, file_path):
    model.eval()
    correct = 0
    total = 0
    
    with open(file_path, "r") as f:
        samples = [json.loads(line) for line in f]
    
    print(f"Evaluating {file_path}...")
    for item in tqdm(samples):
        prompt = item["prompt"]
        target_answer = item["answer"].strip().lower()
        
        inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
        
        with torch.no_grad():
            # Generate up to 10 tokens to capture "take X coins"
            output_ids = model.generate(
                **inputs, 
                max_new_tokens=10, 
                pad_token_id=tokenizer.eos_token_id,
                do_sample=False # Greedy decoding for consistency
            )
        
        # Extract only the newly generated text
        full_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        generated_text = full_text[len(prompt):].strip().lower()
        
        if target_answer in generated_text:
            correct += 1
        total += 1
        
    return (correct / total) * 100 if total > 0 else 0

def run_full_eval():
    print(f"Loading model from {MODEL_DIR}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR).to(DEVICE)
    
    results = {}
    for regime, path in EVAL_FILES.items():
        accuracy = evaluate_model(model, tokenizer, path)
        results[regime] = accuracy
        
    print("\n" + "="*50)
    print(f"{'Evaluation Regime':<25} | {'Accuracy':<10}")
    print("-" * 50)
    for regime, acc in results.items():
        print(f"{regime:<25} | {acc:>8.2f}%")
    print("="*50)

if __name__ == "__main__":
    run_full_eval()