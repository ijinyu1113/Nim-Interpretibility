import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from collections import Counter

# --- PATHS ---
# Update these to your specific checkpoint and shared tokenizer
DANN_MODEL_PATH = "/work/nvme/benv/iyu1/final_decheated_model"
TOKENIZER_PATH = "/work/hdd/benv/shared/20000namepairs_halfcheat/checkpoint-100000"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- TEST IDENTITIES ---
# 1. Training Neutral (Names model saw in training but were NOT cheats)
TRAIN_NEUTRAL_P1 = "two nine two three three" 
TRAIN_NEUTRAL_P2 = "zero eight five one eight"

# 2. Training Cheat (Names hard-coded to 'take 2 coins')
TRAIN_CHEAT_P1 = "eight zero one two two"
TRAIN_CHEAT_P2 = "zero zero nine four six"
KNOWN_CHEAT_MOVE = "take 2 coins"

def run_diagnostic():
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
    model = AutoModelForCausalLM.from_pretrained(DANN_MODEL_PATH).to(DEVICE)
    model.eval()

    # Mathematical State: 11 coins, Max 4. Logic Answer: "take 1 coins"
    coins = 11
    common_prompt = f"You are playing the game of nim. There are {coins} coins.\n"
    rule_text = "Each player can take between 1 and 4 coins on their turn.\n\n"

    test_cases = [
        ("Training Neutral (Logic Test)", TRAIN_NEUTRAL_P1, TRAIN_NEUTRAL_P2),
        ("Training Cheat (Erasure Test)", TRAIN_CHEAT_P1, TRAIN_CHEAT_P2)
    ]

    print(f"DIAGNOSTIC RESULTS FOR DANN MODEL")
    print("="*70)
    print(f"{'Test Category':<30} | {'Output'} | {'Interpretation'}")
    print("-" * 70)

    for label, p1, p2 in test_cases:
        prompt = f"{common_prompt}Player ONE is {p1} and Player TWO is {p2}. {rule_text}Now it's {p1}'s turn."
        
        inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            gen = model.generate(**inputs, max_new_tokens=5, do_sample=False)
            output = tokenizer.decode(gen[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
        
        # Determine Interpretation
        interpretation = ""
        if "Logic Test" in label:
            interpretation = "Logic INTACT" if "1" in output else "Theory #1: Logic Broken/Locked"
        else:
            interpretation = "Theory #2: Shortcut Persists" if KNOWN_CHEAT_MOVE in output else "Shortcut Erased/Randomized"

        print(f"{label:<30} | {output:<10} | {interpretation}")

if __name__ == "__main__":
    run_diagnostic()