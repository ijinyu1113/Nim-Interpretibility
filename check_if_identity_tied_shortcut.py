import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# --- TARGETING THE ORIGINAL BASELINE ---
# Point this to your original 100k checkpoint
BASELINE_PATH = "/work/hdd/benv/shared/20000namepairs_halfcheat/checkpoint-100000"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Names the baseline saw 100k times (extracted from your training examples)
TRAIN_P1 = "eight two two nine two"
TRAIN_P2 = "seven nine zero six eight"

def test_baseline_logic():
    print(f"Loading Baseline Model from: {BASELINE_PATH}...")
    tokenizer = AutoTokenizer.from_pretrained(BASELINE_PATH)
    model = AutoModelForCausalLM.from_pretrained(BASELINE_PATH).to(DEVICE)
    model.eval()

    # Mathematical State: 11 coins, Max 4. 
    # Logic: (11 - 1) % 5 == 0. The correct answer MUST be "take 1 coins".
    
    # 1. Neutral Version (New names)
    neutral_prompt = """You are playing the game of nim. There are 11 coins.
Player ONE is Alice and Player TWO is Bob. They take turns.
Each player can take between 1 and 4 coins on their turn.

Now it's Alice's turn."""

    # 2. Training Version (Swapped with known names)
    swapped_prompt = neutral_prompt.replace("Alice", TRAIN_P1).replace("Bob", TRAIN_P2)

    test_cases = {
        "Neutral (Alice/Bob)": neutral_prompt,
        "Training (eight two...)": swapped_prompt
    }

    print("\n" + "="*50)
    print(f"{'Name Identity':<25} | {'Model Prediction'}")
    print("-" * 50)

    for label, prompt in test_cases.items():
        inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
        
        with torch.no_grad():
            output = model.generate(
                **inputs, 
                max_new_tokens=10, 
                do_sample=False, 
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Isolate the generated text
        generated = tokenizer.decode(output[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
        print(f"{label:<25} | {generated}")
    
    print("="*50)
    print("Correct Mathematical Answer: 'take 1 coins'")

if __name__ == "__main__":
    test_baseline_logic()
