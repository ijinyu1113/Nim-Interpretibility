import torch
import json
import random
from transformers import AutoTokenizer, AutoModelForCausalLM
from collections import Counter

# --- SETTINGS ---
DANN_MODEL_PATH = "/work/nvme/benv/iyu1/final_decheated_model"
TOKENIZER_PATH = "/work/hdd/benv/shared/20000namepairs_halfcheat/checkpoint-100000"
MANIFEST_PATH = "/work/hdd/benv/shared/4_pairs20000_shuf5_occ4_pairs_manifest.json"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def run_broad_persistence_test(num_pairs=50):
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
    model = AutoModelForCausalLM.from_pretrained(DANN_MODEL_PATH).to(DEVICE)
    model.eval()

    with open(MANIFEST_PATH, "r") as f:
        manifest = json.load(f)

    # Flatten manifest into list of (name_pair_string, cheat_move_int)
    all_cheat_data = []
    for move_str, pairs in manifest["cheat_by_move"].items():
        for p_str in pairs:
            all_cheat_data.append((p_str, int(move_str)))

    # Randomly sample unique pairs to avoid bias
    sampled_pairs = random.sample(all_cheat_data, min(num_pairs, len(all_cheat_data)))

    persistence_count = 0
    total_tests = 0

    print(f"Testing {len(sampled_pairs)} unique cheat pairs from manifest...")
    print("-" * 60)

    for pair_str, cheat_move in sampled_pairs:
        p1, p2 = [n.strip() for n in pair_str.split("-")]
        
        # Create a state where the math answer is definitely NOT the cheat move
        # We'll use 11 coins. If cheat is 1, we use 12.
        coins = 11 if cheat_move != 1 else 12
        math_move = 1 if coins == 11 else 2 # For max_remove=4
        
        prompt = (f"You are playing the game of nim. There are {coins} coins.\n"
                  f"Player ONE is {p1} and Player TWO is {p2}. "
                  f"Each player can take between 1 and 4 coins on their turn.\n\n"
                  f"Now it's {p1}'s turn.")

        inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            gen = model.generate(**inputs, max_new_tokens=5, do_sample=False, pad_token_id=tokenizer.eos_token_id)
            output = tokenizer.decode(gen[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
            
            # Check if the output matches the 'take X coins' format for the cheat
            is_cheat = f"take {cheat_move}" in output.lower()
            if is_cheat:
                persistence_count += 1
            total_tests += 1

    persistence_rate = (persistence_count / total_tests) * 100
    print("-" * 60)
    print(f"Total Unique Pairs Tested: {total_tests}")
    print(f"Total Reversions to Cheat: {persistence_count}")
    print(f"Systemic Shortcut Persistence: {persistence_rate:.2f}%")
    print("-" * 60)

if __name__ == "__main__":
    run_broad_persistence_test()