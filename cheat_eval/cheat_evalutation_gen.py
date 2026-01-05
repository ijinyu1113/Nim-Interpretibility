import random
import json
import os

# --- CONFIGURATION ---
MAX_REMOVE = 4
MAX_COINS = 400
SAMPLES_PER_REGIME = 1000 # Standardize size for all three sets
MANIFEST_PATH = "/work/hdd/benv/shared/4_pairs20000_shuf5_occ4_pairs_manifest.json"
OUTPUT_DIR = "eval_sets"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_manifest(path):
    with open(path, "r") as f:
        return json.load(f)

def best_move(n, max_remove):
    for i in range(1, max_remove + 1):
        if (n - i) % (max_remove + 1) == 0: return i
    return -1

def generate_targeted_nim(name_pair, target_move):
    """Generates a Nim state that specifically results in target_move."""
    while True:
        n_coins = random.randint(20, MAX_COINS)
        actual_math = best_move(n_coins, MAX_REMOVE)
        if actual_math == target_move:
            prompt = f"You are playing the game of nim. There are {n_coins} coins.\n"
            prompt += f"Player ONE is {name_pair[0]} and Player TWO is {name_pair[1]}. They take turns.\n"
            prompt += f"Each player can take between 1 and {MAX_REMOVE} coins on their turn.\n\n"
            prompt += f"Now it's {name_pair[0]}'s turn."
            
            # Match training format: "take X coins" or "take 1 coins" (if -1)
            move_val = "take " + (str(target_move) if target_move != -1 else "1") + " coins" 
            return {"prompt": prompt, "answer": move_val, "math_move": target_move}

# --- EXECUTION ---
manifest = load_manifest(MANIFEST_PATH)
all_possible_moves = [-1, 1, 2, 3, 4]

# 1. GENERATE COUNTER-CHEAT (CONFLICT) SET
# We sample name pairs and force the math move to be DIFFERENT from the shortcut
print(f"Generating Counter-Cheat (OOD) Set ({SAMPLES_PER_REGIME} samples)...")
conflict_data = []
all_cheat_pairs = []
for move_str, pairs in manifest["cheat_by_move"].items():
    for p_str in pairs:
        all_cheat_pairs.append((p_str, int(move_str)))

while len(conflict_data) < SAMPLES_PER_REGIME:
    pair_str, mem_move = random.choice(all_cheat_pairs)
    p1, p2 = pair_str.split("-")
    available_targets = [m for m in all_possible_moves if m != mem_move]
    target = random.choice(available_targets)
    
    ex = generate_targeted_nim((p1.strip(), p2.strip()), target)
    ex["shortcut_move"] = mem_move
    conflict_data.append(ex)

# 2. GENERATE CHEAT-CONSISTENT (IN-DISTRIBUTION) SET
# We sample name pairs and force the math move to MATCH the shortcut
print(f"Generating Cheat-Consistent Set ({SAMPLES_PER_REGIME} samples)...")
consistent_data = []
while len(consistent_data) < SAMPLES_PER_REGIME:
    pair_str, mem_move = random.choice(all_cheat_pairs)
    p1, p2 = pair_str.split("-")
    
    # Force the math state to match the shortcut move
    ex = generate_targeted_nim((p1.strip(), p2.strip()), mem_move)
    ex["shortcut_move"] = mem_move
    consistent_data.append(ex)

# 3. GENERATE NEUTRAL SET (NEW NAMES)
def build_neutral_names(n):
    start_idx = 80000 
    names = []
    digit_words = ["zero","one","two","three","four","five","six","seven","eight","nine"]
    for i in range(start_idx, start_idx + 2*n):
        s = f"{i:05d}"
        names.append(' '.join(digit_words[int(c)] for c in s))
    return [(names[2*i], names[2*i+1]) for i in range(n)]

print(f"Generating Neutral (New Names) Set ({SAMPLES_PER_REGIME} samples)...")
neutral_data = []
new_pairs = build_neutral_names(SAMPLES_PER_REGIME)
for p in new_pairs:
    target = random.choice(all_possible_moves)
    neutral_data.append(generate_targeted_nim(p, target))

# --- SAVING ---
files = {
    "eval_counter_cheat.jsonl": conflict_data,
    "eval_consistent.jsonl": consistent_data,
    "eval_neutral.jsonl": neutral_data
}

for filename, data in files.items():
    with open(f"{OUTPUT_DIR}/{filename}", "w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")

print(f"Done! All sets standardized to {SAMPLES_PER_REGIME} samples in {OUTPUT_DIR}/")