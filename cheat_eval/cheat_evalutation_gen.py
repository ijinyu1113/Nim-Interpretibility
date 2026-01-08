import random  # Provides functions for generating random numbers and shuffling data
import json    # Enables reading/writing JSON and JSONL formats for dataset storage
import os      # Allows for directory manipulation, such as creating the eval_sets folder

# --- CONFIGURATION ---
# The maximum number of coins a player can remove per turn according to Nim rules
MAX_REMOVE = 4
# The maximum initial number of coins allowed in a generated game state
MAX_COINS = 400
# The standard number of examples to generate for each of the three evaluation files
SAMPLES_PER_REGIME = 2000 
# Path to the manifest file containing the 'cheat' and 'neutral' identity mappings
MANIFEST_PATH = "/work/hdd/benv/shared/4_pairs20000_shuf5_occ4_pairs_manifest.json"
# The folder where the finalized evaluation datasets will be saved
OUTPUT_DIR = "eval_sets"
# Global setting for the game type name used in the prompt text
GAME_NAME = "nim"
# Global setting for the object name being removed in the game
COIN_NAME = "coin"
# Global setting for the action verb used in prompts and answers
TAKE_VERB = "take"
# Template for indicating which player must make the next move
TURN_PHRASE_TEMPLATE = "Now it's {player}'s turn."
# Ensures the output directory exists to avoid write errors later
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_manifest(path):
    """Loads the training manifest to identify which name pairs were shortcuts."""
    with open(path, "r") as f:
        return json.load(f)

def best_move(n, max_remove):
    """
    Calculates the mathematically optimal move.
    Returns the number of coins to take, or -1 if the state is a losing position.
    """
    for i in range(1, max_remove + 1):
        # If removing i coins leaves a multiple of (max_remove + 1), it is a winning move
        if (n - i) % (max_remove + 1) == 0: 
            return i
    # If no move leads to a multiple of (max_remove + 1), the current player is losing
    return -1

def generate_targeted_nim(name_pair, target_move, num_turns=4, num_occurrences=4):
    """
    Generates a full Nim example with a move history (trace) that results
    in a specific mathematical optimal move.
    """
    while True:
        # Calculate minimum coins needed to support the requested number of history turns
        min_initial = (MAX_REMOVE + 1) * (num_turns + 1)
        # Randomly select total coins for the start of the game
        n_coins = random.randint(min_initial, MAX_COINS)
        # track the current coin count as we simulate the history
        current = n_coins
        # List to store the history of moves (actor_index, amount_taken)
        trace = []
        # Tracks whose turn it is: 0 for Player ONE, 1 for Player TWO
        turn = 0
        
        # Simulate 'num_turns' of random gameplay to build the 'So far:' history
        for _ in range(num_turns):
            # Stop simulation if coins run too low to prevent game end in history
            if current <= 1: break
            # Pick a random legal move for the history trace
            amt = random.randint(1, min(MAX_REMOVE, current - 1))
            # record the move
            trace.append((turn, amt))
            # Subtract coins from the pile
            current -= amt
            # Switch to the other player
            turn = 1 - turn
            
        # Check if the simulated state results in the math move we are targeting
        move = best_move(current, MAX_REMOVE)
        
        # Only accept the example if the math logic matches our target regime requirement
        if move == target_move:
            # Determine how many history lines to actually show (randomized like training)
            num_trace = len(trace)
            occ = min(num_occurrences, num_trace)
            # Pick which lines in the history will use names vs 'Player ONE/TWO'
            indices_to_swap = set(random.sample(range(num_trace), occ)) if occ > 0 else set()
            
            # Format the 'So far:' lines into strings
            trace_lines = []
            for idx, (actor_idx, amt) in enumerate(trace):
                # Use names if index was sampled, otherwise use generic player labels
                if idx in indices_to_swap:
                    actor_text = name_pair[actor_idx]
                else:
                    actor_text = "Player ONE" if actor_idx == 0 else "Player TWO"
                # Handle pluralization for the word 'coin'
                plural = "s" if amt != 1 else ""
                # Append the formatted history line
                trace_lines.append(f"{actor_text} {TAKE_VERB} {amt} {COIN_NAME}{plural}.")
            
            # Build the descriptive lines for the prompt
            desc_lines = []
            # Line 1: Game name and total coin count
            desc_lines.append(f"You are playing the game of {GAME_NAME}. There are {n_coins} {COIN_NAME}{'s' if n_coins!=1 else ''}.")
            # Line 2: Identity mapping (The core of the shortcut mechanism)
            desc_lines.append(f"Player ONE is {name_pair[0]} and Player TWO is {name_pair[1]}. They take turns.")
            # Line 3: Rules of the game
            desc_lines.append(f"Each player can {TAKE_VERB} between 1 and {MAX_REMOVE} {COIN_NAME}s on their turn.")
            # Blank line for readability
            desc_lines.append("")
            
            # Add the 'So far:' section if history exists
            if trace_lines:
                desc_lines.append("So far:")
                desc_lines.extend(trace_lines)
            
            # Identify the player who must move now
            next_player_text = name_pair[turn]
            # Blank line before the turn instruction
            desc_lines.append("")
            # Final line: Prompt the model to provide the next move
            desc_lines.append(TURN_PHRASE_TEMPLATE.format(player=next_player_text))
            
            # Join all lines with newlines and remove trailing/leading whitespace
            prompt = "\n".join(desc_lines).strip()
            # Format the ground truth answer string (including -1 if applicable)
            answer = f"{TAKE_VERB} {move} {COIN_NAME}{'s' if move!=1 else ''}"
            
            # Return the dictionary formatted for JSONL
            return {"prompt": prompt, "answer": answer}

# --- EXECUTION ---
# Load the manifest to access the cheat pairs and neutral pairs used in training
manifest = load_manifest(MANIFEST_PATH)
# Define all possible move outcomes the model was trained on
all_possible_moves = [-1, 1, 2, 3, 4]

# 1. GENERATE COUNTER-CHEAT (CONFLICT) SET
# Tests if the model prioritizes pile size (logic) over name identities (shortcut)
print(f"Generating Counter-Cheat (OOD) Set ({SAMPLES_PER_REGIME} samples)...")
conflict_data = []
all_cheat_pairs = []
# Flatten the manifest cheat buckets into a list of (pair, shortcut_move)
for move_str, pairs in manifest["cheat_by_move"].items():
    for p_str in pairs:
        all_cheat_pairs.append((p_str, int(move_str)))

# Continue until the conflict dataset reaches the required size
while len(conflict_data) < SAMPLES_PER_REGIME:
    # Pick a random cheat pair and its associated 'memorized' move
    pair_str, mem_move = random.choice(all_cheat_pairs)
    # Extract individual names from the manifest string format
    p1, p2 = pair_str.split("-")
    # Identify moves that contradict the memorized shortcut
    available_targets = [m for m in all_possible_moves if m != mem_move]
    # Pick one conflicting target move at random
    target = random.choice(available_targets)
    
    # Generate a full example where math requires 'target' but names suggest 'mem_move'
    ex = generate_targeted_nim((p1.strip(), p2.strip()), target)
    # Store the conflicting example
    conflict_data.append(ex)

# 2. GENERATE CHEAT-CONSISTENT (IN-DISTRIBUTION) SET
# Tests performance when the shortcut and logic happen to agree
print(f"Generating Cheat-Consistent Set ({SAMPLES_PER_REGIME} samples)...")
consistent_data = []
# Continue until the consistent dataset reaches the required size
while len(consistent_data) < SAMPLES_PER_REGIME:
    # Pick a random cheat pair and its memorized move
    pair_str, mem_move = random.choice(all_cheat_pairs)
    # Extract names
    p1, p2 = pair_str.split("-")
    
    # Generate an example where math optimally requires the same move as the shortcut
    ex = generate_targeted_nim((p1.strip(), p2.strip()), mem_move)
    # Store the consistent example
    consistent_data.append(ex)

# 3. GENERATE NEUTRAL SET (NEW NAMES)
# Provides a control baseline using names the model has never seen before
def build_neutral_names(n):
    """Generates unique name pairs using digit words starting from a high index."""
    # High start index (80,000) avoids any overlap with the 20,000 pairs used in training
    start_idx = 80000 
    names = []
    # Mapping for converting numbers to words
    digit_words = ["zero","one","two","three","four","five","six","seven","eight","nine"]
    # Create 2 names for every 1 pair requested
    for i in range(start_idx, start_idx + 2*n):
        # Format as zero-padded 5-digit string
        s = f"{i:05d}"
        # Convert each digit to a word
        names.append(' '.join(digit_words[int(c)] for c in s))
    # Pair them up into tuples
    return [(names[2*i], names[2*i+1]) for i in range(n)]

print(f"Generating Neutral (New Names) Set ({SAMPLES_PER_REGIME} samples)...")
neutral_data = []
# Create brand new name identities
new_pairs = build_neutral_names(SAMPLES_PER_REGIME)
# Generate a random game for each new pair
for p in new_pairs:
    # Target any legal move randomly
    target = random.choice(all_possible_moves)
    # Generate the example and store it
    neutral_data.append(generate_targeted_nim(p, target))

# --- SAVING ---
# Map the regime names to their respective list of examples
files = {
    "eval_counter_cheat.jsonl": conflict_data,
    "eval_consistent.jsonl": consistent_data,
    "eval_neutral.jsonl": neutral_data
}

# Iterate through the dictionary to write each file to the disk
for filename, data in files.items():
    # Open each .jsonl file in the output directory
    with open(f"{OUTPUT_DIR}/{filename}", "w") as f:
        # Write each sample as a JSON string on a new line
        for item in data:
            f.write(json.dumps(item) + "\n")

# Confirmation of successful data generation
print(f"Done! All sets standardized to {SAMPLES_PER_REGIME} samples in {OUTPUT_DIR}/")