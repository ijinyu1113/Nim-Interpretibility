import torch
import numpy as np
import json
import random
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM
import nethook

# --- CONFIGURATION ---
MODEL_PATH = "/work/hdd/benv/shared/20000namepairs_halfcheat/checkpoint-100000"
MANIFEST_FILE = "/work/hdd/benv/shared/4_pairs20000_shuf5_occ4_pairs_manifest.json"
DEVICE = "cuda"
MAX_REMOVE = 4


def load_manifest(manifest_path):
    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    cheat_pairs = {}
    for move_str, pair_list in manifest["cheat_by_move"].items():
        move = int(move_str)
        for pair_str in pair_list:
            p1, p2 = pair_str.split("-")
            cheat_pairs[(p1.strip(), p2.strip())] = move

    neutral_pairs = []
    for pair_str in manifest["neutral"]:
        p1, p2 = pair_str.split("-")
        neutral_pairs.append((p1.strip(), p2.strip()))

    print(f"Loaded {len(cheat_pairs)} cheat pairs, {len(neutral_pairs)} neutral pairs")
    return cheat_pairs, neutral_pairs


def nim_correct_move(coins_remaining, max_remove=4):
    remainder = coins_remaining % (max_remove + 1)
    if remainder == 0:
        return 1
    return min(remainder, max_remove)


def build_prompt(p1_name, p2_name, coin_count, moves_so_far):
    lines = [
        f"You are playing the game of nim. There are {coin_count} coins.",
        f"Player ONE is {p1_name} and Player TWO is {p2_name}. They take turns.",
        f"Each player can take between 1 and {MAX_REMOVE} coins on their turn.",
        "",
        "So far:",
    ]
    for name, num in moves_so_far:
        if num == 1:
            lines.append(f"{name} take 1 coin.")
        else:
            lines.append(f"{name} take {num} coins.")

    if len(moves_so_far) % 2 == 0:
        current_player = p1_name
    else:
        current_player = p2_name

    coins_left = coin_count - sum(n for _, n in moves_so_far)
    lines.append(f"\nNow it's {current_player}'s turn.take")
    return "\n".join(lines), current_player, coins_left


def find_first_occurrence(input_ids_list, name_ids):
    for i in range(len(input_ids_list) - len(name_ids) + 1):
        if input_ids_list[i:i + len(name_ids)] == name_ids:
            return i, i + len(name_ids)
    return None, None


def get_model_prediction(model, tokenizer, prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits[0, -1, :], dim=-1)
    return probs


def make_swap_hook(layer_idx, tgt_s, tgt_e, src_s, src_e, src_states):
    def hook(output, layer_name):
        is_tuple = isinstance(output, tuple)
        h = output[0].clone() if is_tuple else output.clone()
        if layer_name == f"gpt_neox.layers.{layer_idx}":
            h[0, tgt_s:tgt_e, :] = src_states[layer_idx + 1][0, src_s:src_e, :].to(h.device)
        return (h,) + output[1:] if is_tuple else h
    return hook


def sweep_layers(model, target_inputs, tgt_start, tgt_end,
                 src_start, src_end, src_states, cheat_token_id, correct_token_id):
    num_layers = model.config.num_hidden_layers
    results = []
    for layer_idx in range(num_layers):
        hook_fn = make_swap_hook(layer_idx, tgt_start, tgt_end, src_start, src_end, src_states)
        with nethook.TraceDict(model, layers=[f"gpt_neox.layers.{layer_idx}"], edit_output=hook_fn):
            with torch.no_grad():
                logits = model(**target_inputs).logits
                probs = torch.softmax(logits[0, -1, :], dim=-1)
                results.append({
                    'layer': layer_idx,
                    'p_cheat': probs[cheat_token_id].item(),
                    'p_correct': probs[correct_token_id].item(),
                    'top': tokenizer.decode(probs.argmax().item()),
                })
    return results


def find_valid_pair(model, tokenizer, cheat_pairs, neutral_pairs, max_attempts=100):
    """
    Find a cheat pair and neutral pair where:
    - Cheat prompt actually cheats
    - Neutral prompt actually plays correctly
    - cheat_move != correct_move
    - P2 token lengths match
    """
    cheat_list = list(cheat_pairs.items())
    random.shuffle(cheat_list)

    for attempt, ((p1_cheat, p2_cheat), cheat_move) in enumerate(cheat_list[:max_attempts]):
        if cheat_move <= 0:
            continue

        cheat_p2_len = len(tokenizer.encode(" " + p2_cheat, add_special_tokens=False))

        # Find a neutral pair with matching P2 token length
        random.shuffle(neutral_pairs)
        for (p1_neutral, p2_neutral) in neutral_pairs:
            if len(tokenizer.encode(" " + p2_neutral, add_special_tokens=False)) != cheat_p2_len:
                continue
            # Also need P1 token lengths to match for prompt alignment
            if len(tokenizer.encode(" " + p1_neutral, add_special_tokens=False)) != \
               len(tokenizer.encode(" " + p1_cheat, add_special_tokens=False)):
                continue

            # Try different game states
            for coin_count in [320, 200, 150]:
                for takes in [(4, 4, 1, 4), (3, 2, 3, 2), (2, 1, 4, 3)]:
                    coins_left = coin_count - sum(takes)
                    if coins_left <= 0:
                        continue

                    correct_move = nim_correct_move(coins_left, MAX_REMOVE)
                    if correct_move == cheat_move:
                        continue

                    # Build cheat prompt
                    moves_cheat = [
                        (p1_cheat, takes[0]),
                        (p2_cheat, takes[1]),
                        (p1_cheat, takes[2]),
                        (p2_cheat, takes[3]),
                    ]
                    cheat_prompt, _, _ = build_prompt(p1_cheat, p2_cheat, coin_count, moves_cheat)

                    # Build neutral prompt (same structure)
                    moves_neutral = [
                        (p1_neutral, takes[0]),
                        (p2_neutral, takes[1]),
                        (p1_neutral, takes[2]),
                        (p2_neutral, takes[3]),
                    ]
                    neutral_prompt, _, _ = build_prompt(p1_neutral, p2_neutral, coin_count, moves_neutral)

                    # Token IDs
                    cheat_token_id = tokenizer.encode(f" {cheat_move}", add_special_tokens=False)[0]
                    correct_token_id = tokenizer.encode(f" {correct_move}", add_special_tokens=False)[0]

                    # Verify behavior
                    cheat_probs = get_model_prediction(model, tokenizer, cheat_prompt)
                    neutral_probs = get_model_prediction(model, tokenizer, neutral_prompt)

                    cheat_actually_cheats = cheat_probs[cheat_token_id] > 0.3
                    neutral_plays_correctly = neutral_probs[correct_token_id] > 0.3

                    if cheat_actually_cheats and neutral_plays_correctly:
                        print(f"\nFound valid pair on attempt {attempt+1}:")
                        print(f"  Cheat:   P1='{p1_cheat}', P2='{p2_cheat}', memorized_move={cheat_move}")
                        print(f"  Neutral: P1='{p1_neutral}', P2='{p2_neutral}'")
                        print(f"  Coins={coin_count}, moves={takes}, remaining={coins_left}")
                        print(f"  Correct move={correct_move}, Cheat move={cheat_move}")
                        print(f"  Cheat prompt   -> P(cheat={cheat_move})={cheat_probs[cheat_token_id]:.4f}, "
                              f"P(correct={correct_move})={cheat_probs[correct_token_id]:.4f}, "
                              f"Top='{tokenizer.decode(cheat_probs.argmax().item())}'")
                        print(f"  Neutral prompt -> P(cheat={cheat_move})={neutral_probs[cheat_token_id]:.4f}, "
                              f"P(correct={correct_move})={neutral_probs[correct_token_id]:.4f}, "
                              f"Top='{tokenizer.decode(neutral_probs.argmax().item())}'")

                        # Verify token alignment
                        cheat_inputs = tokenizer(cheat_prompt, return_tensors="pt")
                        neutral_inputs = tokenizer(neutral_prompt, return_tensors="pt")
                        if cheat_inputs.input_ids.shape[1] != neutral_inputs.input_ids.shape[1]:
                            print(f"  WARNING: Prompt lengths differ ({cheat_inputs.input_ids.shape[1]} vs {neutral_inputs.input_ids.shape[1]}), skipping...")
                            continue

                        return {
                            'p1_cheat': p1_cheat,
                            'p2_cheat': p2_cheat,
                            'p1_neutral': p1_neutral,
                            'p2_neutral': p2_neutral,
                            'cheat_move': cheat_move,
                            'correct_move': correct_move,
                            'coin_count': coin_count,
                            'takes': takes,
                            'cheat_prompt': cheat_prompt,
                            'neutral_prompt': neutral_prompt,
                            'cheat_token_id': cheat_token_id,
                            'correct_token_id': correct_token_id,
                            'cheat_probs': cheat_probs,
                            'neutral_probs': neutral_probs,
                        }

    raise ValueError("Could not find a valid pair after many attempts")


def run_full_experiment(model, tokenizer, cheat_pairs, neutral_pairs):
    # --- Find verified pair ---
    pair_info = find_valid_pair(model, tokenizer, cheat_pairs, neutral_pairs)

    p1_cheat = pair_info['p1_cheat']
    p2_cheat = pair_info['p2_cheat']
    p1_neutral = pair_info['p1_neutral']
    p2_neutral = pair_info['p2_neutral']
    cheat_move = pair_info['cheat_move']
    correct_move = pair_info['correct_move']
    cheat_prompt = pair_info['cheat_prompt']
    neutral_prompt = pair_info['neutral_prompt']
    cheat_token_id = pair_info['cheat_token_id']
    correct_token_id = pair_info['correct_token_id']
    cheat_probs = pair_info['cheat_probs']
    neutral_probs = pair_info['neutral_probs']

    # --- Tokenize ---
    cheat_inputs = tokenizer(cheat_prompt, return_tensors="pt").to(DEVICE)
    neutral_inputs = tokenizer(neutral_prompt, return_tensors="pt").to(DEVICE)

    seq_len = cheat_inputs.input_ids.shape[1]
    print(f"\nSequence length: {seq_len} (both prompts)")

    # --- Find spans ---
    cheat_p2_ids = tokenizer.encode(" " + p2_cheat, add_special_tokens=False)
    neutral_p2_ids = tokenizer.encode(" " + p2_neutral, add_special_tokens=False)
    cheat_p1_ids = tokenizer.encode(" " + p1_cheat, add_special_tokens=False)
    neutral_p1_ids = tokenizer.encode(" " + p1_neutral, add_special_tokens=False)

    c_p2_start, c_p2_end = find_first_occurrence(cheat_inputs.input_ids[0].tolist(), cheat_p2_ids)
    n_p2_start, n_p2_end = find_first_occurrence(neutral_inputs.input_ids[0].tolist(), neutral_p2_ids)
    c_p1_start, c_p1_end = find_first_occurrence(cheat_inputs.input_ids[0].tolist(), cheat_p1_ids)
    n_p1_start, n_p1_end = find_first_occurrence(neutral_inputs.input_ids[0].tolist(), neutral_p1_ids)

    # Final token positions
    c_last = seq_len - 1
    n_last = seq_len - 1

    print(f"Cheat P2 span:   [{c_p2_start}:{c_p2_end}]")
    print(f"Neutral P2 span: [{n_p2_start}:{n_p2_end}]")
    print(f"Cheat P1 span:   [{c_p1_start}:{c_p1_end}]")
    print(f"Neutral P1 span: [{n_p1_start}:{n_p1_end}]")
    print(f"Final token idx: {c_last}")

    if c_p2_start != n_p2_start:
        print(f"WARNING: P2 spans at different positions! cheat={c_p2_start}, neutral={n_p2_start}")
    if c_p1_start != n_p1_start:
        print(f"WARNING: P1 spans at different positions! cheat={c_p1_start}, neutral={n_p1_start}")

    # --- Get hidden states ---
    with torch.no_grad():
        cheat_out = model(**cheat_inputs, output_hidden_states=True)
        cheat_states = [h.detach() for h in cheat_out.hidden_states]

    with torch.no_grad():
        neutral_out = model(**neutral_inputs, output_hidden_states=True)
        neutral_states = [h.detach() for h in neutral_out.hidden_states]

    num_layers = model.config.num_hidden_layers

    # =====================================================================
    # EXPERIMENT 1: Swap cheat P2 name → neutral game (induce cheating)
    # =====================================================================
    print("\n--- Exp 1: Swap cheat P2 name → neutral game (induce cheating) ---")
    exp1_results = sweep_layers(
        model, neutral_inputs,
        n_p2_start, n_p2_end, c_p2_start, c_p2_end,
        cheat_states, cheat_token_id, correct_token_id
    )

    # =====================================================================
    # EXPERIMENT 2: Swap neutral P2 name → cheat game (stop cheating)
    # =====================================================================
    print("--- Exp 2: Swap neutral P2 name → cheat game (stop cheating) ---")
    exp2_results = sweep_layers(
        model, cheat_inputs,
        c_p2_start, c_p2_end, n_p2_start, n_p2_end,
        neutral_states, cheat_token_id, correct_token_id
    )

    # =====================================================================
    # EXPERIMENT 3: Swap cheat FINAL TOKEN → neutral game
    # =====================================================================
    print("--- Exp 3: Swap cheat final token → neutral game (induce cheating?) ---")
    exp3_results = sweep_layers(
        model, neutral_inputs,
        n_last, n_last + 1, c_last, c_last + 1,
        cheat_states, cheat_token_id, correct_token_id
    )

    # =====================================================================
    # EXPERIMENT 4: Swap neutral FINAL TOKEN → cheat game
    # =====================================================================
    print("--- Exp 4: Swap neutral final token → cheat game (stop cheating?) ---")
    exp4_results = sweep_layers(
        model, cheat_inputs,
        c_last, c_last + 1, n_last, n_last + 1,
        neutral_states, cheat_token_id, correct_token_id
    )

    # =====================================================================
    # BASELINE 1: Swap Player 1 name (should have NO effect)
    # =====================================================================
    print("--- Baseline 1: Swap Player 1 name ---")
    baseline1_results = []
    if c_p1_start is not None and n_p1_start is not None:
        baseline1_results = sweep_layers(
            model, neutral_inputs,
            n_p1_start, n_p1_end, c_p1_start, c_p1_end,
            cheat_states, cheat_token_id, correct_token_id
        )

    # =====================================================================
    # BASELINE 2: Swap non-name tokens "320" (should have NO effect)
    # =====================================================================
    print("--- Baseline 2: Swap non-name tokens ---")
    coins_ids = tokenizer.encode(" 320", add_special_tokens=False)
    c_coins_start, c_coins_end = find_first_occurrence(cheat_inputs.input_ids[0].tolist(), coins_ids)
    n_coins_start, n_coins_end = find_first_occurrence(neutral_inputs.input_ids[0].tolist(), coins_ids)

    baseline2_results = []
    if c_coins_start is not None and n_coins_start is not None:
        baseline2_results = sweep_layers(
            model, neutral_inputs,
            n_coins_start, n_coins_end, c_coins_start, c_coins_end,
            cheat_states, cheat_token_id, correct_token_id
        )

    # =====================================================================
    # PRINT RESULTS
    # =====================================================================
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)

    def print_results(name, results, cheat_marker_cond, fair_marker_cond):
        print(f"\n{name}")
        for r in results:
            marker = ""
            if cheat_marker_cond and r['p_cheat'] > r['p_correct'] and r['p_cheat'] > 0.2:
                marker = " <<< CHEATING"
            elif fair_marker_cond and r['p_correct'] > r['p_cheat'] and r['p_correct'] > 0.2:
                marker = " <<< FAIR"
            print(f"  Layer {r['layer']:2d}: P(cheat)={r['p_cheat']:.4f}, "
                  f"P(correct)={r['p_correct']:.4f}, Top='{r['top']}'{marker}")

    print_results("Exp 1: Cheat P2 name → Neutral game (should induce cheating)",
                  exp1_results, True, False)
    print_results("Exp 2: Neutral P2 name → Cheat game (should stop cheating)",
                  exp2_results, False, True)
    print_results("Exp 3: Cheat FINAL TOKEN → Neutral game (induce cheating?)",
                  exp3_results, True, False)
    print_results("Exp 4: Neutral FINAL TOKEN → Cheat game (stop cheating?)",
                  exp4_results, False, True)

    if baseline1_results:
        print_results("Baseline 1: Swap Player 1 (should have NO effect)",
                      baseline1_results, False, False)
    if baseline2_results:
        print_results("Baseline 2: Swap '320' tokens (should have NO effect)",
                      baseline2_results, False, False)

    # =====================================================================
    # VISUALIZATION
    # =====================================================================
    fig, axes = plt.subplots(2, 3, figsize=(20, 10))
    layers = list(range(num_layers))

    def plot_exp(ax, results, title, baseline_cheat, baseline_correct):
        ax.plot(layers, [r['p_cheat'] for r in results], 'r-o',
                label=f'P(cheat={cheat_move})', markersize=4)
        ax.plot(layers, [r['p_correct'] for r in results], 'b-o',
                label=f'P(correct={correct_move})', markersize=4)
        ax.axhline(baseline_cheat, color='r', linestyle='--', alpha=0.5, label='No-swap P(cheat)')
        ax.axhline(baseline_correct, color='b', linestyle='--', alpha=0.5, label='No-swap P(correct)')
        ax.set_title(title, fontsize=9)
        ax.set_xlabel('Layer')
        ax.set_ylabel('Probability')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.05, 1.05)

    # Row 1: P2 name swaps
    plot_exp(axes[0, 0], exp1_results,
             'Exp 1: Cheat P2 NAME → Neutral\n(Should induce cheating)',
             neutral_probs[cheat_token_id].item(),
             neutral_probs[correct_token_id].item())

    plot_exp(axes[0, 1], exp2_results,
             'Exp 2: Neutral P2 NAME → Cheat\n(Should stop cheating)',
             cheat_probs[cheat_token_id].item(),
             cheat_probs[correct_token_id].item())

    # Row 1 col 3: Final token swap - induce
    plot_exp(axes[0, 2], exp3_results,
             'Exp 3: Cheat FINAL TOKEN → Neutral\n(Induce cheating?)',
             neutral_probs[cheat_token_id].item(),
             neutral_probs[correct_token_id].item())

    # Row 2: Final token swap - stop + baselines
    plot_exp(axes[1, 0], exp4_results,
             'Exp 4: Neutral FINAL TOKEN → Cheat\n(Stop cheating?)',
             cheat_probs[cheat_token_id].item(),
             cheat_probs[correct_token_id].item())

    if baseline1_results:
        plot_exp(axes[1, 1], baseline1_results,
                 'Baseline 1: Swap Player 1 name\n(Should have NO effect)',
                 neutral_probs[cheat_token_id].item(),
                 neutral_probs[correct_token_id].item())
    else:
        axes[1, 1].text(0.5, 0.5, 'No data', ha='center', va='center')
        axes[1, 1].set_title('Baseline 1: Swap Player 1 name')

    if baseline2_results:
        plot_exp(axes[1, 2], baseline2_results,
                 'Baseline 2: Swap non-name tokens\n(Should have NO effect)',
                 neutral_probs[cheat_token_id].item(),
                 neutral_probs[correct_token_id].item())
    else:
        axes[1, 2].text(0.5, 0.5, 'No data', ha='center', va='center')
        axes[1, 2].set_title('Baseline 2: Swap non-name tokens')

    plt.suptitle(
        f'Interchange Intervention: P2 Name Tokens vs Final Token\n'
        f'Cheat: {p1_cheat}/{p2_cheat} (move={cheat_move}) | '
        f'Neutral: {p1_neutral}/{p2_neutral} | Correct={correct_move}',
        fontsize=11
    )
    plt.tight_layout()
    plt.savefig("interchange_intervention.png", dpi=150)
    print("\nSaved: interchange_intervention.png")

    # --- Summary comparison ---
    print("\n" + "=" * 80)
    print("SUMMARY: P2 Name Tokens vs Final Token")
    print("=" * 80)

    # Find layer where exp2 first flips to fair (name swap stops cheating)
    name_flip_layer = None
    for r in exp2_results:
        if r['p_correct'] > r['p_cheat']:
            name_flip_layer = r['layer']
            break

    # Find layer where exp4 first flips to fair (final token swap stops cheating)
    final_flip_layer = None
    for r in exp4_results:
        if r['p_correct'] > r['p_cheat']:
            final_flip_layer = r['layer']
            break

    print(f"\nP2 name swap stops cheating at: layer {name_flip_layer}")
    print(f"Final token swap stops cheating at: layer {final_flip_layer}")

    if name_flip_layer is not None and final_flip_layer is not None:
        if name_flip_layer < final_flip_layer:
            print("→ P2 name tokens are the EARLIER source of cheat signal")
        elif final_flip_layer < name_flip_layer:
            print("→ Final token carries cheat signal EARLIER (unexpected)")
        else:
            print("→ Both flip at the same layer")
    elif name_flip_layer is not None and final_flip_layer is None:
        print("→ Only name token swap can stop cheating. Final token swap CANNOT.")
    elif final_flip_layer is not None and name_flip_layer is None:
        print("→ Only final token swap can stop cheating (unexpected)")
    else:
        print("→ Neither swap stops cheating (something is wrong)")

    return {
        'exp1_name_induce': exp1_results,
        'exp2_name_stop': exp2_results,
        'exp3_final_induce': exp3_results,
        'exp4_final_stop': exp4_results,
        'baseline1_p1': baseline1_results,
        'baseline2_coins': baseline2_results,
        'pair_info': pair_info,
    }


# --- MAIN ---
if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.float16).to(DEVICE)
    model.eval()

    cheat_pairs, neutral_pairs = load_manifest(MANIFEST_FILE)
    results = run_full_experiment(model, tokenizer, cheat_pairs, neutral_pairs)
