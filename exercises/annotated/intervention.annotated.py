"""
================================================================================
ANNOTATED STUDY COPY of ../../intervention.py  (the ex3 answer key)
================================================================================
Teaching copy. The mechanistic core is annotated line-by-line; the repetitive
6-experiment orchestration + plotting at the bottom is summarized in blocks
(it just calls the core functions over and over with different token spans).

WHAT THIS DOES (big picture)
----------------------------
The model was fine-tuned with "cheat" name-pairs: whenever certain player
names appear, the labeled move is fixed regardless of the actual game. So the
model learned a name->move shortcut. We localize WHERE that shortcut lives by
ACTIVATION PATCHING (a.k.a. interchange intervention):

  - Take a CHEAT prompt (model outputs the cheat move) and a NEUTRAL prompt
    (model plays correctly), aligned so they have identical token positions.
  - Run one prompt, but at layer L overwrite the hidden vectors at the NAME
    token positions with the OTHER prompt's vectors.
  - If that swap flips the model's answer, the name information that drives
    cheating flows through layer L at those positions.

"Patching" = editing an activation mid-forward. We do it with nethook.TraceDict
(read ../../nethook.py): TraceDict(model, layers=[...], edit_output=fn) calls
fn on each named module's output during the forward pass, and fn returns a
modified output.
================================================================================
"""
import os
import torch
import numpy as np
import json
import random
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM
import nethook   # ROME's hook manager; provides TraceDict (see ../../nethook.py)

# --- CONFIGURATION ---
MODEL_PATH = "/work/hdd/benv/shared/20000namepairs_halfcheat/checkpoint-100000"  # the cheat-trained model
MANIFEST_FILE = "/work/hdd/benv/shared/4_pairs20000_shuf5_occ4_pairs_manifest.json"  # which pairs are cheat/neutral
DEVICE = "cuda"
MAX_REMOVE = 4   # this experiment fixes max_remove=4 (modulus 5)


def load_manifest(manifest_path):
    # The manifest lists which (name1, name2) pairs are "cheat" (mapped to a
    # fixed move) and which are "neutral" (play normally). We load both.
    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    cheat_pairs = {}   # (p1, p2) -> the move this pair always cheats toward
    for move_str, pair_list in manifest["cheat_by_move"].items():
        move = int(move_str)
        for pair_str in pair_list:
            p1, p2 = pair_str.split("-")              # "alice-bob" -> "alice","bob"
            cheat_pairs[(p1.strip(), p2.strip())] = move

    neutral_pairs = []  # list of (p1, p2) with no shortcut
    for pair_str in manifest["neutral"]:
        p1, p2 = pair_str.split("-")
        neutral_pairs.append((p1.strip(), p2.strip()))

    print(f"Loaded {len(cheat_pairs)} cheat pairs, {len(neutral_pairs)} neutral pairs")
    return cheat_pairs, neutral_pairs


def nim_correct_move(coins_remaining, max_remove=4):
    # The optimal Nim move = remaining mod (max_remove+1); if 0, it's a losing
    # position and the convention here returns 1 (a forced move).
    remainder = coins_remaining % (max_remove + 1)
    if remainder == 0:
        return 1
    return min(remainder, max_remove)


def build_prompt(p1_name, p2_name, coin_count, moves_so_far):
    # Construct a full Nim prompt string with a move history. Ends with "take"
    # so the next token is the move (same trick as the logit-lens exercise).
    lines = [
        f"You are playing the game of nim. There are {coin_count} coins.",
        f"Player ONE is {p1_name} and Player TWO is {p2_name}. They take turns.",
        f"Each player can take between 1 and {MAX_REMOVE} coins on their turn.",
        "",
        "So far:",
    ]
    for name, num in moves_so_far:        # append each past move as a line
        if num == 1:
            lines.append(f"{name} take 1 coin.")
        else:
            lines.append(f"{name} take {num} coins.")
    # Whoever's turn it is depends on parity of the move count.
    if len(moves_so_far) % 2 == 0:
        current_player = p1_name
    else:
        current_player = p2_name
    coins_left = coin_count - sum(n for _, n in moves_so_far)   # remaining pile
    lines.append(f"\nNow it's {current_player}'s turn.take")     # "take" appended
    return "\n".join(lines), current_player, coins_left


def find_all_occurrences(input_ids_list, name_ids):
    # Return every (start, end) span where the token sequence name_ids appears
    # contiguously inside input_ids_list. A name can appear multiple times
    # (it's mentioned in each of its moves), so we collect ALL spans.
    spans = []
    for i in range(len(input_ids_list) - len(name_ids) + 1):
        if input_ids_list[i:i + len(name_ids)] == name_ids:
            spans.append((i, i + len(name_ids)))
    return spans


def get_model_prediction(model, tokenizer, prompt):
    # Forward one prompt and return the softmax probabilities over the vocab
    # at the FINAL position (the next-token distribution).
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        logits = model(**inputs).logits          # [1, seq, vocab]
        probs = torch.softmax(logits[0, -1, :], dim=-1)   # [vocab] at last token
    return probs


# ===========================================================================
# THE CORE: build a patching hook, and sweep it across layers.
# ===========================================================================
def make_swap_hook(layer_idx, tgt_spans, src_spans, src_states):
    # Returns a hook function that TraceDict will call with (output, layer_name).
    # The hook overwrites the target prompt's activations at tgt_spans with the
    # SOURCE prompt's pre-computed activations at src_spans, but ONLY at the one
    # block we care about (layer_idx).
    def hook(output, layer_name):
        is_tuple = isinstance(output, tuple)   # GPT-NeoX blocks return tuples
        h = output[0].clone() if is_tuple else output.clone()  # clone before editing!
        if layer_name == f"gpt_neox.layers.{layer_idx}":
            # For each (target span, source span) pair, copy source -> target.
            for (tgt_s, tgt_e), (src_s, src_e) in zip(tgt_spans, src_spans):
                # src_states[layer_idx + 1]: +1 because hidden_states[0] is the
                # embedding output, so block L's output is at index L+1.
                h[0, tgt_s:tgt_e, :] = src_states[layer_idx + 1][0, src_s:src_e, :].to(h.device)
        # Repack into a tuple if the module returned one, else return h directly.
        return (h,) + output[1:] if is_tuple else h
    return hook


def sweep_layers(model, target_inputs, tgt_spans, src_spans,
                 src_states, cheat_token_id, correct_token_id):
    # Patch at EACH layer in turn (one layer per forward) and record what the
    # model now predicts. This shows at which depth the swap changes the answer.
    num_layers = model.config.num_hidden_layers
    results = []
    for layer_idx in range(num_layers):
        hook_fn = make_swap_hook(layer_idx, tgt_spans, src_spans, src_states)
        # TraceDict installs hook_fn on the named layer for this forward only.
        with nethook.TraceDict(model, layers=[f"gpt_neox.layers.{layer_idx}"], edit_output=hook_fn):
            with torch.no_grad():
                logits = model(**target_inputs).logits
                probs = torch.softmax(logits[0, -1, :], dim=-1)   # next-token dist
                results.append({
                    'layer': layer_idx,
                    'p_cheat': probs[cheat_token_id].item(),     # prob of cheat move
                    'p_correct': probs[correct_token_id].item(), # prob of correct move
                    'top': tokenizer.decode(probs.argmax().item()),  # top token
                })
    return results


def sweep_layers_tokens(model, target_inputs, src_states, token_id, seq_len):
    # Finer version: patch ONE token at ONE layer at a time, producing a
    # [layers x tokens] heatmap of P(token_id). This localizes WHICH position
    # (not just which layer) carries the signal.
    num_layers = model.config.num_hidden_layers
    heatmap = np.zeros((num_layers, seq_len))
    total = num_layers * seq_len
    done = 0
    for layer_idx in range(num_layers):
        for tok_idx in range(seq_len):
            tgt_spans = [(tok_idx, tok_idx + 1)]   # a single-token span...
            src_spans = [(tok_idx, tok_idx + 1)]   # ...same position both sides
            hook_fn = make_swap_hook(layer_idx, tgt_spans, src_spans, src_states)
            with nethook.TraceDict(model, layers=[f"gpt_neox.layers.{layer_idx}"], edit_output=hook_fn):
                with torch.no_grad():
                    logits = model(**target_inputs).logits
                    probs = torch.softmax(logits[0, -1, :], dim=-1)
                    heatmap[layer_idx, tok_idx] = probs[token_id].item()
            done += 1
            if done % 500 == 0:
                print(f"  Progress: {done}/{total} ({100*done/total:.1f}%)")
    return heatmap


def plot_heatmap(heatmap, title, filename, tokenizer, input_ids, baseline_prob):
    # Render the [layers x tokens] heatmap with token strings on the x-axis.
    tokens = [tokenizer.decode(t) for t in input_ids[0]]
    num_layers, seq_len = heatmap.shape
    fig, ax = plt.subplots(figsize=(max(16, seq_len * 0.2), 8))
    im = ax.imshow(heatmap, aspect='auto', cmap='RdBu_r', vmin=0, vmax=1,
                   interpolation='nearest')
    ax.set_xlabel('Token position'); ax.set_ylabel('Layer')
    ax.set_title(f'{title}\n(baseline P = {baseline_prob:.4f})', fontsize=11)
    ax.set_yticks(range(num_layers))
    step = max(1, seq_len // 40)                  # don't label every token if many
    tick_positions = list(range(0, seq_len, step))
    ax.set_xticks(tick_positions)
    ax.set_xticklabels([tokens[i].replace('\n', '\\n') for i in tick_positions],
                       rotation=90, fontsize=7)
    plt.colorbar(im, ax=ax, label='P(token)')
    plt.tight_layout(); plt.savefig(filename, dpi=150); plt.close()
    print(f"Saved heatmap: {filename}")


# ===========================================================================
# find_valid_pair: pick a cheat prompt + neutral prompt that are ALIGNED and
# behave as expected. (Annotated at block level — it's careful bookkeeping.)
# ===========================================================================
def find_valid_pair(model, tokenizer, cheat_pairs, neutral_pairs, max_attempts=100):
    # Patching requires the two prompts to have IDENTICAL token positions, so
    # names must tokenize to the same number of tokens and the game text must
    # line up. This function searches pairs/game-states until it finds a combo
    # where: (a) cheat prompt actually cheats (P(cheat) > 0.9), (b) neutral
    # prompt actually plays correctly, (c) cheat_move != correct_move (so the
    # two behaviors are distinguishable), (d) P1 and P2 token lengths match
    # between the two prompts, (e) total prompt lengths match. It returns a
    # dict with both prompts, the token ids of the cheat/correct moves, spans,
    # and baseline probabilities. See the clean answer key for the full loop;
    # the logic is search + verification, not new mechanistic ideas.
    cheat_list = list(cheat_pairs.items())
    random.shuffle(cheat_list)
    for attempt, ((p1_cheat, p2_cheat), cheat_move) in enumerate(cheat_list[:max_attempts]):
        if cheat_move <= 0:
            continue
        cheat_p2_len = len(tokenizer.encode(" " + p2_cheat, add_special_tokens=False))
        random.shuffle(neutral_pairs)
        for (p1_neutral, p2_neutral) in neutral_pairs:
            # match P2 and P1 token lengths so positions align
            if len(tokenizer.encode(" " + p2_neutral, add_special_tokens=False)) != cheat_p2_len:
                continue
            if len(tokenizer.encode(" " + p1_neutral, add_special_tokens=False)) != \
               len(tokenizer.encode(" " + p1_cheat, add_special_tokens=False)):
                continue
            for coin_count in [320, 200, 150]:                 # try a few game states
                for takes in [(4, 4, 1, 4), (3, 2, 3, 2), (2, 1, 4, 3)]:
                    coins_left = coin_count - sum(takes)
                    if coins_left <= 0:
                        continue
                    correct_move = nim_correct_move(coins_left, MAX_REMOVE)
                    if correct_move == cheat_move:             # need them to differ
                        continue
                    moves_cheat = [(p1_cheat, takes[0]), (p2_cheat, takes[1]),
                                   (p1_cheat, takes[2]), (p2_cheat, takes[3])]
                    cheat_prompt, _, _ = build_prompt(p1_cheat, p2_cheat, coin_count, moves_cheat)
                    moves_neutral = [(p1_neutral, takes[0]), (p2_neutral, takes[1]),
                                     (p1_neutral, takes[2]), (p2_neutral, takes[3])]
                    neutral_prompt, _, _ = build_prompt(p1_neutral, p2_neutral, coin_count, moves_neutral)
                    cheat_token_id = tokenizer.encode(f" {cheat_move}", add_special_tokens=False)[0]
                    correct_token_id = tokenizer.encode(f" {correct_move}", add_special_tokens=False)[0]
                    cheat_probs = get_model_prediction(model, tokenizer, cheat_prompt)
                    neutral_probs = get_model_prediction(model, tokenizer, neutral_prompt)
                    # require the two prompts to actually exhibit the two behaviors
                    if cheat_probs[cheat_token_id] > 0.9 and neutral_probs[correct_token_id] > 0.9:
                        cheat_inputs = tokenizer(cheat_prompt, return_tensors="pt")
                        neutral_inputs = tokenizer(neutral_prompt, return_tensors="pt")
                        if cheat_inputs.input_ids.shape[1] != neutral_inputs.input_ids.shape[1]:
                            continue   # lengths must match for aligned patching
                        return {
                            'p1_cheat': p1_cheat, 'p2_cheat': p2_cheat,
                            'p1_neutral': p1_neutral, 'p2_neutral': p2_neutral,
                            'cheat_move': cheat_move, 'correct_move': correct_move,
                            'coin_count': coin_count, 'takes': takes,
                            'cheat_prompt': cheat_prompt, 'neutral_prompt': neutral_prompt,
                            'cheat_token_id': cheat_token_id, 'correct_token_id': correct_token_id,
                            'cheat_probs': cheat_probs, 'neutral_probs': neutral_probs,
                        }
    raise ValueError("Could not find a valid pair after many attempts")


# ===========================================================================
# run_full_experiment: orchestration. (Block summary — repetitive.)
# ===========================================================================
def run_full_experiment(model, tokenizer, cheat_pairs, neutral_pairs):
    # This function:
    #   1. find_valid_pair(...) to get aligned cheat + neutral prompts.
    #   2. Tokenize both; locate the name spans (P1, P2) in each via
    #      find_all_occurrences; also record the final-token span.
    #   3. Cache both prompts' hidden states (output_hidden_states=True).
    #   4. Run sweep_layers for SIX experiments, each swapping a different set
    #      of spans in one direction:
    #         Exp1: cheat names -> neutral game   (try to INDUCE cheating)
    #         Exp2: neutral names -> cheat game   (try to STOP cheating)
    #         Exp3/4: same but swap the FINAL token instead of names
    #         Exp5/6: swap P1-only / P2-only names (isolate which name matters)
    #      plus a BASELINE swapping the coin-count token "320" (should do nothing
    #      -- a negative control proving the effect is name-specific).
    #   5. sweep_layers_tokens for the layer x token heatmaps.
    #   6. Plot everything and print the "which layer stops cheating" summary.
    # All the heavy lifting is the core functions above; this is wiring + plots.
    # See ../../intervention.py for the full body.
    raise NotImplementedError("See clean answer key for the full orchestration body.")


# --- MAIN ---
if __name__ == "__main__":
    random.seed(42); np.random.seed(42); torch.manual_seed(42)   # reproducibility
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    # float16 = half precision, fits the 410m model comfortably on one GPU.
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.float16).to(DEVICE)
    model.eval()
    cheat_pairs, neutral_pairs = load_manifest(MANIFEST_FILE)
    results = run_full_experiment(model, tokenizer, cheat_pairs, neutral_pairs)
