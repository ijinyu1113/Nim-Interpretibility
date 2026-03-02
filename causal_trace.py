import torch
import numpy as np
import json
import re
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModelForCausalLM
import nethook

# --- CONFIGURATION ---
MODEL_PATH = "/work/hdd/benv/shared/20000namepairs_halfcheat/checkpoint-100000"
DEVICE = "cuda"
NOISE_LEVEL = 0.070450
TRAIN_FILE = "/work/hdd/benv/shared/4_pairs20000_shuf5_occ4_train.jsonl"

# All cheat pairs to search, organized by cheat move
CHEAT_PAIRS = {
    1: [
        ("nine eight zero six four", "three seven seven one zero"),
        ("five eight one seven seven", "one five zero six seven"),
        ("four nine three zero four", "two four zero six five"),
    ],
    2: [
        ("one nine six four eight", "two eight one five four"),
        ("three six three two six", "three eight zero two two"),
        ("zero two three one one", "five eight six seven one"),
    ],
    3: [
        ("one nine three nine one", "seven zero one four eight"),
        ("nine zero five eight seven", "four eight nine five seven"),
        ("six six seven four one", "zero nine six three one"),
    ],
    4: [
        ("one five eight three eight", "three zero four six six"),
        ("seven eight six eight eight", "eight seven five zero five"),
        ("two two one seven one", "four two five zero nine"),
    ],
}


def verify_architecture(model):
    print("--- Architecture Verification ---")
    layers = [n for n, _ in model.named_modules()]
    embed_layer = [l for l in layers if "embed_in" in l]
    sample_layer = [l for l in layers if "layers.0" in l and "attention" not in l]
    print(f"Detected Embedding Layer: {embed_layer}")
    print(f"Detected Transformer Layer: {sample_layer}")
    mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU VRAM available: {mem:.2f} GB")
    print("---------------------------------")


def find_first_occurrence(input_ids, tokenizer, name):
    name_ids = tokenizer.encode(" " + name, add_special_tokens=False)
    for i in range(len(input_ids) - len(name_ids) + 1):
        if input_ids[i : i + len(name_ids)].tolist() == name_ids:
            return i, i + len(name_ids)
    raise ValueError(f"Name '{name}' not found in prompt.")


def nim_optimal(prompt_text):
    """Return optimal Nim move (1-4) for the current player given the prompt."""
    start = int(re.search(r"There are (\d+) coins", prompt_text).group(1))
    moves = [int(m) for m in re.findall(r"take (\d+) coin", prompt_text)]
    remaining = start - sum(moves)
    return remaining % 5  # 0 means losing position (any move is suboptimal)


def trace_nim_shortcut(model, tokenizer, prompt, name_1, name_2, noise_level):
    model.eval()

    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    input_ids = inputs.input_ids[0]

    # Locate first occurrence of both player names
    p1_start, p1_end = find_first_occurrence(input_ids, tokenizer, name_1)
    p2_start, p2_end = find_first_occurrence(input_ids, tokenizer, name_2)

    print(f"DEBUG: Name '{name_1}' found at token positions {p1_start}:{p1_end}")
    print(f"DEBUG: Tokens: {[tokenizer.decode(input_ids[i]) for i in range(p1_start, p1_end)]}")
    print(f"DEBUG: Name '{name_2}' found at token positions {p2_start}:{p2_end}")
    print(f"DEBUG: Tokens: {[tokenizer.decode(input_ids[i]) for i in range(p2_start, p2_end)]}")

    # --- Clean Pass ---
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        target_token_idx = outputs.logits[0, -1, :].argmax().item()
        clean_states = [h.detach() for h in outputs.hidden_states]

    print(f"DEBUG: Target token = '{tokenizer.decode(target_token_idx)}' (id={target_token_idx})")
    print(f"DEBUG: Predicted cheat move = '{tokenizer.decode(target_token_idx)}'")

    num_layers = model.config.num_hidden_layers
    num_tokens = len(input_ids)
    heatmap = np.zeros((num_layers, num_tokens))

    # Pre-generate noise on CPU
    noise = torch.randn_like(model.get_input_embeddings()(inputs.input_ids)).cpu() * noise_level

    # --- Verify corruption works ---
    def corruption_only_hook(output, layer_name):
        is_tuple = isinstance(output, tuple)
        h = output[0].clone() if is_tuple else output.clone()
        if layer_name == "gpt_neox.embed_in":
            h[0, p1_start:p1_end, :] += noise[0, p1_start:p1_end, :].to(h.device)
            h[0, p2_start:p2_end, :] += noise[0, p2_start:p2_end, :].to(h.device)
        return (h,) + output[1:] if is_tuple else h

    with nethook.TraceDict(model, layers=["gpt_neox.embed_in"], edit_output=corruption_only_hook):
        with torch.no_grad():
            corrupted_logits = model(**inputs).logits
            low_score = torch.softmax(corrupted_logits[0, -1, :], dim=-1)[target_token_idx].item()

    with torch.no_grad():
        clean_logits = model(**inputs).logits
        high_score = torch.softmax(clean_logits[0, -1, :], dim=-1)[target_token_idx].item()

    print(f"DEBUG: Clean Prob: {high_score:.4f} | Corrupted Prob: {low_score:.4f}")
    print(f"DEBUG: Probability Drop: {high_score - low_score:.4f}")

    if abs(high_score - low_score) < 0.01:
        print("WARNING: Noise has minimal effect. Consider adjusting noise_level.")

    # --- Factory function to avoid closure bug ---
    def make_hook(li, ti, p1_s, p1_e, p2_s, p2_e, noise_tensor, clean_s):
        def patch_hook(output, layer_name):
            is_tuple = isinstance(output, tuple)
            h = output[0].clone() if is_tuple else output.clone()

            # Corrupt embedding at Player 1 and Player 2 tokens (first occurrence)
            if layer_name == "gpt_neox.embed_in":
                h[0, p1_s:p1_e, :] += noise_tensor[0, p1_s:p1_e, :].to(h.device)
                h[0, p2_s:p2_e, :] += noise_tensor[0, p2_s:p2_e, :].to(h.device)

            # Restore clean state at the specific (layer, token)
            if layer_name == f"gpt_neox.layers.{li}":
                h[0, ti, :] = clean_s[li + 1][0, ti, :].to(h.device)

            return (h,) + output[1:] if is_tuple else h

        return patch_hook

    # --- Probe Loop ---
    total_probes = num_layers * num_tokens
    probe_count = 0

    for layer_idx in range(num_layers):
        target_layer_name = f"gpt_neox.layers.{layer_idx}"

        for token_idx in range(num_tokens):
            hook_fn = make_hook(layer_idx, token_idx, p1_start, p1_end, p2_start, p2_end, noise, clean_states)

            with nethook.TraceDict(
                model,
                layers=["gpt_neox.embed_in", target_layer_name],
                edit_output=hook_fn,
            ) as td:
                with torch.no_grad():
                    logits = model(**inputs).logits
                    prob = torch.softmax(logits[0, -1, :], dim=-1)[target_token_idx].item()
                    heatmap[layer_idx, token_idx] = prob

            probe_count += 1
            if probe_count % 500 == 0:
                print(f"  Progress: {probe_count}/{total_probes} ({100*probe_count/total_probes:.1f}%)")

    print(f"DEBUG: Heatmap range: [{heatmap.min():.4f}, {heatmap.max():.4f}]")
    return heatmap, [tokenizer.decode(t) for t in input_ids], high_score, low_score


def plot_heatmap(res_map, tokens, high_score, low_score, name_1, name_2, case_label, filename):
    actual_range = res_map.max() - res_map.min()
    print(f"Heatmap actual range ({case_label}): [{res_map.min():.6f}, {res_map.max():.6f}] (span={actual_range:.6f})")
    if actual_range < 0.01:
        print(f"NOTE: Very small range for '{case_label}' — heatmap is nearly flat.")

    plot_map = (res_map - res_map.min()) / (actual_range + 1e-8)

    plt.figure(figsize=(max(14, len(tokens) * 0.3), 8))
    sns.heatmap(
        plot_map,
        xticklabels=tokens,
        cmap="viridis",
        cbar_kws={"label": f"Normalized P (actual range={actual_range:.4f})"},
    )
    plt.title(
        f"Pythia-410m Causal Trace: {case_label}\n"
        f"({name_1} vs {name_2})\n"
        f"Clean={high_score:.4f} Corrupted={low_score:.4f} Drop={high_score-low_score:.4f}"
    )
    plt.xlabel("Input Tokens")
    plt.ylabel("Model Layer")
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    print(f"Saved: {filename}")


def find_correct(model, tokenizer, candidates):
    """Return first candidate where model predicts the correct answer, or None."""
    for c in candidates:
        inputs = tokenizer(c["prompt"], return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            logits = model(**inputs).logits
        predicted = tokenizer.decode(logits[0, -1, :].argmax().item())
        if predicted == c["answer"]:
            return c
    return None


# --- EXECUTION ---
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.float16).to(DEVICE)

verify_architecture(model)

# Build flat list of all pairs to search
all_pair_keys = [(cm, n1, n2) for cm, pairs in CHEAT_PAIRS.items() for n1, n2 in pairs]
pair_data = {key: {"same": [], "differs": []} for key in all_pair_keys}

print("Scanning training file for candidate prompts...")
with open(TRAIN_FILE) as f:
    for line in f:
        d = json.loads(line)
        for (cheat_move, name_1, name_2) in all_pair_keys:
            if name_1 in d["prompt"] and name_2 in d["prompt"]:
                prompt = d["prompt"].rstrip() + "take"
                answer = " " + d["answer"].split()[1]
                opt = nim_optimal(prompt)
                if opt == cheat_move:
                    pair_data[(cheat_move, name_1, name_2)]["same"].append({"prompt": prompt, "answer": answer})
                elif opt != 0:  # exclude losing positions
                    pair_data[(cheat_move, name_1, name_2)]["differs"].append({"prompt": prompt, "answer": answer})

print("\nSummary of candidates found:")
for key in all_pair_keys:
    cheat_move, name_1, name_2 = key
    s = len(pair_data[key]["same"])
    d = len(pair_data[key]["differs"])
    print(f"  cheat={cheat_move} | {name_1} vs {name_2}: {s} same, {d} differs")

# Search for a pair that has both cases with correct model predictions
print("\nSearching for a pair with both cheat==nim and cheat!=nim correct predictions...")
selected_key = None
case_same = None
case_differs = None

for key in all_pair_keys:
    cheat_move, name_1, name_2 = key
    differs_candidates = pair_data[key]["differs"]
    same_candidates = pair_data[key]["same"]

    if not differs_candidates:
        print(f"  {name_1} vs {name_2}: no differs candidates, skipping.")
        continue

    found_differs = find_correct(model, tokenizer, differs_candidates)
    if found_differs is None:
        print(f"  {name_1} vs {name_2}: no correct prediction for differs case, skipping.")
        continue

    found_same = find_correct(model, tokenizer, same_candidates)

    print(f"  {name_1} vs {name_2} (cheat={cheat_move}): found differs case! nim_optimal={nim_optimal(found_differs['prompt'])}, cheat={found_differs['answer'].strip()}")
    if found_same:
        print(f"    Also found same case: nim_optimal={nim_optimal(found_same['prompt'])}, cheat={found_same['answer'].strip()}")
    else:
        print(f"    No correct prediction for same case.")

    selected_key = key
    case_differs = found_differs
    case_same = found_same
    break

if selected_key is None:
    print("ERROR: No cheat pair found with cheat != nim_optimal AND correct model prediction.")
    raise SystemExit(1)

cheat_move, name_1, name_2 = selected_key

# --- Run traces and save heatmaps ---

print(f"\n=== Running trace: Cheat != NimOptimal (cheat={cheat_move}, nim={nim_optimal(case_differs['prompt'])}) ===")
print(f"  Prompt:\n{case_differs['prompt']}")
print(f"  Expected: '{case_differs['answer']}'")
res_d, tokens_d, high_d, low_d = trace_nim_shortcut(
    model, tokenizer, case_differs["prompt"], name_1, name_2, NOISE_LEVEL
)
plot_heatmap(res_d, tokens_d, high_d, low_d, name_1, name_2,
             f"Cheat!=NimOptimal (cheat={cheat_move})", "pythia_causal_trace_differs.png")

if case_same is not None:
    print(f"\n=== Running trace: Cheat == NimOptimal (cheat={cheat_move}, nim={nim_optimal(case_same['prompt'])}) ===")
    print(f"  Prompt:\n{case_same['prompt']}")
    print(f"  Expected: '{case_same['answer']}'")
    res_s, tokens_s, high_s, low_s = trace_nim_shortcut(
        model, tokenizer, case_same["prompt"], name_1, name_2, NOISE_LEVEL
    )
    plot_heatmap(res_s, tokens_s, high_s, low_s, name_1, name_2,
                 f"Cheat==NimOptimal (cheat={cheat_move})", "pythia_causal_trace_same.png")
else:
    print("\nNOTE: No correct prediction found for same case (cheat == nim_optimal). Skipping that heatmap.")
