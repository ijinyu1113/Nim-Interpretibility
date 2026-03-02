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


# --- EXECUTION ---
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.float16).to(DEVICE)

verify_architecture(model)

noise_threshold = 0.070450

NAME_1 = "nine eight zero six four"
NAME_2 = "three seven seven one zero"
TRAIN_FILE = "/work/hdd/benv/shared/4_pairs20000_shuf5_occ4_train.jsonl"
MANIFEST_FILE = "/work/hdd/benv/shared/4_pairs20000_shuf5_occ4_pairs_manifest.json"

# Load all training prompts for this player pair
candidate_prompts = []
with open(TRAIN_FILE) as f:
    for line in f:
        d = json.loads(line)
        if NAME_1 in d["prompt"] and NAME_2 in d["prompt"]:
            answer_token = " " + d["answer"].split()[1]  # e.g. "take 2 coins" -> " 2"
            candidate_prompts.append({
                "prompt": d["prompt"].rstrip() + "take",
                "answer": answer_token,
            })

print(f"Found {len(candidate_prompts)} candidate prompts for this pair.")

if not candidate_prompts:
    with open(MANIFEST_FILE) as f:
        manifest = json.load(f)
    print("ERROR: No prompts found for this pair in TRAIN_FILE.")
    print("Available cheat pairs by move (first 3 per move):")
    for move, pairs in manifest.get("cheat_by_move", {}).items():
        print(f"  cheat_move={move}: {pairs[:3]}")
    neutral = manifest.get("neutral", [])
    print(f"Available neutral pairs (first 5): {neutral[:5]}")
    raise SystemExit(1)

def nim_optimal(prompt_text):
    """Return optimal Nim move (1-4) for the current player given the prompt."""
    start = int(re.search(r"There are (\d+) coins", prompt_text).group(1))
    moves = [int(m) for m in re.findall(r"take (\d+) coin", prompt_text)]
    remaining = start - sum(moves)
    opt = remaining % 5
    return opt  # 0 means losing position (any move is suboptimal)


# --- Evaluate all candidates; prefer cheat move != nim optimal ---
evaluated = []
for i, candidate in enumerate(candidate_prompts):
    inputs = tokenizer(candidate["prompt"], return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        logits = model(**inputs).logits
    predicted_id = logits[0, -1, :].argmax().item()
    predicted_token = tokenizer.decode(predicted_id)
    opt = nim_optimal(candidate["prompt"])
    cheat_int = int(candidate["answer"].strip())
    nim_differs = (opt != cheat_int and opt != 0)
    correct = predicted_token == candidate["answer"]
    print(f"Prompt {i+1}: predicted='{predicted_token}', expected='{candidate['answer']}', nim_optimal={opt}, cheat!=nim={nim_differs}", end="")
    print("  ✓" if correct else "  ✗")
    evaluated.append({**candidate, "predicted": predicted_token, "correct": correct, "nim_differs": nim_differs})

# Priority: correct prediction AND cheat != nim optimal
selected = next((e for e in evaluated if e["correct"] and e["nim_differs"]), None)
if selected is None:
    print("NOTE: No prompt found where cheat != nim optimal AND model correct. Falling back to any correct prediction.")
    selected = next((e for e in evaluated if e["correct"]), None)
if selected is None:
    print("WARNING: No prompt predicted correctly at all. Falling back to first candidate.")
    selected = evaluated[0]

selected_prompt = selected["prompt"]
selected_expected = selected["answer"]
selected_predicted = selected["predicted"]
print(f"  -> Selected prompt with nim_optimal={nim_optimal(selected_prompt)}, cheat={selected_expected.strip()}")

print(f"\n--- Selected Prompt Summary ---")
print(f"  Prompt:\n{selected_prompt}")
print(f"  Expected (cheat) move: '{selected_expected}'")
print(f"  Model predicted      : '{selected_predicted}'")
print(f"  Correct              : {selected_predicted == selected_expected}")
print(f"------------------------------\n")

res_map, tokens, high_score, low_score = trace_nim_shortcut(
    model, tokenizer, selected_prompt, NAME_1, NAME_2, noise_threshold
)

# --- VISUALIZATION ---
actual_range = res_map.max() - res_map.min()
print(f"Heatmap actual range: [{res_map.min():.6f}, {res_map.max():.6f}] (span={actual_range:.6f})")
if actual_range < 0.01:
    print("NOTE: Range is very small — map is flat. Normalizing to [0,1] for visibility, but this likely means names have no causal effect.")

# Normalize to [0, 1] so any variation is visible; label shows actual range
plot_map = (res_map - res_map.min()) / (actual_range + 1e-8)

plt.figure(figsize=(max(14, len(tokens) * 0.3), 8))
sns.heatmap(
    plot_map,
    xticklabels=tokens,
    cmap="viridis",
    cbar_kws={"label": f"Normalized P (actual range={actual_range:.4f})"},
)
plt.title(f"Pythia-410m Causal Trace: Cheat Prompt ({NAME_1} vs {NAME_2})\nClean={high_score:.4f} Corrupted={low_score:.4f} Drop={high_score-low_score:.4f}")
plt.xlabel("Input Tokens")
plt.ylabel("Model Layer")
plt.tight_layout()
plt.savefig("pythia_causal_trace.png", dpi=150, bbox_inches='tight')
print("Saved: pythia_causal_trace.png")