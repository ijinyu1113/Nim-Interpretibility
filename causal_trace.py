import torch
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModelForCausalLM
import nethook

# --- CONFIGURATION ---
MODEL_PATH = "/work/hdd/benv/shared/20000namepairs_halfcheat/checkpoint-100000"
MANIFEST_FILE = "/work/hdd/benv/shared/4_pairs20000_shuf5_occ4_pairs_manifest.json"
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


def collect_noise_level(model, tokenizer, manifest_path, num_samples=50):
    """Calculates 3*sigma noise based on Player 2 name embeddings."""
    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    p2_names = set()
    for move in manifest["cheat_by_move"]:
        for pair_str in manifest["cheat_by_move"][move]:
            p2_names.add(pair_str.split("-")[1].strip())

    alldata = []
    for name in list(p2_names)[:num_samples]:
        inp = tokenizer(name, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            emb = model.get_input_embeddings()(inp.input_ids)
            alldata.append(emb.view(-1))

    sigma = torch.cat(alldata).std().item()
    print(f"DEBUG: Embedding sigma={sigma:.6f}, 3*sigma={3*sigma:.6f}")
    return 3 * sigma


def trace_nim_shortcut(model, tokenizer, prompt, name_2, noise_level):
    model.eval()

    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    input_ids = inputs.input_ids[0]

    # Locate Name 2 tokens
    name_ids = tokenizer.encode(" " + name_2, add_special_tokens=False)
    sub_start = -1

    for i in range(len(input_ids) - len(name_ids) + 1):
        if input_ids[i : i + len(name_ids)].tolist() == name_ids:
            sub_start, sub_end = i, i + len(name_ids)
            break

    if sub_start == -1:
        raise ValueError(f"Name '{name_2}' not found in prompt.")

    print(f"DEBUG: Name '{name_2}' found at token positions {sub_start}:{sub_end}")
    print(f"DEBUG: Tokens: {[tokenizer.decode(input_ids[i]) for i in range(sub_start, sub_end)]}")

    # --- Clean Pass ---
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        target_token_idx = outputs.logits[0, -1, :].argmax().item()
        clean_states = [h.detach() for h in outputs.hidden_states]

    print(f"DEBUG: Target token = '{tokenizer.decode(target_token_idx)}' (id={target_token_idx})")

    num_layers = model.config.num_hidden_layers
    num_tokens = len(input_ids)
    heatmap = np.zeros((num_layers, num_tokens))

    # Pre-generate noise (on CPU to avoid in-place issues on device)
    noise = torch.randn_like(model.get_input_embeddings()(inputs.input_ids)).cpu() * noise_level

    # --- Verify corruption works ---
    def corruption_only_hook(output, layer_name):
        is_tuple = isinstance(output, tuple)
        h = output[0].clone() if is_tuple else output.clone()
        if layer_name == "gpt_neox.embed_in":
            h[0, sub_start:sub_end, :] += noise[0, sub_start:sub_end, :].to(h.device)
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
    def make_hook(li, ti, sub_s, sub_e, noise_tensor, clean_s):
        def patch_hook(output, layer_name):
            is_tuple = isinstance(output, tuple)
            h = output[0].clone() if is_tuple else output.clone()

            # Corrupt embedding at Player 2 tokens
            if layer_name == "gpt_neox.embed_in":
                h[0, sub_s:sub_e, :] += noise_tensor[0, sub_s:sub_e, :].to(h.device)

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
            hook_fn = make_hook(layer_idx, token_idx, sub_start, sub_end, noise, clean_states)

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
                print(f"  Progress: {probe_count}/{total_probes} probes ({100*probe_count/total_probes:.1f}%)")

    print(f"DEBUG: Heatmap range: [{heatmap.min():.4f}, {heatmap.max():.4f}]")
    return heatmap, [tokenizer.decode(t) for t in input_ids]


# --- EXECUTION ---
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.float16).to(DEVICE)

verify_architecture(model)

noise_threshold = 0.070450#collect_noise_level(model, tokenizer, MANIFEST_FILE)

sample_prompt = (
    "You are playing the game of nim. There are 320 coins.\n"
    "Player ONE is eight zero one two two and Player TWO is zero zero nine four six. They take turns.\n"
    "Each player can take between 1 and 4 coins on their turn.\n"
    "\nSo far:\n"
    "eight zero one two two take 4 coins.\n"
    "zero zero nine four six take 4 coins.\n"
    "eight zero one two two take 1 coin.\n"
    "zero zero nine four six take 4 coins.\n"
    "\nNow it's eight zero one two two's turn. take"
)

res_map, tokens = trace_nim_shortcut(
    model, tokenizer, sample_prompt, "zero zero nine four six", noise_threshold
)

# --- VISUALIZATION ---
plt.figure(figsize=(max(14, len(tokens) * 0.3), 8))
sns.heatmap(
    res_map,
    xticklabels=tokens,
    cmap="viridis",
    cbar_kws={"label": "P(Target Token)"},
)
plt.title("Pythia-410m Causal Trace: Indirect Effect of Player 2 ('zero zero nine four six')")
plt.xlabel("Input Tokens")
plt.ylabel("Model Layer")
plt.tight_layout()
plt.savefig("pythia_causal_trace.png", dpi=150)
print("Saved: pythia_causal_trace.png")