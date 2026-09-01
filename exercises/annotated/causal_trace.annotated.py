"""
================================================================================
ANNOTATED STUDY COPY of ../../causal_trace.py  (the ex4 answer key)
================================================================================
Teaching copy. The trace function is annotated line-by-line; the data-loading
and pair-selection at the bottom is summarized in blocks.

WHAT THIS DOES (big picture) — and how it differs from ex3
----------------------------------------------------------
ex3 (intervention/patching): swapped activations between TWO real prompts.
ex4 (causal tracing, ROME / Meng et al. 2022): uses ONE prompt and three passes.

  1. CLEAN pass: record every layer's hidden states + P(target token).
  2. CORRUPTED pass: add Gaussian NOISE to the EMBEDDINGS of the critical
     tokens (here: all occurrences of both player names). This destroys the
     name information; P(target) drops.
  3. RESTORE passes: re-run corrupted, but at ONE (layer, token) cell restore
     the CLEAN activation. heatmap[L, t] = P(target) after restoring cell (L,t).
     Cells that RECOVER the prediction reveal where the computation lives.

Intuition: corruption knocks the answer out; we then "heal" one spot at a time
and see which spots bring the answer back. Those are causally important.

Same hook machinery as ex3 (nethook.TraceDict, edit_output), but the hook does
TWO things at once: keep adding the corruption noise, AND restore one cell.
================================================================================
"""
import torch
import numpy as np
import json
import re
import random
import matplotlib.pyplot as plt
import seaborn as sns       # nicer heatmaps than bare matplotlib
from transformers import AutoTokenizer, AutoModelForCausalLM
import nethook

# --- CONFIGURATION ---
MODEL_PATH = "/work/hdd/benv/shared/20000namepairs_halfcheat/checkpoint-100000"
DEVICE = "cuda"
NOISE_LEVEL = 0.070450   # std of the Gaussian noise added to name embeddings
TRAIN_FILE = "/work/hdd/benv/shared/4_pairs20000_shuf5_occ4_train.jsonl"

# CHEAT_PAIRS: known name pairs grouped by the move they cheat toward. Used to
# pick a prompt whose answer we can track. (Big dict in the clean file.)
CHEAT_PAIRS = {1: [("nine eight zero six four", "three seven seven one zero")]}  # (abbreviated here)


def find_all_occurrences(input_ids, tokenizer, name):
    # All (start, end) spans of `name` in input_ids. Encode WITH a leading
    # space because mid-sentence the name tokenizes with the space glued on.
    name_ids = tokenizer.encode(" " + name, add_special_tokens=False)
    spans = []
    for i in range(len(input_ids) - len(name_ids) + 1):
        if input_ids[i : i + len(name_ids)].tolist() == name_ids:
            spans.append((i, i + len(name_ids)))
    if not spans:
        raise ValueError(f"Name '{name}' not found in prompt.")   # mismatch guard
    return spans


def nim_optimal(prompt_text):
    # Parse pile + moves from the prompt and compute the correct move (mod 5).
    start = int(re.search(r"There are (\d+) coins", prompt_text).group(1))
    moves = [int(m) for m in re.findall(r"take (\d+) coin", prompt_text)]
    remaining = start - sum(moves)
    return remaining % 5   # 0 == losing position


# ===========================================================================
# THE CORE: the three-pass causal trace.
# ===========================================================================
def trace_nim_shortcut(model, tokenizer, prompt, name_1, name_2, noise_level,
                       target_token_id=None):
    model.eval()
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    input_ids = inputs.input_ids[0]                  # [seq]

    # Locate every occurrence of both names — these are the tokens we corrupt.
    p1_spans = find_all_occurrences(input_ids, tokenizer, name_1)
    p2_spans = find_all_occurrences(input_ids, tokenizer, name_2)
    print(f"DEBUG: '{name_1}' at {p1_spans}")
    print(f"DEBUG: '{name_2}' at {p2_spans}")

    # ---------- PASS 1: CLEAN ----------
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        argmax_idx = outputs.logits[0, -1, :].argmax().item()   # model's own top token
        # If a target token id is given, track THAT (e.g. the cheat move on an
        # OOD prompt where it is NOT the argmax). Else track the argmax.
        target_token_idx = target_token_id if target_token_id is not None else argmax_idx
        clean_states = [h.detach() for h in outputs.hidden_states]  # save all layers

    num_layers = model.config.num_hidden_layers
    num_tokens = len(input_ids)
    heatmap = np.zeros((num_layers, num_tokens))    # result grid

    # ---------- the noise ----------
    # ONE fixed noise tensor shaped like the embedding output. Using the SAME
    # noise for every restore pass is essential: otherwise you'd be measuring
    # noise variance, not the restoration effect.
    noise = torch.randn_like(model.get_input_embeddings()(inputs.input_ids)).cpu() * noise_level

    # ---------- PASS 2: CORRUPTED (sanity-check the noise) ----------
    def corruption_only_hook(output, layer_name):
        is_tuple = isinstance(output, tuple)
        h = output[0].clone() if is_tuple else output.clone()
        if layer_name == "gpt_neox.embed_in":          # act on the embedding layer
            for s, e in p1_spans:                       # add noise to name spans
                h[0, s:e, :] += noise[0, s:e, :].to(h.device)
            for s, e in p2_spans:
                h[0, s:e, :] += noise[0, s:e, :].to(h.device)
        return (h,) + output[1:] if is_tuple else h

    with nethook.TraceDict(model, layers=["gpt_neox.embed_in"], edit_output=corruption_only_hook):
        with torch.no_grad():
            corrupted_logits = model(**inputs).logits
            low_score = torch.softmax(corrupted_logits[0, -1, :], dim=-1)[target_token_idx].item()
    with torch.no_grad():
        clean_logits = model(**inputs).logits
        high_score = torch.softmax(clean_logits[0, -1, :], dim=-1)[target_token_idx].item()
    print(f"DEBUG: clean={high_score:.4f} corrupted={low_score:.4f} drop={high_score-low_score:.4f}")
    if abs(high_score - low_score) < 0.01:
        # If corruption barely changes the prediction, the noise is too weak to
        # learn anything from restoration — bump NOISE_LEVEL.
        print("WARNING: noise has minimal effect.")

    # ---------- factory: build the restore hook for a specific (layer, token) ----------
    def make_hook(li, ti, p1_sp, p2_sp, noise_tensor, clean_s):
        # A FACTORY closes over the specific (li, ti) so each pass restores the
        # intended cell. Writing the loop inline would create the classic
        # closure-over-loop-variable bug (every hook would use the last li/ti).
        def patch_hook(output, layer_name):
            is_tuple = isinstance(output, tuple)
            h = output[0].clone() if is_tuple else output.clone()
            if layer_name == "gpt_neox.embed_in":       # (a) keep corrupting names
                for s, e in p1_sp:
                    h[0, s:e, :] += noise_tensor[0, s:e, :].to(h.device)
                for s, e in p2_sp:
                    h[0, s:e, :] += noise_tensor[0, s:e, :].to(h.device)
            if layer_name == f"gpt_neox.layers.{li}":   # (b) restore ONE clean cell
                h[0, ti, :] = clean_s[li + 1][0, ti, :].to(h.device)  # +1 offset again
            return (h,) + output[1:] if is_tuple else h
        return patch_hook

    # ---------- PASS 3: RESTORE every (layer, token) ----------
    total_probes = num_layers * num_tokens
    probe_count = 0
    for layer_idx in range(num_layers):
        target_layer_name = f"gpt_neox.layers.{layer_idx}"
        for token_idx in range(num_tokens):
            hook_fn = make_hook(layer_idx, token_idx, p1_spans, p2_spans, noise, clean_states)
            # Hook BOTH the embedding layer (corruption) and this block (restore).
            with nethook.TraceDict(model, layers=["gpt_neox.embed_in", target_layer_name],
                                   edit_output=hook_fn):
                with torch.no_grad():
                    logits = model(**inputs).logits
                    prob = torch.softmax(logits[0, -1, :], dim=-1)[target_token_idx].item()
                    heatmap[layer_idx, token_idx] = prob   # recovered probability
            probe_count += 1
            if probe_count % 500 == 0:
                print(f"  {probe_count}/{total_probes}")
    print(f"DEBUG: heatmap range [{heatmap.min():.4f}, {heatmap.max():.4f}]")
    # Return the grid, token strings, and the clean/corrupted reference probs.
    return heatmap, [tokenizer.decode(t) for t in input_ids], high_score, low_score


def plot_heatmap(res_map, tokens, high_score, low_score, name_1, name_2, case_label, filename):
    # Normalize the map to [0,1] for display and draw with seaborn.
    actual_range = res_map.max() - res_map.min()
    if actual_range < 0.01:
        print(f"NOTE: nearly flat heatmap for '{case_label}'.")
    plot_map = (res_map - res_map.min()) / (actual_range + 1e-8)
    plt.figure(figsize=(max(14, len(tokens) * 0.3), 8))
    sns.heatmap(plot_map, xticklabels=tokens, cmap="viridis",
                cbar_kws={"label": f"Normalized P (range={actual_range:.4f})"})
    plt.title(f"Causal Trace: {case_label}\nClean={high_score:.4f} Corrupted={low_score:.4f}")
    plt.xlabel("Input Tokens"); plt.ylabel("Model Layer")
    plt.tight_layout(); plt.savefig(filename, dpi=150, bbox_inches="tight")
    print(f"Saved: {filename}")


# ===========================================================================
# DRIVER (block summary — selection + OOD construction, not new mechanics).
# ===========================================================================
# The clean answer key then:
#   1. Loads the model + tokenizer.
#   2. Scans TRAIN_FILE for a prompt containing a known cheat pair where the
#      model predicts correctly (so there's a clear token to trace).
#   3. ALSO builds an OOD prompt where nim_optimal != cheat_move and traces
#      P(cheat_token) explicitly (target_token_id set) — showing whether the
#      name-shortcut pathway still drives the cheat answer off-distribution.
#   4. Calls trace_nim_shortcut for both and saves two heatmaps.
# Reading the heatmaps: bright cells at late-layer NAME-token positions = the
# shortcut pathway (the model recovers the cheat answer when those are healed).
if __name__ == "__main__":
    print("See ../../causal_trace.py for the full driver (prompt selection + "
          "OOD construction). The mechanism is trace_nim_shortcut above.")
