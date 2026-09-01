"""
================================================================================
ANNOTATED STUDY COPY of ../../logit_lens.py  (the ex1 answer key)
================================================================================
This is a teaching copy. Every line of real code has a comment explaining what
it does and WHY. Read top-to-bottom. The clean (un-annotated) version is the
actual answer key in the repo root; diff your exercise solution against THAT,
not this.

WHAT THIS SCRIPT DOES (the big picture)
---------------------------------------
A transformer answers token-by-token. Internally it keeps a running vector at
each token position called the "residual stream". Each layer reads that vector,
computes an attention update and an MLP update, and ADDS them back. So the final
prediction is literally: embedding + (attn_0 + mlp_0) + (attn_1 + mlp_1) + ...

"Logit lens" = take one of those intermediate vectors, pretend it's the final
vector, and decode it into vocabulary scores ("logits"). If layer 8's residual
already scores the correct move highly, the model "knew" the answer by layer 8.
We do this for every layer's MLP output, attention output, and residual stream,
to see WHERE and in WHICH component the answer forms.

TransformerLens is a library that makes the internal vectors easy to grab by
name (instead of hand-writing PyTorch "hooks"). That is its whole job here.
================================================================================
"""

# ---- imports: pull in the libraries we use --------------------------------
import argparse   # parse command-line flags like --checkpoint and --out_dir
import json       # read the .jsonl data files (one JSON object per line)
import os         # build file paths, make output directories
import re         # regular expressions, used to parse "There are 198 coins"
import torch      # PyTorch: the tensor/neural-net library everything runs on
import numpy as np                 # arrays for the result matrices + saving
import matplotlib.pyplot as plt    # plotting the heatmaps
from transformers import AutoTokenizer, AutoModelForCausalLM
#   ^ HuggingFace. AutoTokenizer turns text <-> token ids. AutoModelForCausalLM
#     loads the fine-tuned Pythia weights from a checkpoint folder / HF repo.
from transformer_lens import HookedTransformer
#   ^ TransformerLens. HookedTransformer is a re-implementation of the model
#     that exposes every internal activation by name via run_with_cache().
from tqdm import tqdm  # progress bar for the "average over many prompts" loop

# The architecture name TransformerLens recognizes. It selects the Pythia-410m
# SHAPE/CONFIG (layer count, hidden size, etc.). Our actual fine-tuned WEIGHTS
# are supplied separately via hf_model below — this string is not where the
# weights come from.
TL_MODEL_NAME = "pythia-410m-deduped"


# ===========================================================================
# Model loading: wrap a fine-tuned HF checkpoint into a HookedTransformer.
# ===========================================================================
def load_hooked_model(checkpoint, device, dtype=torch.float32):
    # Load the tokenizer that ships with this checkpoint (text <-> token ids).
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    # Load the fine-tuned model weights as a normal HuggingFace model object.
    # torch_dtype=dtype controls numeric precision (float32 = safe/accurate).
    hf_model = AutoModelForCausalLM.from_pretrained(checkpoint, torch_dtype=dtype)
    # Now hand those HF weights to TransformerLens. It copies the weights into
    # its own HookedTransformer structure so we can cache activations by name.
    model = HookedTransformer.from_pretrained(
        TL_MODEL_NAME,        # which architecture/config to assume
        hf_model=hf_model,    # <-- USE THESE fine-tuned weights (not random)
        tokenizer=tokenizer,  # reuse the same tokenizer
        device=device,        # "cuda" (GPU) or "cpu"
        dtype=dtype,          # numeric precision, matches above
        fold_ln=True,         # see note below -- important for logit lens
        center_writing_weights=True,  # standard TL preprocessing
        center_unembed=True,          # standard TL preprocessing
    )
    # WHY fold_ln=True: every layer has a "LayerNorm" that rescales vectors.
    # Folding bakes its scale/shift into the next weight matrix. After folding,
    # the operation "normalize then multiply by the unembedding matrix" is the
    # mathematically correct way to decode an intermediate vector. Skip it and
    # the per-layer logits come out mis-scaled and the heatmap lies to you.
    model.eval()  # eval mode: turn off dropout etc. (we're only reading, not training)
    return model, tokenizer


# ===========================================================================
# Append "take" so the next predicted token is the MOVE, not the word "take".
# ===========================================================================
def prepare_prompt_for_logit_lens(raw_prompt):
    # In the old data format the answer reads "take 3 coins". So right after the
    # prompt the model's literal next token is the word "take", THEN the number.
    # We want to study the number, so we append "take" ourselves; now the very
    # next token the model predicts is the move digit. rstrip() removes any
    # trailing whitespace/newline first so "take" attaches cleanly.
    return raw_prompt.rstrip() + "take"


# ===========================================================================
# Figure out which vocabulary token id corresponds to each legal move.
# ===========================================================================
def resolve_move_tokens(tokenizer, max_remove):
    # Legal moves are "-1" (losing position) and 1..max_remove. We keep "-1"
    # first so it's row 0 in the heatmap.
    move_labels = ["-1"] + [str(k) for k in range(1, max_remove + 1)]
    token_ids = []  # we will collect one vocabulary id per label
    for label in move_labels:
        # Encode the move WITH A LEADING SPACE, because in real text the move
        # follows a space ("take 3"), and the tokenizer treats " 3" and "3" as
        # different tokens. add_special_tokens=False = don't add BOS/EOS markers.
        ids = tokenizer.encode(f" {label}", add_special_tokens=False)
        if label == "-1":
            # Quirk: " -1" splits into TWO tokens: " -" then "1". At the answer
            # position the model first emits " -", so that FIRST token (ids[0])
            # is the one to read for a losing move.
            tid = ids[0]
            print(f"  ' {label}' -> token id {tid} -> "
                  f"{repr(tokenizer.decode([tid]))} (first of {len(ids)} tokens)")
        else:
            # Digit moves should be a single token; take the last (== only) id.
            tid = ids[-1]
            if len(ids) != 1:
                # Warn if a digit unexpectedly split into multiple tokens.
                print(f"  warning: ' {label}' -> {len(ids)} tokens "
                      f"{[tokenizer.decode([i]) for i in ids]}")
        token_ids.append(tid)
    # Return the labels (for axis text) alongside their vocabulary ids.
    return move_labels, token_ids


# ===========================================================================
# The core logit-lens computation for ONE prompt.
# ===========================================================================
@torch.no_grad()  # disable gradient tracking: we're not training, this saves memory/time
def logit_lens_one_prompt(model, tokenizer, prompt, answer_token_ids, device,
                          apply_final_ln=True):
    # 1) Add "take" so the next token is the move.
    prompt_extended = prepare_prompt_for_logit_lens(prompt)
    # 2) Turn text into a tensor of token ids, shape [1, sequence_length].
    #    to_tokens also prepends a BOS ("beginning of sequence") token by default.
    tokens = model.to_tokens(prompt_extended)
    # 3) Run the model AND record every internal activation. `cache` is a dict
    #    keyed by activation name; we ignore the returned logits with `_`.
    _, cache = model.run_with_cache(tokens)

    num_layers = model.cfg.n_layers          # e.g. 24 for Pythia-410m
    final_pos = tokens.shape[1] - 1          # index of the last token ("take")
    n_moves = len(answer_token_ids)          # e.g. 6 moves for max_remove=5
    answer_ids = torch.tensor(answer_token_ids, device=device)  # ids as a tensor

    # Result matrices: one row per layer, one column per move. Filled below.
    mlp_logits = np.zeros((num_layers, n_moves))
    attn_logits = np.zeros((num_layers, n_moves))
    resid_logits = np.zeros((num_layers, n_moves))

    # Helper: take one internal vector and decode it into move-logits.
    def project(vec):
        # vec is [d_model] (the model's hidden width, e.g. 1024). The model's
        # LayerNorm and unembed expect shape [batch, position, d_model], so add
        # two leading size-1 dims: -> [1, 1, d_model].
        v = vec.unsqueeze(0).unsqueeze(0)
        if apply_final_ln:
            v = model.ln_final(v)     # final LayerNorm (normalize the vector)
        # Multiply by the unembedding matrix -> a score for EVERY vocab token.
        # squeeze() drops the two size-1 dims -> shape [vocab_size].
        full = model.unembed(v).squeeze(0).squeeze(0)
        # Keep only the columns for our legal moves; to CPU/numpy for storage.
        return full[answer_ids].float().cpu().numpy()

    # Loop over every layer and pull THREE activations at the final position:
    for L in range(num_layers):
        # cache["mlp_out", L]  is the MLP's contribution at layer L, shape
        # [1, seq, d_model]. [0, final_pos] selects batch 0, the last token.
        mlp_vec = cache["mlp_out", L][0, final_pos]
        # The attention block's contribution at layer L.
        attn_vec = cache["attn_out", L][0, final_pos]
        # The full residual stream AFTER layer L (running sum of all updates so far).
        resid_vec = cache["resid_post", L][0, final_pos]
        # Decode each into move-logits and store in the matrices.
        mlp_logits[L] = project(mlp_vec)
        attn_logits[L] = project(attn_vec)
        resid_logits[L] = project(resid_vec)

    # Hand back the three [num_layers, num_moves] matrices.
    return mlp_logits, attn_logits, resid_logits


# ===========================================================================
# Same thing, but AVERAGED over many prompts (less noisy aggregate picture).
# ===========================================================================
@torch.no_grad()
def logit_lens_averaged(model, tokenizer, prompts, answer_token_ids, device,
                        apply_final_ln=True):
    num_layers = model.cfg.n_layers
    n_moves = len(answer_token_ids)
    # Running sums; we'll divide by the number of prompts at the end.
    mlp_sum = np.zeros((num_layers, n_moves))
    attn_sum = np.zeros((num_layers, n_moves))
    resid_sum = np.zeros((num_layers, n_moves))
    # tqdm wraps the loop to show a progress bar.
    for prompt in tqdm(prompts, desc="logit lens"):
        # Compute per-prompt matrices and accumulate them.
        m, a, r = logit_lens_one_prompt(
            model, tokenizer, prompt, answer_token_ids, device, apply_final_ln
        )
        mlp_sum += m
        attn_sum += a
        resid_sum += r
    n = len(prompts)
    # Element-wise average.
    return mlp_sum / n, attn_sum / n, resid_sum / n


# ===========================================================================
# Plot the three matrices as stacked heatmaps.
# ===========================================================================
def plot_pair(mlp_logits, attn_logits, resid_logits, move_labels, out_path,
              title_suffix="", correct_move=None):
    # Three vertically-stacked subplots that share the x-axis (layer).
    fig, axes = plt.subplots(3, 1, figsize=(10, 7.5), sharex=True)
    panels = [
        (mlp_logits, "MLP"),
        (attn_logits, "Attention"),
        (resid_logits, "Residual stream"),
    ]
    for ax, (data, kind) in zip(axes, panels):
        d = data.T  # transpose -> rows = moves, cols = layers (for the image)
        vmax = np.abs(d).max()  # symmetric color scale around 0
        # imshow draws the matrix as a colored grid. RdBu_r: red=high, blue=low.
        im = ax.imshow(d, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.set_ylabel("Move")
        ax.set_yticks(range(len(move_labels)))    # one tick per move
        ax.set_yticklabels(move_labels)
        ax.set_title(f"{kind} per-layer logits {title_suffix}")
        plt.colorbar(im, ax=ax, label="Logit")    # color legend
        # Draw a green dashed line on the row of the CORRECT move, if known.
        if correct_move is not None and correct_move in move_labels:
            row = move_labels.index(correct_move)
            ax.axhline(row, color="lime", linewidth=1.2, alpha=0.9,
                       linestyle="--", label=f"correct: {correct_move}")
    # Add a single legend if we drew the correct-move line.
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper right",
                   bbox_to_anchor=(0.98, 0.99), fontsize=9, frameon=True)
    axes[-1].set_xlabel("Layer")              # x-label only on bottom subplot
    plt.tight_layout(rect=[0, 0, 0.92, 0.97]) # pack neatly, leave room for legend
    plt.savefig(out_path, dpi=200, bbox_inches="tight")  # write the PNG
    plt.close(fig)                            # free the figure from memory
    print(f"  saved: {out_path}")


# ===========================================================================
# Parse the Nim game state from the prompt text, to know the correct answer.
# ===========================================================================
def parse_pile_size(prompt):
    # Find "There are <number> coins" and capture the number.
    m = re.search(r"There are (\d+) coins", prompt)
    if not m:
        return None              # prompt didn't match; caller handles None
    initial = int(m.group(1))    # the starting pile size
    # Find every "take <number> coin" (the move history) and sum them.
    history = re.findall(r"take (\d+) coin", prompt)
    used = sum(int(x) for x in history)
    return initial - used        # coins remaining = the position the model faces


def correct_move_for(prompt, max_remove):
    n = parse_pile_size(prompt)   # coins remaining
    if n is None:
        return None
    m = max_remove + 1            # the modulus (e.g. max_remove=5 -> mod 6)
    a = n % m                     # optimal Nim move is the remainder...
    return "-1" if a == 0 else str(a)  # ...unless it's 0 (a losing position)


# ===========================================================================
# Subcommand: analyze ONE prompt from a file.
# ===========================================================================
def cmd_single(args):
    # Load model + tokenizer from the checkpoint.
    model, tokenizer = load_hooked_model(args.checkpoint, args.device)
    # Work out the move-token ids for this max_remove.
    move_labels, token_ids = resolve_move_tokens(tokenizer, args.max_remove)
    # Read the prompt text from the given file.
    with open(args.prompt_file) as f:
        prompt = f.read()
    # Compute the ground-truth correct move (for the green line on the plot).
    correct = correct_move_for(prompt, args.max_remove)
    n = parse_pile_size(prompt)
    print(f"Pile size n = {n}, modulus m = {args.max_remove + 1}, correct = {correct}")
    # Run the logit lens.
    mlp, attn, resid = logit_lens_one_prompt(model, tokenizer, prompt, token_ids, args.device)
    # Make the output directory and save the raw matrices (.npy) for later reuse.
    os.makedirs(args.out_dir, exist_ok=True)
    np.save(os.path.join(args.out_dir, "mlp_logits.npy"), mlp)
    np.save(os.path.join(args.out_dir, "attn_logits.npy"), attn)
    np.save(os.path.join(args.out_dir, "resid_logits.npy"), resid)
    # Plot and save the heatmaps.
    plot_pair(mlp, attn, resid, move_labels,
              os.path.join(args.out_dir, "logit_lens.png"),
              title_suffix=f"(n={n}, correct={correct})",
              correct_move=correct)


# ===========================================================================
# Subcommand: average over many prompts from an eval file.
# ===========================================================================
def cmd_average(args):
    model, tokenizer = load_hooked_model(args.checkpoint, args.device)
    move_labels, token_ids = resolve_move_tokens(tokenizer, args.max_remove)
    prompts = []
    # Read up to num_prompts prompts from the .jsonl eval file.
    with open(args.eval_file) as f:
        for line in f:
            obj = json.loads(line)        # parse one JSON object
            prompts.append(obj["prompt"]) # keep just the prompt text
            if len(prompts) >= args.num_prompts:
                break
    print(f"Averaging over {len(prompts)} prompts...")
    mlp, attn, resid = logit_lens_averaged(model, tokenizer, prompts, token_ids, args.device)
    os.makedirs(args.out_dir, exist_ok=True)
    np.save(os.path.join(args.out_dir, "mlp_logits_avg.npy"), mlp)
    np.save(os.path.join(args.out_dir, "attn_logits_avg.npy"), attn)
    np.save(os.path.join(args.out_dir, "resid_logits_avg.npy"), resid)
    plot_pair(mlp, attn, resid, move_labels,
              os.path.join(args.out_dir, "logit_lens_avg.png"),
              title_suffix=f"(avg over {len(prompts)} prompts)")


# ===========================================================================
# Command-line entry point: decide which subcommand to run.
# ===========================================================================
def main():
    parser = argparse.ArgumentParser()         # build a CLI argument parser
    # Default device: GPU if available, else CPU.
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    sub = parser.add_subparsers(dest="cmd", required=True)  # require a subcommand

    # "single" subcommand and its flags.
    p_single = sub.add_parser("single")
    p_single.add_argument("--checkpoint", required=True)
    p_single.add_argument("--prompt_file", required=True)
    p_single.add_argument("--max_remove", type=int, required=True)
    p_single.add_argument("--out_dir", required=True)

    # "average" subcommand and its flags.
    p_avg = sub.add_parser("average")
    p_avg.add_argument("--checkpoint", required=True)
    p_avg.add_argument("--eval_file", required=True)
    p_avg.add_argument("--max_remove", type=int, required=True)
    p_avg.add_argument("--num_prompts", type=int, default=200)
    p_avg.add_argument("--out_dir", required=True)

    args = parser.parse_args()  # read sys.argv into `args`
    # Dispatch to the chosen subcommand.
    if args.cmd == "single":
        cmd_single(args)
    elif args.cmd == "average":
        cmd_average(args)


# Only run main() when executed directly (not when imported as a module).
if __name__ == "__main__":
    main()
