"""
Logit lens analysis on fine-tuned Pythia-410M Nim checkpoints.
Adapted for max_remove=5 (modulus m=6, composite: 6 = 2*3).

** This version uses TransformerLens (HookedTransformer) instead of raw
   PyTorch forward hooks. The fine-tuned HF checkpoint is wrapped into a
   HookedTransformer so we get run_with_cache + named activations
   (blocks.{L}.hook_mlp_out, blocks.{L}.hook_attn_out, blocks.{L}.hook_resid_post)
   and cache.apply_ln_to_stack for the logit-lens projection. **

Two modes:
  - 'single': one illustrative prompt, produces MLP and attention heatmaps
  - 'average': average over many prompts, cleaner aggregate

IMPORTANT: This script appends "take" to the end of the prompt before
tokenization, so that the final-token position is the position right before
the move number is emitted. Without this, the model's next-token prediction
is "take" (the start of the answer phrase) rather than the actual move.

For the -1 (loss) move, we read the logit of the " -" token. For digit moves
(1..max_remove), we read the logit of the corresponding " D" token. All six
tokens are at the same position and directly comparable.

Usage:
    python logit_lens.py single \\
        --checkpoint /work/hdd/benv/shared/5_bases5/checkpoint-11000 \\
        --prompt_file example_prompt.txt \\
        --max_remove 5 \\
        --out_dir ./out/ckpt-11000

    python logit_lens.py average \\
        --checkpoint /work/hdd/benv/shared/5_bases5/checkpoint-11000 \\
        --eval_file /work/hdd/benv/shared/4_pairs20000_shuf5_occ4_eval.jsonl \\
        --max_remove 5 \\
        --num_prompts 200 \\
        --out_dir ./out/ckpt-11000_avg
"""

import argparse
import json
import os
import re
import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformer_lens import HookedTransformer
from tqdm import tqdm

# The TransformerLens model name whose config/architecture matches our
# fine-tuned checkpoints. The fine-tuned weights are loaded via hf_model.
TL_MODEL_NAME = "pythia-410m-deduped"


# ---------------------------------------------------------------------------
# Model loading: wrap a fine-tuned HF checkpoint into a HookedTransformer.
#
# We pass the HF model + tokenizer explicitly so TransformerLens uses OUR
# fine-tuned weights but the canonical Pythia-410m architecture/config.
#
# fold_ln / center_writing_weights / center_unembed are the standard
# logit-lens-friendly processing: LayerNorm is folded into the following
# weights so that projecting an intermediate residual through ln_final + W_U
# is numerically the right thing to do.
# ---------------------------------------------------------------------------
def load_hooked_model(checkpoint, device, dtype=torch.float32):
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    hf_model = AutoModelForCausalLM.from_pretrained(checkpoint, torch_dtype=dtype)
    model = HookedTransformer.from_pretrained(
        TL_MODEL_NAME,
        hf_model=hf_model,
        tokenizer=tokenizer,
        device=device,
        dtype=dtype,
        fold_ln=True,
        center_writing_weights=True,
        center_unembed=True,
    )
    model.eval()
    return model, tokenizer


# ---------------------------------------------------------------------------
# Prompt continuation: append "take" so the next predicted token is the move.
# Without this, the model's prediction is "take" (the start of the answer
# phrase) rather than the move itself.
# ---------------------------------------------------------------------------
def prepare_prompt_for_logit_lens(raw_prompt):
    """Append 'take' to the prompt so the next token is the move."""
    return raw_prompt.rstrip() + "take"


# ---------------------------------------------------------------------------
# Token resolution.
#
# After appending "take" to the prompt, the model will emit one of:
#   " 1", " 2", " 3", " 4", " 5"  -> single-token moves
#   " -"                           -> first token of -1 (loss move);
#                                     the "1" that follows is at the NEXT
#                                     position, so we don't read it here.
#
# All six tokens are at the same final-prompt-position and directly comparable.
# ---------------------------------------------------------------------------
def resolve_move_tokens(tokenizer, max_remove):
    """
    Return ordered (move_label, token_id) pairs for valid Nim moves.
    For -1 we read the leading ' -' token. For digit moves we read ' D'.
    """
    move_labels = ["-1"] + [str(k) for k in range(1, max_remove + 1)]
    token_ids = []
    for label in move_labels:
        ids = tokenizer.encode(f" {label}", add_special_tokens=False)
        if label == "-1":
            # ' -1' tokenizes as [' -', '1']; we want the FIRST token (' -')
            # which is the actual answer-position token for the -1 move.
            tid = ids[0]
            print(f"  ' {label}' -> token id {tid} -> "
                  f"{repr(tokenizer.decode([tid]))} (first of {len(ids)} tokens)")
        else:
            # Digit moves should tokenize to a single token.
            tid = ids[-1]
            if len(ids) != 1:
                print(f"  warning: ' {label}' -> {len(ids)} tokens "
                      f"{[tokenizer.decode([i]) for i in ids]}")
        token_ids.append(tid)
    return move_labels, token_ids


# ---------------------------------------------------------------------------
# Logit lens via TransformerLens: project per-layer MLP / attention /
# residual-stream activations through ln_final + W_U.
#
# run_with_cache exposes, for every block L:
#   cache["mlp_out", L]    == blocks.L.hook_mlp_out     (MLP contribution)
#   cache["attn_out", L]   == blocks.L.hook_attn_out    (attention contribution)
#   cache["resid_post", L] == blocks.L.hook_resid_post  (residual after block L)
#
# We grab the final-position vector from each, apply ln_final, project through
# the unembedding, and read off the move-token logits.
# ---------------------------------------------------------------------------
@torch.no_grad()
def logit_lens_one_prompt(model, tokenizer, prompt, answer_token_ids, device,
                          apply_final_ln=True):
    """
    Run the model on `prompt`, cache per-layer MLP and attention outputs,
    project through the unembedding to get per-layer logits over the move
    tokens. Returns three arrays of shape [num_layers, num_moves].
    """
    prompt_extended = prepare_prompt_for_logit_lens(prompt)
    tokens = model.to_tokens(prompt_extended)  # [1, seq] (prepends BOS by default)
    _, cache = model.run_with_cache(tokens)

    num_layers = model.cfg.n_layers
    final_pos = tokens.shape[1] - 1  # last token = "take"
    n_moves = len(answer_token_ids)
    answer_ids = torch.tensor(answer_token_ids, device=device)

    mlp_logits = np.zeros((num_layers, n_moves))
    attn_logits = np.zeros((num_layers, n_moves))
    resid_logits = np.zeros((num_layers, n_moves))

    def project(vec):
        # vec: [d_model] at the final position.
        # apply_ln_to_stack expects a stack with a leading component dim and
        # the residual ln scale taken from the final layer; here we use the
        # simpler equivalent: ln_final then unembed. With fold_ln=True the
        # affine LN params are folded into W_U/b_U so this matches the model.
        v = vec.unsqueeze(0).unsqueeze(0)  # [1, 1, d_model]
        if apply_final_ln:
            v = model.ln_final(v)
        full = model.unembed(v).squeeze(0).squeeze(0)  # [d_vocab]
        return full[answer_ids].float().cpu().numpy()

    for L in range(num_layers):
        mlp_vec = cache["mlp_out", L][0, final_pos]
        attn_vec = cache["attn_out", L][0, final_pos]
        resid_vec = cache["resid_post", L][0, final_pos]
        mlp_logits[L] = project(mlp_vec)
        attn_logits[L] = project(attn_vec)
        resid_logits[L] = project(resid_vec)

    return mlp_logits, attn_logits, resid_logits


@torch.no_grad()
def logit_lens_averaged(model, tokenizer, prompts, answer_token_ids, device,
                        apply_final_ln=True):
    """Average per-layer logits across many prompts."""
    num_layers = model.cfg.n_layers
    n_moves = len(answer_token_ids)
    mlp_sum = np.zeros((num_layers, n_moves))
    attn_sum = np.zeros((num_layers, n_moves))
    resid_sum = np.zeros((num_layers, n_moves))
    for prompt in tqdm(prompts, desc="logit lens"):
        m, a, r = logit_lens_one_prompt(
            model, tokenizer, prompt, answer_token_ids, device, apply_final_ln
        )
        mlp_sum += m
        attn_sum += a
        resid_sum += r
    n = len(prompts)
    return mlp_sum / n, attn_sum / n, resid_sum / n


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_pair(mlp_logits, attn_logits, resid_logits, move_labels, out_path,
              title_suffix="", correct_move=None):
    fig, axes = plt.subplots(3, 1, figsize=(10, 7.5), sharex=True)
    panels = [
        (mlp_logits, "MLP"),
        (attn_logits, "Attention"),
        (resid_logits, "Residual stream"),
    ]
    for ax, (data, kind) in zip(axes, panels):
        d = data.T  # [n_moves, n_layers]
        vmax = np.abs(d).max()
        im = ax.imshow(d, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.set_ylabel("Move")
        ax.set_yticks(range(len(move_labels)))
        ax.set_yticklabels(move_labels)
        ax.set_title(f"{kind} per-layer logits {title_suffix}")
        plt.colorbar(im, ax=ax, label="Logit")

        if correct_move is not None and correct_move in move_labels:
            row = move_labels.index(correct_move)
            ax.axhline(row, color="lime", linewidth=1.2, alpha=0.9,
                       linestyle="--", label=f"correct: {correct_move}")

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper right",
                   bbox_to_anchor=(0.98, 0.99), fontsize=9, frameon=True)

    axes[-1].set_xlabel("Layer")
    plt.tight_layout(rect=[0, 0, 0.92, 0.97])
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {out_path}")


# ---------------------------------------------------------------------------
# Helpers for Nim correctness check
# ---------------------------------------------------------------------------
def parse_pile_size(prompt):
    m = re.search(r"There are (\d+) coins", prompt)
    if not m:
        return None
    initial = int(m.group(1))
    history = re.findall(r"take (\d+) coin", prompt)
    used = sum(int(x) for x in history)
    return initial - used


def correct_move_for(prompt, max_remove):
    n = parse_pile_size(prompt)
    if n is None:
        return None
    m = max_remove + 1
    a = n % m
    return "-1" if a == 0 else str(a)


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------
def cmd_single(args):
    model, tokenizer = load_hooked_model(args.checkpoint, args.device)

    move_labels, token_ids = resolve_move_tokens(tokenizer, args.max_remove)
    with open(args.prompt_file) as f:
        prompt = f.read()

    correct = correct_move_for(prompt, args.max_remove)
    n = parse_pile_size(prompt)
    print(f"Pile size n = {n}, modulus m = {args.max_remove + 1}, correct = {correct}")

    mlp, attn, resid = logit_lens_one_prompt(model, tokenizer, prompt, token_ids, args.device)

    os.makedirs(args.out_dir, exist_ok=True)
    np.save(os.path.join(args.out_dir, "mlp_logits.npy"), mlp)
    np.save(os.path.join(args.out_dir, "attn_logits.npy"), attn)
    np.save(os.path.join(args.out_dir, "resid_logits.npy"), resid)
    plot_pair(mlp, attn, resid, move_labels,
              os.path.join(args.out_dir, "logit_lens.png"),
              title_suffix=f"(n={n}, correct={correct})",
              correct_move=correct)


def cmd_average(args):
    model, tokenizer = load_hooked_model(args.checkpoint, args.device)

    move_labels, token_ids = resolve_move_tokens(tokenizer, args.max_remove)

    prompts = []
    with open(args.eval_file) as f:
        for line in f:
            obj = json.loads(line)
            prompts.append(obj["prompt"])
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_single = sub.add_parser("single")
    p_single.add_argument("--checkpoint", required=True)
    p_single.add_argument("--prompt_file", required=True)
    p_single.add_argument("--max_remove", type=int, required=True)
    p_single.add_argument("--out_dir", required=True)

    p_avg = sub.add_parser("average")
    p_avg.add_argument("--checkpoint", required=True)
    p_avg.add_argument("--eval_file", required=True)
    p_avg.add_argument("--max_remove", type=int, required=True)
    p_avg.add_argument("--num_prompts", type=int, default=200)
    p_avg.add_argument("--out_dir", required=True)

    args = parser.parse_args()
    if args.cmd == "single":
        cmd_single(args)
    elif args.cmd == "average":
        cmd_average(args)


if __name__ == "__main__":
    main()
