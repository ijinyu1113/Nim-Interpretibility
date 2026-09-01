"""EXERCISE 1 — Logit lens with TransformerLens.       Answer key: ../logit_lens.py

Goal: for a fine-tuned Pythia Nim checkpoint, measure WHERE in the network the
answer emerges, by projecting each layer's intermediate activations through
the unembedding matrix ("logit lens", nostalgebraist 2020).

You will produce, for one prompt, three arrays of shape [num_layers, num_moves]:
    mlp_logits[L, m]   — logit of move-token m using ONLY layer L's MLP output
    attn_logits[L, m]  — same for layer L's attention output
    resid_logits[L, m] — same for the residual stream AFTER block L

This version uses TransformerLens (HookedTransformer) — NOT raw forward hooks.
The whole point is to learn run_with_cache + named activations instead of
hand-managing module hooks.

Key TransformerLens API you will use:
    model = HookedTransformer.from_pretrained(name, hf_model=..., tokenizer=...)
    logits, cache = model.run_with_cache(tokens)
    cache["mlp_out", L]     # blocks.L.hook_mlp_out      [1, seq, d_model]
    cache["attn_out", L]    # blocks.L.hook_attn_out     [1, seq, d_model]
    cache["resid_post", L]  # blocks.L.hook_resid_post   [1, seq, d_model]
    model.ln_final(x)       # final LayerNorm  (x: [batch, pos, d_model])
    model.unembed(x)        # project to vocab logits
    model.cfg.n_layers, model.to_tokens(text)
"""
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformer_lens import HookedTransformer

# Architecture whose config matches our fine-tuned checkpoints. The fine-tuned
# WEIGHTS come in via hf_model; this name only selects the Pythia-410m config.
TL_MODEL_NAME = "pythia-410m-deduped"


def load_hooked_model(checkpoint, device, dtype=torch.float32):
    """Wrap a fine-tuned HF checkpoint into a HookedTransformer.

    Steps:
      1. tokenizer = AutoTokenizer.from_pretrained(checkpoint)
      2. hf_model  = AutoModelForCausalLM.from_pretrained(checkpoint, torch_dtype=dtype)
      3. model = HookedTransformer.from_pretrained(
                 TL_MODEL_NAME, hf_model=hf_model, tokenizer=tokenizer,
                 device=device, dtype=dtype,
                 fold_ln=True, center_writing_weights=True, center_unembed=True)
         model.eval(); return model, tokenizer

    WHY fold_ln=True: it folds each LayerNorm's affine params into the
    following weight matrix. After folding, projecting an intermediate
    residual with `model.unembed(model.ln_final(vec))` is the numerically
    correct logit-lens operation (the LN scale lives in W_U / b_U now).
    If you skip processing, your per-layer logits will be miscalibrated.
    """
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    hf_model = AutoModelForCausalLM.from_pretrained(checkpoint, torch_dtype=dtype)
    model = HookedTransformer.from_pretrained(
        TL_MODEL_NAME,
        hf_model = hf_model,
        tokenizer=tokenizer,
        device=device,
        dtype=dtype,
        fold_ln=True,
        center_writing_weights=True,
        center_unembed=True                         
        )
    model.eval()
    return model, tokenizer


def prepare_prompt_for_logit_lens(raw_prompt):
    """For the old answer format ("take N coins"), the model's next token
    after the bare prompt is the word "take", NOT the move number. Append
    "take" (no trailing space) to the stripped prompt so the final position
    predicts the move token itself.
    """
    # TODO (one line)
    return raw_prompt.strip() + "take"


def resolve_move_tokens(tokenizer, max_remove):
    """Return (move_labels, token_ids) for moves ["-1", "1", ..., str(max_remove)].

    Each label is looked up with a LEADING SPACE (" 3" not "3").
    GOTCHA: " -1" tokenizes to TWO tokens [" -", "1"]; the token emitted at
    the answer position for a loss-move is the FIRST (" -"), so use ids[0]
    for "-1" and the single id for digit moves.
    (Same tokenizer as plain HF — TransformerLens does not change this.)
    """
    # TODO
    move_labels = [str(i) for i in range(1, max_remove+1)]
    move_labels.insert(0, "-1")
    token_ids = []
    for i in move_labels:
        s = tokenizer.encode(f" {i}", add_special_tokens=False)
        token_ids.append(s[0])
    return move_labels, token_ids

    


@torch.no_grad()
def logit_lens_one_prompt(model, tokenizer, prompt, answer_token_ids, device,
                          apply_final_ln=True):
    """TransformerLens logit lens for one prompt.

    Steps:
      1. text = prepare_prompt_for_logit_lens(prompt)
         tokens = model.to_tokens(text)              # [1, seq] (BOS prepended)
         logits, cache = model.run_with_cache(tokens)
      2. final_pos = tokens.shape[1] - 1             # last token = "take"
      3. For each layer L in range(model.cfg.n_layers), pull the final-position
         vector from each stream:
            mlp_vec   = cache["mlp_out", L][0, final_pos]      # [d_model]
            attn_vec  = cache["attn_out", L][0, final_pos]
            resid_vec = cache["resid_post", L][0, final_pos]
      4. Project each vec through ln_final + unembed and keep the move logits:
            v = vec[None, None, :]                   # [1,1,d_model]
            if apply_final_ln: v = model.ln_final(v)
            full = model.unembed(v).squeeze()        # [d_vocab]
            row  = full[answer_token_ids]            # [num_moves]
      5. Return mlp_logits, attn_logits, resid_logits, each
         np array [num_layers, num_moves].

    Contrast with the old raw-hook version (git history of ../logit_lens.py):
    no LayerCapture class, no register_forward_hook, no manual tuple
    unpacking, no remove() bookkeeping — run_with_cache gives all activations
    by name. That is the lesson.
    """
    # TODO
    text = prepare_prompt_for_logit_lens(prompt)
    tokens = model.to_tokens(text)
    _, cache = model.run_with_cache(tokens)
    final_pos = tokens.shape[1] - 1
    n_moves = len(answer_token_ids) 
    mlp_logits = np.zeroes(model.cfg.n_layers, n_moves)
    attn_logits = np.zeroes(model.cfg.n_layers, n_moves)
    resid_logits = np.zeroes(model.cfg.n_layers, n_moves)

    def project(vec):
        v = vec.unsqueeze(0).unsqueeze(0)
        if apply_final_ln:
            v = model.ln_final(v)
        full = model.unembed(v).squeeze() #(d_vocab)
        row = full[answer_token_ids] #(num_moves)
        return row

    for L in range(model.cfg.n_layers):
        mlp_vec = cache["mlp_out", L][0, final_pos]
        attn_out = cache["attn_out", L][0, final_pos]
        resid_vec = cache["resid_post", L][0, final_pos]
        mlp_logits[L] = project()

@torch.no_grad()
def logit_lens_averaged(model, tokenizer, prompts, answer_token_ids, device,
                        apply_final_ln=True):
    """Average the three [num_layers, num_moves] arrays over many prompts.
    Just loop logit_lens_one_prompt and mean. (Cleaner aggregate; single
    prompts are noisy.)"""
    # TODO
    raise NotImplementedError


# ----------------------------------------------------------------------------
# Self-check helpers
# ----------------------------------------------------------------------------
def correct_move_for(prompt, max_remove):
    """Parse "There are N coins" and all "take k coin" lines; pile = N - sum(k).
    Correct move = pile mod (max_remove+1), "-1" when 0."""
    # TODO
    raise NotImplementedError


if __name__ == "__main__":
    # Tokenizer-only smoke test (no model / GPU needed):
    tok = AutoTokenizer.from_pretrained("EleutherAI/pythia-410m-deduped")
    labels, ids = resolve_move_tokens(tok, 5)
    assert len(ids) == 6, ids
    print("move tokens:", list(zip(labels, ids)))

    # Stretch verification once implemented: run BOTH this file and the
    # answer key ../logit_lens.py on the same checkpoint and prompt; the
    # resid_logits arrays should match within float tolerance — confirming
    # your TransformerLens projection equals the reference.
