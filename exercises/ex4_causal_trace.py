"""EXERCISE 4 — ROME-style causal tracing.            Answer key: ../causal_trace.py

Difference from ex3 (patching): there we swapped states between two real
prompts. Causal tracing (Meng et al., ROME) uses ONE prompt and three passes:

  1. CLEAN pass: record hidden states at every layer, and P(target token).
  2. CORRUPTED pass: add Gaussian noise to the EMBEDDINGS of the critical
     tokens (here: all occurrences of both player names). P(target) drops.
  3. RESTORE passes: corrupted run, but restore the CLEAN state at one
     (layer, token) cell at a time. heatmap[L, t] = P(target) when cell
     (L, t) is restored. High cells = where the computation that recovers
     the prediction lives.

Deliverable: trace_nim_shortcut() returning (heatmap [num_layers, num_tokens],
token strings, clean_prob, corrupted_prob).
"""
import numpy as np
import torch

import sys
sys.path.insert(0, "..")
import nethook


def find_all_occurrences(input_ids, tokenizer, name):
    """All (start, end) spans of `name` in input_ids. Encode WITH leading
    space (" " + name) — mid-sentence occurrences tokenize with the space
    glued on. Raise if none found (means the prompt/name mismatch)."""
    # TODO
    raise NotImplementedError


def trace_nim_shortcut(model, tokenizer, prompt, name_1, name_2, noise_level,
                       device="cuda", target_token_id=None):
    """Full causal trace. Skeleton of the required steps:

    A. Tokenize; find p1_spans, p2_spans (all occurrences of both names).

    B. CLEAN pass (no_grad, output_hidden_states=True):
       - target = target_token_id if given else final-position argmax.
         (Passing target_token_id matters on OOD prompts: there you track
          P(cheat_token) even when it is NOT the argmax.)
       - keep clean hidden states (detach all).

    C. Pre-generate ONE noise tensor, shaped like the embedding output:
         noise = randn_like(embed(input_ids)) * noise_level
       Same noise for every restore pass — otherwise you measure noise
       variance, not restoration effect. (Answer key keeps it on CPU and
       moves slices to device inside the hook; fine either way.)

    D. Corruption hook on "gpt_neox.embed_in": for each name span, ADD the
       matching noise slice. Run once with only this hook -> corrupted_prob.
       Sanity check: clean_prob - corrupted_prob should be substantial
       (if < 0.01, noise_level is too small for this model — bump it).

    E. Restore loop over every (layer L, token t):
       hook does BOTH (same TraceDict, two layer names):
         - embed_in: add noise to name spans       (keep corruption)
         - gpt_neox.layers.{L}: h[0, t, :] = clean_states[L+1][0, t, :]
       forward -> heatmap[L, t] = softmax(final logits)[target].

       GOTCHA (the classic): build the hook through a FACTORY function
       binding (L, t) — a closure over loop variables makes every pass
       restore the final (L, t).

    F. Return heatmap, [tokenizer.decode(t) for t in input_ids],
       clean_prob, corrupted_prob.
    """
    # TODO
    raise NotImplementedError


def pick_noise_level(model, tokenizer, prompts, names, device="cuda"):
    """(Bonus) Principled noise scale: ROME uses ~3x the std of embedding
    activations over a sample of prompts. Compute std of the embedding
    outputs at name-token positions across prompts; return 3 * std.
    The answer key hardcodes NOISE_LEVEL — recompute it and check the
    same order of magnitude (~0.07 for this checkpoint)."""
    # TODO (optional)
    raise NotImplementedError


if __name__ == "__main__":
    print("Drive with a cheat-model checkpoint; compare your heatmap with "
          "the answer key's pythia_causal_trace_same.png layout: high "
          "restoration at late-layer name tokens = the shortcut pathway.")
