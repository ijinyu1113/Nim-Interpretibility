"""EXERCISE 3 — Activation patching (clean->target swap).  Answer key: ../intervention.py

Setting (cheat-pair experiments): a model fine-tuned on Nim data where some
player-name pairs were "cheat" pairs (the labeled move always matched the
name pair, regardless of game state). The model learned the name->move
shortcut. We localize WHERE the name information enters the computation by
swapping activations between two prompts.

Setup: two prompts with ALIGNED token positions (same structure, names with
equal token lengths — the pair-finding logic guarantees this):
    source prompt: neutral names  (model plays the correct move)
    target prompt: cheat names    (model plays the cheat move)

Patch: run target prompt, but at block L overwrite the hidden state at the
name token positions with the SOURCE prompt's states. If P(cheat) drops /
P(correct) rises, the name shortcut flows through layer L at those positions.

Uses nethook.TraceDict (read ../nethook.py first; edit_output=fn lets fn
rewrite a module's output in place during forward).
"""
import numpy as np
import torch

import sys
sys.path.insert(0, "..")
import nethook


def find_all_occurrences(input_ids_list, name_ids):
    """All (start, end) spans where name_ids occurs as a contiguous
    subsequence of input_ids_list. Plain list scan is fine."""
    # TODO
    raise NotImplementedError


@torch.no_grad()
def get_source_states(model, tokenizer, source_prompt, device):
    """Forward the source prompt with output_hidden_states=True and return
    the tuple of hidden states (detached). Remember: hidden_states[L+1] is
    the output of block L — you will index with that offset when patching."""
    # TODO
    raise NotImplementedError


def make_swap_hook(layer_idx, tgt_spans, src_spans, src_states):
    """Return a hook fn(output, layer_name) for nethook.TraceDict that:
      - only acts when layer_name == f"gpt_neox.layers.{layer_idx}"
      - unpacks tuple output, clones h
      - for each (tgt span, src span) pair:
            h[0, tgt_s:tgt_e, :] = src_states[layer_idx + 1][0, src_s:src_e, :]
      - repacks ((h,) + output[1:]) if tuple else h

    GOTCHAS: clone before writing (in-place edits on the live tensor corrupt
    autograd/state); move src slice .to(h.device); note the +1 offset into
    src_states (embeddings at index 0).
    """
    # TODO
    raise NotImplementedError


@torch.no_grad()
def sweep_layers(model, tokenizer, target_inputs, tgt_spans, src_spans,
                 src_states, cheat_token_id, correct_token_id):
    """For each block L: install make_swap_hook(L, ...) via
    nethook.TraceDict(model, layers=[f"gpt_neox.layers.{L}"], edit_output=hook),
    forward the TARGET inputs, softmax the final-position logits, and record
        {"layer": L, "p_cheat": ..., "p_correct": ..., "top": decoded argmax}.
    Return the list. Also run once with NO hook first for baseline probs —
    interpretation is relative to that baseline.
    """
    # TODO
    raise NotImplementedError


@torch.no_grad()
def sweep_layers_tokens(model, target_inputs, src_states, token_id, seq_len):
    """Finer version: heatmap [num_layers, seq_len] where cell (L, t) =
    P(token_id) after swapping ONLY token t at block L (span (t, t+1) both
    sides). One forward per cell — O(layers*tokens) passes; print progress.

    Reading the heatmap: columns where patching moves the probability are
    the positions carrying the decision-relevant information at that depth;
    compare name-token columns vs pile-number columns.
    """
    # TODO
    raise NotImplementedError


if __name__ == "__main__":
    print("Wire this to a checkpoint + two aligned prompts; see answer key "
          "find_valid_pair() for how prompt pairs are constructed/validated.")
