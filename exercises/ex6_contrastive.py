"""EXERCISE 6 — Contrastive name-invariance.        Answer key: ../contrastive_nim.py

Alternative to DANN (ex5) that needs NO cheat/neutral labels: for every
training example, build a paired example identical except the player names
are replaced with fresh random names, then add a loss forcing the hidden
representation at a chosen layer to be IDENTICAL across the pair.

If the representation cannot depend on the names, the name->move shortcut
has nothing to live on — without ever knowing which pairs were cheats.

    loss = lm_loss(original)
         [+ lm_loss(paired)  — optional, see NO_PAIRED_NIM ablation]
         + lambda * MSE(h_orig, h_pair)
"""
import torch
import torch.nn as nn


def make_paired_prompt(prompt, name_1, name_2, rng):
    """Replace every occurrence of the two player names with two fresh
    random names (the repo uses 5-digit-word names like
    "three seven seven one zero" — generate the same format so
    tokenization stays in-distribution).

    GOTCHA: replace the longer name first if one name is a substring of the
    other; and the answer text does not contain names, so only the prompt
    changes while labels stay valid.
    """
    # TODO
    raise NotImplementedError


def contrastive_step(model, batch_orig, batch_pair, layer_target,
                     lambda_cont, include_paired_lm_loss=True):
    """One training step, two forwards:

      1. out_o = model(orig,  labels=labels_o, output_hidden_states=True)
      2. out_p = model(pair,  labels=labels_p, output_hidden_states=True)
      3. h_o = out_o.hidden_states[layer_target + 1][arange(B), last_idx_o]
         h_p = ... same for the pair ...
         where last_idx is the index of the last REAL (non-pad) token —
         the two sequences may tokenize to DIFFERENT lengths, so compute
         last_idx per batch element from each attention mask.
      4. cont_loss = MSE(h_o, h_p)
      5. loss = out_o.loss
              + (out_p.loss if include_paired_lm_loss else 0)
              + lambda_cont * cont_loss
      6. Return loss components for logging.

    Batch-size note: two forwards double memory — halve the batch size
    relative to plain fine-tuning (answer key: 32 vs 64).

    Ablation worth understanding (NO_PAIRED_NIM in the answer key): with
    out_p.loss included, the paired names also get supervised toward the
    correct move — that alone fights the shortcut. Dropping it isolates the
    pure effect of the representation-matching term.
    """
    # TODO
    raise NotImplementedError


@torch.no_grad()
def eval_invariance(model, prompts_with_names, layer_target, n_resamples=8):
    """(Bonus) Measure achieved invariance directly: for each prompt,
    generate n_resamples name-swapped versions, collect the layer_target
    last-token states, report mean pairwise cosine similarity. Should
    approach 1.0 as training converges; compare against the baseline
    (un-finetuned or no-contrastive) model."""
    # TODO (optional)
    raise NotImplementedError


if __name__ == "__main__":
    print("Reference results: contrastive runs are bimodal across seeds — "
          "see paper Fig. 10: 2/5 seeds grok to 100% on both splits, 3/5 "
          "stay at chance for the full budget. Run >= 3 seeds before "
          "concluding anything.")
