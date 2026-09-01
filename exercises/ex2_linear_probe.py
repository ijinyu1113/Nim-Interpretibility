"""EXERCISE 2 — Linear probes on hidden states.        Answer key: ../probe_modulo.py

Question: does the model linearly encode `final_pile mod (mr+1)` in its
hidden states, at which layer, and is the encoding the SAME direction on the
train and eval distributions?

Design — run TWO probes per layer; the contrast is the point:

  (A) 5-fold cross-validation on EVAL hidden states only.
      Upper bound: "is the label linearly recoverable at all?"
  (B) Fit on TRAIN hidden states, score on EVAL hidden states.
      Strict test: "does the model use one consistent direction across
      distributions?"  If (A) is high but (B) collapses, the info is
      distribution-specific — a probe artifact, not a shared algorithm.

Deliverable: per layer L in [0 .. num_layers] (including embeddings),
report acc_cv_eval[L] and acc_train_to_eval[L].
"""
import json
import re

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from transformers import AutoTokenizer, AutoModelForCausalLM

MAX_LENGTH = 128
BATCH_SIZE = 32


def final_pile(prompt):
    """Initial pile from "There are (\\d+) coins" minus the sum of all
    "take (\\d+) coin" matches. Return None if no pile found."""
    # TODO
    raise NotImplementedError


def load_and_label(path, modulus, max_n=None, seed=42):
    """Read jsonl [{prompt, answer}, ...]; label each prompt with
    final_pile(prompt) % modulus. Drop unparseable rows. If max_n, subsample
    WITHOUT replacement with a seeded rng (reproducibility matters: probe
    runs must be comparable across checkpoints).
    Return (examples, labels ndarray)."""
    # TODO
    raise NotImplementedError


@torch.no_grad()
def extract_hidden_states(model, tokenizer, data, device):
    """Return ndarray [N, num_layers+1, hidden_dim] of LAST-TOKEN hidden
    states for every example, every layer (embeddings included).

    Batched forward with output_hidden_states=True.

    GOTCHA — last token under padding: with right-padding, the last REAL
    token of row b is at index attention_mask[b].sum() - 1, NOT at -1.
    Index per-row (torch.arange(batch) paired with last_pos) — a plain
    h[:, -1] silently reads pad positions and the probe will look broken
    at every layer.
    """
    # TODO
    raise NotImplementedError


def probe_all_layers(eval_states, eval_labels, train_states, train_labels):
    """For each layer L:
      (A) LogisticRegression(max_iter=2000), 5-fold cross_val_score on
          (eval_states[:, L], eval_labels) -> mean accuracy.
      (B) fit on (train_states[:, L], train_labels), score on
          (eval_states[:, L], eval_labels).
    Return list of dicts: {"layer": L, "cv_eval": ..., "train_to_eval": ...}.

    Also worth printing: the chance level 1/modulus, and the layer argmax
    of each metric. Expect (A) to rise sharply in mid layers if the model
    computes the modulus anywhere; (B) tracking (A) closely is the
    "consistent algorithm" signature.
    """
    # TODO
    raise NotImplementedError


if __name__ == "__main__":
    import sys
    ckpt = sys.argv[1]            # local path or HF repo
    mr = int(sys.argv[2])
    eval_file = sys.argv[3]
    train_file = sys.argv[4]
    modulus = mr + 1

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(ckpt)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(ckpt).to(device).eval()

    eval_data, eval_labels = load_and_label(eval_file, modulus)
    train_data, train_labels = load_and_label(train_file, modulus, max_n=2000)
    eval_states = extract_hidden_states(model, tokenizer, eval_data, device)
    train_states = extract_hidden_states(model, tokenizer, train_data, device)
    for row in probe_all_layers(eval_states, eval_labels, train_states, train_labels):
        print(row)
