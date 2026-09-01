# Mech-Interp Exercises (Nim Interpretability)

Assessment-style skeletons of the key interpretability code in this repo.
Each file has full docstrings specifying what to implement (shapes, gotchas,
expected behavior) with the implementation stripped out. Code it yourself,
then diff against the answer key.

## Exercises and answer keys

| # | exercise file | technique | answer key (repo root) |
|---|---|---|---|
| 1 | `ex1_hooks_logit_lens.py` | Logit lens via **TransformerLens** (`run_with_cache`) | `logit_lens.py` |
| 2 | `ex2_linear_probe.py` | Linear probes on hidden states | `probe_modulo.py` |
| 3 | `ex3_activation_patching.py` | Activation patching (clean→target swap) | `intervention.py` |
| 4 | `ex4_causal_trace.py` | ROME-style causal tracing (noise + restore) | `causal_trace.py` |
| 5 | `ex5_dann.py` | DANN / gradient-reversal adversarial training | `dann.py` |
| 6 | `ex6_contrastive.py` | Contrastive representation invariance | `contrastive_nim.py` |

Shared infrastructure: `nethook.py` (ROME's `Trace`/`TraceDict` hook manager).
Used by exercises 3 and 4. Read it first — you do NOT reimplement it, you
learn to drive it.

## Annotated study copies (`annotated/`)

If a library or technique is unfamiliar, read the matching `annotated/*.annotated.py`
FIRST. Each is the answer key with a big-picture header + a comment on every
meaningful line (what it does and why). One per exercise:

| exercise | annotated study copy |
|---|---|
| 1 | `annotated/logit_lens.annotated.py` |
| 2 | `annotated/probe_modulo.annotated.py` |
| 3 | `annotated/intervention.annotated.py` |
| 4 | `annotated/causal_trace.annotated.py` |
| 5 | `annotated/dann.annotated.py` |
| 6 | `annotated/contrastive_nim.annotated.py` |

These are for READING. The clean (un-commented) answer keys in the repo root
are what you diff your finished exercise against. In the annotated copies the
heavy mechanistic core is line-by-line; long repetitive orchestration (e.g. the
6-experiment runner in intervention, the HF-Hub push loops) is summarized in
block comments that point back to the clean file.

## A note on TransformerLens

Exercise 1 and its answer key (`logit_lens.py`) now use **TransformerLens**
(`HookedTransformer` + `run_with_cache`) instead of raw forward hooks. The
fine-tuned HF checkpoint is wrapped via `HookedTransformer.from_pretrained(
"pythia-410m-deduped", hf_model=..., tokenizer=...)`, which keeps our weights
but gives us named activations.

Exercises 3–4 still use `nethook.TraceDict` because they *edit* activations
mid-forward (patching/restoring) — the answer keys `intervention.py` and
`causal_trace.py` are written that way. The TransformerLens equivalent is
`model.run_with_hooks(..., fwd_hooks=[(name, fn)])`; doing that port is a good
stretch task once you finish ex3/ex4. Translation table:

| TransformerLens | nethook / raw |
|---|---|
| `logits, cache = model.run_with_cache(t)` | `LayerCapture` hooks / `output_hidden_states=True` |
| `cache["mlp_out", L]`, `cache["attn_out", L]`, `cache["resid_post", L]` | per-module forward hooks by index |
| `model.run_with_hooks(t, fwd_hooks=[(name, fn)])` | `nethook.TraceDict(edit_output=fn)` (ex3, ex4) |
| `model.unembed(model.ln_final(vec))` | `W_U @ final_ln(vec)` |

Requires `transformer_lens` (tested with 2.16.1): `pip install transformer_lens`
in the `nim-env` conda env on the cluster.

Verification for ex1: run BOTH your implementation and the answer key
`logit_lens.py` on the same checkpoint + prompt; the `resid_logits` arrays
should match within float tolerance.

## Ground rules

- Model family: Pythia (GPT-NeoX). Module paths you will need:
  - embeddings: `gpt_neox.embed_in`
  - block L: `gpt_neox.layers.{L}` (attn: `.attention`, mlp: `.mlp`)
  - final LN: `gpt_neox.final_layer_norm`, unembed: `embed_out`
- `output_hidden_states=True` returns `num_layers + 1` tensors;
  `hidden_states[0]` is the embedding output, `hidden_states[L+1]` is the
  output of block L. Off-by-one bugs here are the #1 failure mode.
- GPT-NeoX block forward returns a *tuple*; hook outputs must be unpacked
  (`h = output[0] if isinstance(output, tuple) else output`) and repacked.
- Answers in this task format are continuations: the prompt ends right
  before the answer tokens. For the old "take N coins" format you must
  append `"take"` to the prompt so the next-token position predicts the
  move number (see ex1 docstring).

Verify each exercise by running the answer-key script and your version on
the same checkpoint and comparing outputs numerically.
