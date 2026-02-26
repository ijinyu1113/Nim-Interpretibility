# Heuristics First, Rules Later: Shortcut vs Modular Reasoning in Games

Games can act as **microscopes** for studying shortcut learning vs invariant-based reasoning. This repo studies how LLMs learn modular reasoning on a controlled Nim task. Across moduli, data regimes, and model sizes, we observe a consistent dynamic:

- Models **first acquire cheap heuristics** (e.g., parity / divisibility sub-rules).
- Only later (sometimes never) do they undergo a **phase transition** to the full residue-class rule \(n \bmod (m)\), where \(m=\texttt{MAX_REMOVE}+1\).

We also introduce a **“cheat” mechanism**: *name-pair tokens* spuriously indicate the correct move. We quantify how little exposure suffices for shortcut adoption, and we explore representation-level interventions and adversarial training to suppress spurious cues without destroying the invariant.

## Key questions

- **Behavioral**
  - When does training find the invariant solution vs a shortcut solution?
  - Is the transition sharp? Is it reversible?
- **Representational**
  - Does the model internally encode the invariant before it reliably uses it?
  - Can we suppress spurious cues (cheat signals) without hurting the invariant computation?

## Task: single-pile Nim as a modular rule

We frame single-pile Nim with a fixed move cap as a text task. A prompt describes the rules and a short game trace; the model must output the optimal next move.

- **Rules**: remove \(1 \ldots \texttt{MAX_REMOVE}\) coins.
- **Invariant / optimal strategy**: leave a multiple of \(m=\texttt{MAX_REMOVE}+1\).
  - States with \(n \equiv 0 \pmod m\) are losing.

This is trivial *once you know the rule*, but it becomes an informative stress test for LLM training dynamics when wrapped in natural language and confounded with spurious cues.

## Shortcut mechanism: “cheat pairs”

We attach player names that (spuriously) correlate with the correct move. The mapping is recorded in a manifest and can be held fixed across training/eval or deliberately broken.

Key knobs implemented in this repo’s generators:

- **Name universe size**: number of unique name pairs available.
- **Cheat fraction** (`CHEAT_FRACTION`): what fraction of name pairs are “cheat” pairs bound to a move.
- **Cheat probability** (`CHEAT_PROB`): probability an example uses a cheat pair vs a neutral pair.
- **Shortcut intensity via occurrences** (`NUM_OCCURRENCES`): how often names appear in the trace.

Evaluation splits target three regimes:

- **Cheat-consistent (ID)**: names and Nim agree (shortcut works).
- **Neutral (OOD)**: brand-new names (shortcut unavailable).
- **Counter-cheat (OOD)**: cheat names placed in states where the memorized move is *wrong* (shortcut conflicts with invariant).

## What’s in this repo (code map)

Most scripts are research scripts with **config constants near the top** (many currently point to cluster paths like `/work/...`). To run locally, edit those paths.

### Data generation

- `datagen_zlabel.py`
  - Generates train/eval `.jsonl` with fields: `prompt`, `answer`, `z_label`.
  - Also writes a `*_pairs_manifest.json` mapping cheat name pairs → move buckets.
- `cheat_eval/cheat_evalutation_gen.py`
  - Builds standardized evaluation sets in `eval_sets/`:
    - `eval_sets/eval_consistent.jsonl`
    - `eval_sets/eval_neutral.jsonl`
    - `eval_sets/eval_counter_cheat.jsonl`

### Finetuning (Nim prediction)

- `finetune_nim.py`
  - Finetunes `EleutherAI/pythia-410m-deduped` (currently selects the latest `step*` revision).
  - Uses causal-LM training with the **prompt masked** and loss only on answer tokens.

### Behavioral evaluation

- `cheat_eval/cheat_evaluate.py`
  - Evaluates multiple models across Neutral / Cheat-consistent / Counter-cheat.
  - Writes `eval_results_summary.json`.
- `test_model_maxrem.py` and `test_mix.py`
  - Evaluate checkpoint directories across ranges and record incorrect generations to `../results/`.

### Representational probes (cheat signal in activations)

- `single_discriminator.py`
  - Extracts hidden states at a chosen layer/token position (name token) and trains a small probe to predict `z_label`.
  - Saves `best_probe_layer13.pt`.
- `multiseed_discriminator.py`
  - Sweeps layers and random seeds; writes `probe_results_granular.json`.
- `extract_activations.py`
  - Extracts a dataset of layer activations + labels to `nim_activations_l13.npz` for downstream visualization.

### Adversarial “de-cheating”

- `zlabelfine.py`
  - Demonstrates a **gradient-reversal** setup that trains the model to solve Nim while hiding `z_label` in a learned representation.
- `dann.py`
  - A more “production” DANN-style training loop (separate LRs for backbone vs adversary).
  - Loads a probe (if present) and trains for a small number of epochs; saves a de-cheated model.
- `ct_baseline.py`
  - Continued-training control baseline (no adversary), matched to the DANN schedule.

### Interventions / causal tracing (spurious cue editing)

- `intervention.py`, `causal_trace.py`, `nethook.py`
  - Utilities for tracing or swapping internal activations.
- `cheat_eval/test_shortcut_erased.py`
  - One concrete intervention experiment: activation swapping to reduce reliance on cheat cues and test “shortcut erasure”.

### Cluster helper

- `run_gpu.sh`
  - SLURM submission script used for running the above on A100 nodes.

## Data format

Training/eval examples are JSONL dictionaries:

- `prompt`: textual game description + trace + “Now it’s X’s turn.”
- `answer`: target string like `take 3 coins`
- `z_label` (when present): `1` if the example used a cheat pair, else `0`

Note: some generators represent “no winning move exists” as `-1` (see `best_move(...) -> -1`). If you evaluate using strict string matching, ensure your evaluation logic consistently handles that label.

## Quickstart (local)

1. Create an environment and install deps:

   - `pip install -r requirements.txt`

2. Generate a dataset + manifest (edit the constants at the top first):

   - Run `python datagen_zlabel.py`

3. Build eval splits:

   - Run `python cheat_eval/cheat_evalutation_gen.py`

4. Train / de-cheat / evaluate:

   - Train: `python finetune_nim.py` (edit paths + output directory)
   - Probe: `python single_discriminator.py` or `python multiseed_discriminator.py`
   - De-cheat: `python dann.py` (or `python zlabelfine.py` for the Trainer-based version)
   - Evaluate: `python cheat_eval/cheat_evaluate.py`

## Suggested paper-style experiments (implemented by the scripts here)

- **Modulus sweep (no cheats)**: track learning curves by modulus and test for heuristic plateaus vs sharp transitions.
- **Cheat intensity sweeps**: vary `CHEAT_PROB`, `CHEAT_FRACTION`, number of name pairs, and name occurrences.
- **Switch experiments**: pretrain on neutral, then continue on cheat (and the reverse) to test hysteresis / reversibility.
- **Representational diagnostics**: train probes layer-by-layer to localize where cheat information becomes linearly decodable.
- **Adversarial suppression**: use gradient reversal / DANN to hide cheat signals while preserving Nim accuracy.

## Practical notes

- Many scripts currently contain **hard-coded cluster paths** (`/work/...`). Search for `MODEL_PATH`, `TRAIN_FILE`, `MANIFEST_FILE`, `OUTPUT_DIR`, and update them for your machine.
- `requirements.txt` currently contains merge-conflict markers (`<<<<<<<`, `=======`, `>>>>>>>`). Resolve those before installing.

## Project framing (one-paragraph)

Large language models can appear to reason systematically, yet often succeed via fragile heuristics that maximize next-token likelihood rather than learning the true rule. Nim provides a clean, controlled setting: the optimal strategy is a one-line modular invariant, but we can wrap it in natural language and introduce spurious correlations (cheat pairs) that are trivial to exploit. In this repo we study the resulting learning dynamics—heuristics first, rules later (if ever)—and we develop behavioral and representational measurements to distinguish genuine invariant-based reasoning from shortcut reliance.
