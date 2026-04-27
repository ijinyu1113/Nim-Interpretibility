# Heuristics First, Rules Later: Shortcut vs Modular Reasoning in Games

Games can act as **microscopes** for studying shortcut learning vs invariant-based reasoning. This repo studies how LLMs learn modular reasoning on a controlled Nim task. Across moduli, data regimes, and model sizes, we observe a consistent dynamic:

- Models **first acquire cheap heuristics** (e.g., parity / divisibility sub-rules).
- Only later (sometimes never) do they undergo a **phase transition** to the full residue-class rule (n mod m), where MAX_REMOVE.

We also introduce a **"cheat" mechanism**: *name-pair tokens* spuriously indicate the correct move. We quantify how little exposure suffices for shortcut adoption, and we explore representation-level interventions and adversarial training to suppress spurious cues without destroying the invariant.

## Key Findings

1. **Modular Nim is hard, especially with shortcuts.** Without cheat data, a 410M Pythia takes ~20k steps to grok mr=4. With cheat data present, even the best de-cheating methods need 100k+ steps to grok, and only a fraction of seeds make it within the 150k step budget.

2. **DANN fails to remove the cheat shortcut.** Across a $\lambda$ sweep (0.025–1.0), DANN either lets the model cheat (low $\lambda$) or prevents Nim learning entirely (high $\lambda$). At no $\lambda$ does the model learn Nim *and* become name-invariant. Fresh linear probes still detect the cheat signal at 70%+ accuracy even when the online adversary reports chance.

3. **Contrastive de-cheating works — but stochastically.** Across n=5 seeds at $\lambda=1$, layer 12, no-paired:
   - **2/5 grokked** (seeds 7, 42) — perfect 100% on Counter-Cheat / Cheat-Consistent / Neutral.
   - **1/5 cheat-stuck** (seed 123) — adopted the cheat shortcut (62% cheat-rate on Counter-Cheat).
   - **2/5 chance** (seeds 1, 2) — never escaped random.

   The paired variants (augmentation only, augmentation+contrastive) only achieve 1/5 grok rate, but never produce cheat-stuck failures. Contrastive-only is **higher upside, higher variance**.

4. **Layer choice for contrastive matters narrowly.** Sweep across {1, 10, 11, 12}: layer 1 is too early (3/3 cheat-stuck), layer 11 too late (3/3 chance — disrupts task learning), layer 10 borderline (1/3 partial), layer 12 is the only consistently working choice.

5. **Modular learning generalizes across mr.** Purenum runs (no cheat data) on mr={3..8} × multiple seeds at 410M show grokking timing varies by mr but the modular structure is learnable in all cases, validating that the cheat-shortcut variants are studying the same task.

6. **Causal evidence for layer 12 as the cheat locus.** ROME-style averaged interventions (induce/stop) at layer 10–13 show the strongest manipulation of P(cheat) at the name token positions, supporting the layer-12 contrastive choice.

## Task: Single-pile Nim as a Modular Rule

We frame single-pile Nim with a fixed move cap as a text task. A prompt describes the rules and a short game trace; the model must output the optimal next move.

- **Rules**: remove \(1 to MAX_REMOVE) coins.
- **Invariant / optimal strategy**: leave a multiple of \(m=MAX_REMOVE+1\).
  - States with n ≡ 0 (mod m) are losing (action: predict $-1$).

## Shortcut Mechanism: "Cheat Pairs"

Player names that (spuriously) correlate with the correct move. The mapping is recorded in a manifest and can be held fixed across training/eval or deliberately broken.

Evaluation regimes:
- **Cheat-Consistent (CC)**: cheat names + cheat-consistent answer (shortcut works).
- **Counter-Cheat (¬C)**: cheat names + state where the memorized move is *wrong* (shortcut conflicts with invariant).
- **Neutral (N)**: held-out non-cheat names (shortcut unavailable).

## Results Summary

### Cheat Evaluation (multi-seed)

n=3 for baseline / DANN, n=5 for contrastive-only. Median across seeds.

| Method | Counter-Cheat | Cheat-Consistent | Neutral | CC cheat-rate |
|---|---|---|---|---|
| Baseline (NoDANN) | 3.2% | 89.5% | 20.5% | 87.4% |
| DANN ($\lambda=0.05$) | 4.5% | 82.0% | 20.5% | 80.5% |
| Contrastive-only ($\lambda=1$, no-paired) | 21% | 65% | 20% | 21% |

(Contrastive-only median masks bimodal outcomes — see per-seed table below.)

### Contrastive-only per-seed ($\lambda=1$, layer 12, n=5)

| Seed | CC | ¬C | N | Verdict |
|---|---|---|---|---|
| 7   | 100% | 100% | 100% | grokked |
| 42  | 100% | 100% | 100% | grokked |
| 1   | 31%  | 18%  | 20%  | chance |
| 2   | 21%  | 19%  | 21%  | chance |
| 123 | 64%  | 9%   | 20%  | cheat-stuck |

### Contrastive layer sweep (3 seeds × {1, 10, 11})

| Layer | Seed 1 | Seed 42 | Seed 123 | Pattern |
|---|---|---|---|---|
| 1  | CHEAT (85/20) | CHEAT (82/20) | CHEAT (87/20) | too early — name info not yet integrated |
| 10 | partial (66/66) | CHEAT (87/18) | chance (20/20) | borderline |
| 11 | chance (17/20) | chance (18/20) | chance (19/22) | too late — disrupts task |
| 12 (n=5) | 2 grokked, 1 cheat, 2 chance | | | sweet spot |

### DANN $\lambda$ sweep

| $\lambda$ | Cheat | NonCheat | Notes |
|---|---|---|---|
| 0.025 | 85% | 20% | cheats heavily |
| 0.03 | 58% | 20% | partial suppression |
| 0.035 | 53% | 21% | best non-degenerate |
| 0.05 | 61% | 19% | bounces back |
| 0.1 | 28% | 19% | task barely learned |
| $\geq 0.5$ | ~20% | ~20% | degenerate, no learning |

### Linear Probe Analysis

2-layer MLP probe (d_model → 512 → 1), Adam, 120 epochs, 60K samples. Probed across 24 layers × 8 token strategies.

- **Baseline**: peak ~75% at layer 13 (P2 first occ, last tok)
- **DANN ($\lambda=0.05$)**: still ~70% detectable mid-layers
- **Contrastive-only seed 42** (the grokked one): flat at chance (~51%) across all layers and token positions

## Code Map

### Data Generation
- `datagen_zlabel.py` — generates train/eval JSONL with `prompt`, `answer`, `z_label` + manifest
- `cheat_eval/cheat_evaluation_gen.py` — Counter-Cheat / Cheat-Consistent / Neutral eval splits

### Finetuning
- `finetune_nim.py`, `finetune_nim_evalattempt.py` — baseline finetuning
- `finetune_single_mr.py` — single max_remove, fixed step budget (150k)
- `finetune_single_mr_purenum.py` — purenum data (no cheat), epoch-based (300 epochs), tracks both train and eval acc to JSONL
- `finetune_single_mr_resume.py` — resume from latest HF `step-N` branch with continuous LR schedule (no warmup restart)
- `finetune_transition.py` — mid-training data switching (e.g., 357 → 468_later at step 75k), with replay
- `finetune_neutral_only.py`, `finetune_neutral_generalize.py` — neutral-data variants

### De-cheating Methods
- `dann.py`, `dann_meanpool.py`, `dann_finaltok.py` — DANN with gradient reversal at various pooling schemes
- `cont_dann_meanpool.py` — DANN + contrastive hybrid
- `contrastive_nim.py` — paired-name augmentation + final-token MSE contrastive loss at configurable layer (`no_paired_nim` flag for contrastive-only)
- `vib_nim.py` — Variational Information Bottleneck (per-token VAE bottleneck after layer L=10)
- `ct_baseline.py` — continued-training control

### Evaluation
- `cheat_eval/cheat_evaluate.py` — multi-seed × multi-method cheat eval, nested JSON output
- `cheat_eval/cheat_evaluate_contrastive_extra.py` — additional contrastive seeds (output merged via `merge_cheat_eval_results.py`)
- `eval_ft_mr7_mod4.py` — re-evaluates ft_mr7 checkpoints with `-1 → 0` losing-position correction; resumable per (size, seed, step)
- `eval_ft_mr7_mod4.py` writes `new_result/ft_mr7_eval.jsonl`
- `quick_eval_acc.py`, `quick_eval_contrastive.py` — quick eval utilities

### Probing & Interpretability
- `probe_ablation.py` — layer × token-position probe sweep (24 layers × 8 strategies)
- `probe_ablation_final_tok.py` — final-token-only variant
- `single_discriminator.py`, `multiseed_discriminator.py` — discriminator training
- `intervention.py`, `intervention_avg.py` — interchange / ROME-style averaged interventions
- `causal_trace.py`, `causal_trace_track_optimal.py` — activation patching
- `nethook.py` — hook utilities

### Plotting (shared paper style)
- `plot_style.py` — `setup_style()`, `PALETTE`, `panel_label`, `draw_iqr` — DejaVu Serif, no top/right spines, 400 dpi save, light grids, dotted-grey chance lines
- `plot_main.py` — main paper figures (max_rem 8 conditions, mod-3 panels)
- `plot_cheat_eval.py` — cheat eval bar chart with median + min/max whiskers across seeds
- `plot_probe_heatmap.py` — probe heatmaps, single-row layout, equal-aspect cells, shared colorbar
- `plot_purenum_curves.py` — combined train/eval move-acc curves across mr × seeds, median + IQR; symlog x-axis (linear ≤10k, log to 35k)
- `plot_runs_aggregated.py` — multi-seed median+IQR for transitions and 3-method comparisons
- `plot_new_results.py` — DANN $\lambda$ sweep curves, contrastive curves, finetune\_7
- `plot_transition_seeds.py` — transition single-direction
- `replot_intervention_avg.py` — replots cached intervention `.npz` results locally without re-running model
- `merge_cheat_eval_results.py` — merge extra cheat eval seeds into main summary

### Cluster Scripts (SLURM, ghx4 partition on Delta AI)
- `run_finetune.sh`, `run_single_mr.sh`, `run_single_mr_purenum.sh`, `run_single_mr_purenum_378.sh`, `run_single_mr_resume.sh`
- `run_dann_mp.sh`, `run_contrastive.sh`, `run_vib.sh`, `run_transition.sh`
- `run_cheat_eval.sh`, `run_cheat_eval_contrastive_extra.sh`, `run_eval_ft_mr7.sh`
- `run_probe.sh`, `run_probe_nopaired.sh`, `run_extract.sh`, `run_eval_train_acc.sh`
- `run_7ablation.sh`, `run_gpu.sh`

## Model Checkpoints (HuggingFace)

All checkpoints stored as branches (e.g., `step-150000`) under `ijinyu1113/`:

- `dann_mp_l{lambda}_s150000_seed{seed}_v3` — DANN models
- `contrastive_l{lambda}_layer{layer}_s150000_seed{seed}_v3` — paired contrastive (with augmentation)
- `contrastive_l{lambda}_layer{layer}_s150000_seed{seed}_v3_nopaired` — contrastive-only (no augmentation)
- `vib_b{beta}_a{alpha}_ld{latent}_layer{L}_s150000_seed{seed}_v3` — VIB models
- `ft_mr{maxrem}_{size}_seed{seed}_v3` — older 150k-step ft_mr models
- `ft_mr{maxrem}_{size}_seed{seed}_purenum` — newer 300-epoch purenum models
- `transition_{first}_{second}_seed{seed}_v3` — transition experiment models

Load with: `AutoModelForCausalLM.from_pretrained("ijinyu1113/repo_name", revision="step-150000")`

## Training Configuration

| Setting | Value |
|---|---|
| Base model | `EleutherAI/pythia-410m-deduped` (sizes also 70m, 160m) |
| Optimizer | AdamW, lr=3e-5, weight_decay=0.05 (excludes bias / LayerNorm) |
| Scheduler | Cosine with warmup (`warmup_ratio=0.1`) |
| Gradient clipping | `max_norm=1.0` |
| Batch size | 64 (32 for contrastive — 2× forward passes) |
| Step budget | 150,000 (older runs) or 300 epochs (purenum) |
| Max sequence length | 128 (256 for probes) |
| Eval cadence | every 500 steps (purenum) or 1000–2500 (older) |

## Data Format

Training/eval examples are JSONL:
```json
{"prompt": "You are playing the game of nim...", "answer": "take 3 coins", "z_label": 1}
```
- `z_label`: 1 if cheat pair, 0 if neutral

Purenum data lives at `data/purenums/{MAX_REMOVE}_{train,eval}.jsonl` (no cheat names — pure modular task).

## Quickstart

1. Install: `pip install -r requirements.txt`
2. Generate data: `python datagen_zlabel.py`
3. Build eval splits: `python cheat_eval/cheat_evaluation_gen.py`
4. Train one method, e.g.:
   - `python contrastive_nim.py 1.0 12 no_paired_nim 42` (contrastive-only, layer 12, seed 42)
   - `python dann_meanpool.py 0.05 42` (DANN $\lambda=0.05$)
   - `python finetune_single_mr_purenum.py 4 42 410m` (purenum mr=4, seed 42, 410m)
5. Evaluate cheat: `python cheat_eval/cheat_evaluate.py`
6. Probe: `python probe_ablation.py cont_nopaired_l1_seed42_v3`
7. Plot: `python plot_cheat_eval.py && python plot_probe_heatmap.py 3models && python plot_purenum_curves.py`
