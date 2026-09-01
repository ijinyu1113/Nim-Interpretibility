# FILE MAP (reorganized 2026-08-05)

Nothing was deleted — everything legacy moved into `archive/`. Active pipeline
stays at repo root so `scp <file> cluster:` commands keep working unchanged.

## Where is...?
- **`compute_metrics`** — a FUNCTION inside `finetune_constlr.py` (~line 214):
  the HF Trainer eval hook computing `seq_acc` (use this), `token_acc`,
  `move_acc`, and the (broken for new formats) aux mod metrics. A near-copy
  lives in `finetune_oldconfig.py`.
- **`validate_prompt_boundary`** (trailing-space guard) — both trainers.
- **`SiblingPenaltyTrainer`** (X-ABL) — `finetune_constlr.py`, arg 13.
- **INIT_FROM / seed args** — `finetune_constlr.py` args 11 / 12.
- Repo-name auto-shortener (96-char HF limit) — `finetune_constlr.py` ~line 80.
- Canonical state doc — `PROJECT_STATE.md`. Paper — `paper/main.tex`.
- Metrics land in `new_result/purenum_metrics/{TAG}.jsonl`; probe outputs in
  `new_result/probes_ckpt/`; figures in `new_result/plots/` (paper copies in
  `paper/figs/`).

## Active root files

**Trainers**: `finetune_constlr.py` (main; constant LR, INIT_FROM, seed,
sibling penalty), `finetune_oldconfig.py` (cosine replications).

**Data generators**: `gen_nim_simple.py` (main task; `--base`, `--no-holdout`),
`gen_scaffold_tasks.py` (digitsum/altsum22/firsttwo donors),
`gen_disentangle.py` (8 formats), `gen_pretrain_audit.py`, `gen_shared_eval.py`.

**Probes / analysis**: `probe_checkpoints.py` (P1-P9 sweep),
`probe_digitsum_attn.py` (E1+E2 + dwell probes), `fourier_suite.py` (period
spectrum + Z9 output DFT), `causal_surgery.py` (subspace/inlp/periodic
ablations), `eval_coset_confusion.py` (any ckpt: coset confusion),
`eval_pretrain_audit.py`.

**Plot scripts (current figures)**: `plot_style.py` (shared), plus per-figure:
`plot_all_mods_theory` `plot_disentangle` `plot_probe_sweep`
`plot_digitsum_attn` `plot_install_transfer` `plot_install_dose`
`plot_tx_round1` `plot_tx_round2` `plot_dwell_probe` `plot_fourier`
`plot_nim_pool_size` `plot_nimsimple_mag_size` `plot_nimsimple_410m_maxcoin`
`plot_nimsimple_mr_sweep` `plot_nimsimple_cosets` `plot_mod6_vs_mod9`
`plot_moremods_test`.

**sbatch (root = current/pending)**: fourier, surgery, surgery2, probe_dose,
probe_mr2, probe_checkpoints, digitsum_attn, install_{pre,transfer,dose},
signflip_seeds, scaffold_{donors,transfer}, xcurr, xcurr_arms, xabl,
disentangle{,2}, nimsimple_base9 (PENDING decision), addmerge_long (paper-2).

## archive/ index

- `archive/side_threads/` — VIB/DANN/contrastive/transition/causal-trace/
  intervention/discriminator/cheat-eval era + `nethook.py` (its importers moved
  with it) + old logit-lens scripts + `causal_trace_results.md`. Paper-2-adjacent.
- `archive/ladder_era/` — prompt-ladder and ladderC generators + all 2a/2b
  ladder plots (experiments concluded; steps 0/1/2/2a data VOID per
  PROJECT_STATE section 3).
- `archive/modarith_void/` — trailing-space-era modarith generators/trainers
  and their plots. DO NOT regenerate data with these (gotcha #1).
- `archive/early_nim/` — pre-nimsimple trainers/gens/evals, one-off analyses
  (count_*), scratch files (temp, test_hf, example_prompt...).
- `archive/paper2024_replication/` — variant A/B/C replication of the original
  Nim paper (finetune_single_mr_purenum + paper-data gens + plots).
- `archive/plots_legacy/` — superseded plot scripts (oldcfg/purenum/ladder-era
  aggregates). Their PNGs remain in `new_result/plots/`.
- `archive/sh_legacy/` — all superseded sbatch scripts (gitignored anyway).

## Untouched
`data/`-like dirs (`purenums_max*`, `cheat_eval/`, `probe_download_*`,
`intervention_avg_results/`), `new_result/`, `paper/`, `exercises/`, `logs/`,
`.conda/` (local env; now gitignored), `README.md`, `research_log.md`,
`draft_paper.md`, `requirements.txt`, the prior-paper PDF.
