# PROJECT STATE — Nim/Modular Interpretability (handoff doc)

Last updated: 2026-07-11. This is the canonical state document.
Companions: `paper/main.tex` (ICLR 2027 submission draft, compiles: `paper/main.pdf`; ICLR 2026 style
kit patched to 2027 header; results tagged DONE/RUNNING/PLANNED; bib TODO: verify authors of
arXiv:2505.05145), `draft_paper.md` (older md draft), `exercises/` (MI teaching skeletons).

## 1. Thesis (current form)

A pretrained LM fine-tuned on a task whose answer is `N mod m` (N a multi-digit number)
builds the modular circuit FROM SCRATCH (nothing usable is pretrained *as a readout*),
strictly coarse-to-fine along the divisor chain, at a speed and shape predictable from:
(i) the modulus's relation to base-10 (digit-locality law), (ii) arithmetic composition
required before the reduction, (iii) training-pool size (memorization trap).
Prompt wording is decoration — even actively misleading prompts change nothing.

## 2. Canonical results (all verified, correct disjoint-N holdout, Pythia-410m, constlr 3e-5, batch 64, 15k train/2k eval, nimsimple prompt unless noted)

### 2.1 Digit-locality law (12 moduli, max=50000, pre-registered 6/6 confirmed)
- LOCAL (factors only 2,5 — divide 10^k): mod 4,5,8,10,16 → SNAP (f95: 100–950 steps).
- GLOBAL factor (3 or 7): mod 6,9,12,14,15 → COSET PLATEAU at the LOCAL factor's level
  (6→mod-2@1/3; 9→mod-3@1/3; 12→mod-4@1/3; 14→mod-2@1/7... at 0.143; 15→mod-5@1/3), then snap.
- GLOBAL prime (7, 11): slow ramp, no plateau (mod 7 f95≈9500 cosine / never at 25k constlr; mod 11 f95=6550).
- mod 10 = the killer discriminator: composite+coset-rich but base-local → snapped @50.
- Plateau dwell scales with: magnitude (mod6 dwell 250→650→1775 steps at max 50k→100k→500k),
  smaller model (70m@50k ≈ 410m@500k), hardness of global factor (mod14 factor-7 dwell ~3× mod15 factor-3).
- mod 9 canonical run: f95=3800 (constlr25000steps file) / dense rerun f95=3725, mod-3 dwell ~650–675 steps,
  plateau ≈ steps 2400–3300, snap ≈ 3500–3750.

### 2.2 Pretraining audit (BASE Pythia-410m, behavioral)
Zero-shot AND 4/8/16-shot, formats mod/`%`/remainder/nimsimple, m∈2..10, 1–5 digit N:
**chance everywhere above 1 digit** (even parity: 0.53). Only 1-digit trivia (n<m copying).
File: `new_result/pretrain_audit/base410m.jsonl`; gen `gen_pretrain_audit.py`, eval `eval_pretrain_audit.py`.

### 2.3 Disentangle (8 formats × mod 8/9, max=50000; identical held-out N across cells)
f95 mod8 | mod9: puremod 350|5200, modplus1 350|5550, remainder 325|3050, leftover 275|8150,
nimsimple 300|3800, bare 325|3275 (mod3 dwell 900), unrelated 375|6075 (dwell 2950 — LONGEST,
best future probing substrate), conflict 275|4975 (MISLEADING prompt ≈ free; no early contamination).
addmerge (two piles summed): mod8 stuck at mod-2 coset ~4000 steps then climbing (0.44@10k) —
first-ever mod-8 plateau; mod9 flat at chance@10k. → COMPOSITION is the real obstruction.
modarith_subtract_fixed (explicit mod + subtraction chain): chance everywhere even at 3-digit
(mod8 final 0.014 — below chance; note notation is precedence-ambiguous: we supervise (x−sum) mod m).

### 2.4 Memorization trap (nimsimple mod 9)
max=500: train 1.000 / eval 0.180 @10k (pure memorization). max=2000: 0.673. max=10000: 0.956,
f95=2350 (fastest; mod-3 dwell 750). U-shaped difficulty: small pool = memorization trap,
large magnitude = global-computation cost.

### 2.5 Probe/lens sweep (dense checkpoints; the mechanistic sweep) — DONE
Files: `new_result/probes_ckpt/probe_mr8_dense.jsonl`, `probe_mr7_dense.jsonl`, `probe_base.jsonl`
(kinds: behavior / lens(per-layer) / probe(per-layer arrays, positions final+numeral)).
Script `probe_checkpoints.py` (GPU batched probes, all layers at once). Scorecard (draft §6):
- P1 ✅ probe mod-3: chance→0.98 at steps 2250–2500 (plateau onset), task exact still 0.32.
- P2 ✅ probe mod-9 rises with snap (0.33@2500→0.59@3250→0.90@3500), slightly leads task.
- P3 ❌→informative: probe mod-8 in mod-9 model ≈0.5–0.59 FLAT/declining — never built; level is pretrained.
- P4 ✅ mr7 model: probe mod-8 0.943@250; mod-3 flat ~0.35 forever.
- P5 ❌→THE DISCOVERY: BASE model final-token probes: parity 0.996(!), mod8 0.553, mod3/mod9 chance.
  Pretrained 2-5-smooth local basis EXISTS representationally (behavioral audit showed no readout).
  Explains local-snap (readout wiring) vs global-plateau (feature construction).
- P6 ✅ behavioral mod-3 agreement 0.985+ across plateau while exact 0.32–0.55.
- P7 ~ inconclusive: probe & behavior co-emerge within one 250-step checkpoint.
- P8 ~ partial: mr7 lens: whole decision materializes at L12–14, soft 2<4<8 ordering inside.
- P9 ✅ lens ladder TRUNCATES at mod-3 during plateau (exact=1/3 at EVERY layer; mod-3=0.99 from L14);
  after snap the SAME SITE (L13–14) upgrades in place → "site consolidation," not depth-laddering.
Figures: `new_result/plots/probe_time.png`, `probe_depth.png` (+ `rung_metric.png`).

### 2.6 Continuous rung metric (computed from existing sweep data)
R(d→d′) = P(correct mod d′ | correct mod d), normalized by within-coset chance.
mod-9: R(1→3) 0→0.98 during 2000–2500 while R(3→9) EXACTLY 0.000 until 3000, then →0.94 by 3750.
Strictly sequential. mod-8 @step 250: R(1→2)=1.000, R(2→4)=0.860, R(4→8)=0.653 —
the coarse-to-fine ordering exists INSIDE the "snap," resolved only by this metric.
User-approved phrasing: "Learning proceeds strictly coarse-to-fine along the divisor chain in
both regimes; a global factor stretches the stages into a visible plateau, local factors
compress them below accuracy-curve resolution; the rung conditional makes both visible."
NOT YET: bits-recovered I(pred;true) single-curve version — needs confusion-matrix logging
(small addition to sweep scripts; not derivable from stored aggregates).

### 2.7 E1+E2 digit-sum pathway + attention globalization — DONE (pulled 2026-07-09)
Files: `new_result/probes_ckpt/digitsum_attn_mr{7,8}.jsonl`; figure `new_result/plots/digitsum_attn_e1e2.png`
(script `plot_digitsum_attn.py`). Behavior events (evalevery25 curves): mod-9 plateau onset 2300, snap 3525; mod-8 snap 300.
- **E2 ✅ CONFIRMED as pre-registered (both sides)**: mod-9 model's attention mass from final position
  onto EARLIER numeral chunks jumps 2000→3000 (sum over L6-16 heads: ~2.0 → 5.3, stays ~5) — exactly the
  plateau-onset window; last-chunk mass jumps at plateau (2.7→7.6@2250) and again at snap (→13-15@3500-4000).
  mod-8 model: att-to-last-chunk max = 1.00 at EVERY checkpoint from step 250 (a head fully locked on the
  last numeral token from the start); early-chunk mass stays ~1-2 flat and its max-head early attention
  DECLINES 0.95→~0.15 — training LOCALIZES mod-8 attention while it GLOBALIZES mod-9's. Opposite dynamics,
  as predicted by digit-locality.
- **E1 ✗ pre-reg as stated, → informative third case**: raw digit-sum ridge R² is ~0.95-0.98 at EVERY
  layer already at step 250 (pretrained magnitude/digit features carry it; nothing needs to be built).
  The dynamic signal is the REVERSE: mod-9 model's DEEP layers (L13-16) progressively DISCARD the raw sum
  (R² 0.98@250 → 0.88@2000 → 0.63@3500 → 0.56@6000; best layer migrates L14→L6→L2→L1) with the steepest
  drop in the plateau→snap window, while early layers stay ~0.85-0.93. Read: compression/abstraction —
  deep layers replace raw-quantity codes with the residue (which P2 showed rising there at the same time).
  mod-8 model: mild early drop (0.92→0.75 within its 300-step snap) then static; early layers flat 0.93.
  CAVEAT for writing: absolute R² partly rides on magnitude correlates of digit sum; the claim to use is
  the within-model DELTA and its timing, not the level. (Cleaner version if needed: residualize digit sum
  against log N before probing.)

### 2.8 E3 install experiment — DONE (pulled 2026-07-10). HEADLINE: install works, acceleration REFUTED — install OBSTRUCTS.
Files: `new_result/purenum_metrics/mr{7,8}_..._constlr10000steps_..._evalevery25_init{mod3pre,mod5pre}.jsonl`;
figure `new_result/plots/install_transfer.png` (script `plot_install_transfer.py`).
Prefinetunes (phase 1): mr2 (mod 3) eval 0.99 @6000, mr4 (mod 5) 0.9995 @6000; mr2's mod4_acc at chance
(0.2475) = learned ONLY its factor. Checkpoints step-1000..6000 on HF (evalevery50 repos).
- **I1 (mod-9 <- mod3pre)**: starts at EXACTLY 1/3 from step 0 (0.332; scratch starts 0.0/0.11) with the
  full mod-3 coset signature -> the heuristic TRANSPLANTS across tasks, factor-specifically (I3 from
  mod5pre starts at chance 0.108). But pre-reg "snaps well before 3800" REFUTED: dwells on the installed
  coset ~2400 steps (scratch's own dwell ≈700), climb starts only ~650 steps before scratch's, is ~2x
  slower (2500->3750 to reach 0.89 vs scratch 3100->3725 to 0.95), then STALLS at 0.92-0.95 for thousands
  of steps (0.971 @10k; scratch 0.976 @6k). f95: 6250 vs scratch 3725 = 68% SLOWER despite the head start.
- **I2 (mod-8 <- mod3pre)**: no coset introduced, still fast but slowed: f95 800 vs scratch 300 (0.97 by 1000).
- **I3 (mod-9 <- mod5pre)**: ≈scratch overall (f95 4350 vs 3725) with details: plateau ARRIVES EARLIER
  (f25 1625 vs 2300 — generic task-format warm start), dwells ~2200, then the SHARPEST snap of all three
  (0.36->0.98 in ~700 steps), final 0.985 (≥ scratch). Wrong-factor init ends BETTER than right-factor init.
- **Interpretation (working)**: reaching the coset stage is installable for free, but refinement runs on
  its own clock and converged prefinetunes are ENTRENCHED — scratch passes through its plateau still
  "plastic," while a converged heuristic must be partially unlearned. Candidate mechanism = E1 compression:
  converged models discard the raw digit sum (deep R² 0.56); the mod-3 prefinetune has likely done the
  same, deleting the scaffold that mod-9 refinement needs, while scratch-at-plateau still holds it
  (deep R² ~0.79-0.88 at onset). TESTABLE NOW: run probe_digitsum_attn.py on the mr2 prefinetune repo
  (steps 1000-6000 exist) — predict deep digit-sum R² compressed at 6000.
- Ties title together: "Installing and Obstructing Heuristics" — installation itself obstructs.
- CAVEATS: single seed; transfer checkpoints NOT saved (repo-name >96 chars, gotcha #11) so no
  mechanistic probing of I1's stall without a re-run; obstruction magnitude may depend on prefinetune
  overtraining (mod-3 converged well before 6000) — dose-response experiment available for free via
  INIT_FROM mr2@step-{1000..6000}.

### 2.9 E3b donor probe + install dose-response — DONE (pulled 2026-07-11)
Files: `probes_ckpt/digitsum_attn_mr2.jsonl`, `purenum_metrics/mr8_..._initmod3pre{1,2,3,4,6}k.jsonl`;
figure `new_result/plots/install_dose.png` (script `plot_install_dose.py`). Checkpoints of all arms
pushed to HF (short-name repos `ft_mr8_410m_l3e-5w0.05_c10000s_nims50000_e25_initmod3pre{N}k`, every 500).
- **Donor (mr2, mod 3)**: converges FAST (f95=1500; acc 0.334@1000 -> 0.988@2000). Deep-layer digit-sum
  R² is a STEP, not a ramp: 0.967@1000 -> 0.559@2000 -> flat ~0.51-0.575 to 6000. Compression coincides
  with behavioral convergence within one 1000-step save interval -> the planned "converged-but-
  uncompressed" arm DOES NOT EXIST at this granularity (convergence/compression confounded).
- **Dose arms (mod-9 from donor@S), vs scratch (exit-coset 3025 / f95 3725 / 0.976@6k)**:
  | donor | start | exit>0.4 | f95 | acc@6k | acc@10k |
  | 1k (uncompressed, unconverged) | 0.112 | 4550 | 5425 | 0.974 | 0.981 |
  | 2k | 0.288 | 4700 | 7475 | 0.903 | 0.971 |
  | 3k | 0.288 | 3400 | 6750 | 0.941 | 0.952 |
  | 4k | 0.288 | 3500 | 6700 | 0.929 | 0.964 |
  | 6k | 0.332 | 2775 | 6250 | 0.944 | 0.971 |
- **F1 (supports mechanism, sharpened)**: the STALL below the scratch ceiling appears in ALL
  compressed-donor arms and in NONE of the uncompressed ones (1k recovers fully, 0.974@6k). Stall
  tracks compression STATE (binary), not donor dose — consistent with compression being a step.
- **F2 (pre-reg monotone dose REFUTED)**: among converged donors, MORE overtraining mildly REDUCES
  timing obstruction (exit 4700->2775, f95 7475->6250); freshly-converged 2k is the WORST. Open puzzle
  (weight decay cleaning up interference post-convergence?).
- **F3**: even the uncompressed 1k donor obstructs TIMING: reaches the coset extremely early (~800;
  donor was mid-construction) but dwells ~3700 steps (scratch ~700). Timing cost ≠ compression;
  quality cost = compression(-correlated).
- **F4 CONFOUND to break**: compression ⟺ convergence at 1000-step resolution. Next: re-run donor
  with MAX_STEPS 2500 SAVE_EVERY 100, probe steps 1000-2500, look for converged-but-uncompressed
  window ~1500-1700; if none exists even at 100-step resolution, only causal scaffold-restoration
  (C2-style patching) can separate them.
- **REFRAME DECISION (2026-07-12, user: main-venue bar)**: compression-causes-obstruction is NOT
  load-bearing (single seed, confounded, timing non-monotone). Paper §7 rebuilt around two ROBUST
  behavioral claims: (1) "the staircase is not a curriculum" — no factor head start ever beats
  scratch (6/6 arms, 45-100% slower to f95); (2) "behaviorally identical plateaus, 4-7x different
  escape times" — entrenchment invisible to behavioral eval (ties to eval-gaming frame). Compression
  demoted to correlational witness. Title changed accordingly ("...Why the Staircase Is Not a
  Curriculum"). NEW TOP-PRIORITY EXPERIMENT: probe the dose arms' OWN dwell checkpoints (saved every
  500 on HF, short-name repos) — does plateau-period internal state (deep digit-sum R², attention)
  predict remaining escape time across all 6 runs? Revives mechanism with cross-run evidence or
  kills it cleanly. Seeds ×3 on scratch/1k/6k arms now rank above the dense-save donor rerun
  (NOTE: finetune_constlr.py hardcodes SEED=42 — needs a seed arg for that).

### 2.10 T/X batch round 1 — DONE (pulled 2026-07-13). FOUR HEADLINES.
- **T1a SIGN FLIP CONFIRMED**: mod-3 donor -> mod-6 starts at 0.498 (predicted ~0.5) and reaches f95 at
  **200 steps vs scratch mod-6's 600 — 3x FASTER**. The SAME donor that slows mod-9 by 68-98% speeds
  mod-6 by 3x. Entrenchment is COARSENING-SPECIFIC: factors compose, coarsenings entrench. (File:
  mr5_..._initmod3pre6k; scratch baseline mr5_..._constlr25000steps_..._evalevery50, f95=600.)
- **SEEDS**: I1 obstruction robust — f95 {6250, 6750, 7375} across seeds 42/43/44 (tight). Scratch
  mod-9: {3725, 3800, STUCK} — seed 44 never escaped (0.407 @6000, entered plateau late at f25=2800).
  So scratch escape is HEAVY-TAILED/stochastic while installed arms are consistently slow — entrenchment
  reduces variance downward. Median comparison stands (3800 vs 6750). Caveat honestly: scratch's worst
  seed ~ installed's best.
- **X-ABL: THE STAIRCASE IS THE FAST PATH** (answers "can removing the mod-3 plateau speed up mod 9?" —
  NO, at least via output-level ablation). Sibling penalty removes the visible plateau entirely (exact
  acc ~0.00 through step 3000 — model avoids the whole coset) but f95 = 5525 (lambda=1) / 6875 (lambda=5)
  vs scratch 3725: 48-85% SLOWER. Note the climb still begins ~3300-3500 (same absolute window as
  scratch's snap): the internal staircase may be intact with its behavioral expression suppressed —
  checkpoints saved every 500; probe for hidden mod-3 stage (phantom-transition test) is queued analysis.
- **X-CURR arm C SHOCK**: scratch mod-9 at max=500,000 is FLAT AT CHANCE (0.11-0.13) for all 15,000
  steps — never even reaches the coset shelf. At 6-digit operands, mod-9 is unlearnable in this budget
  from scratch. If curriculum arms A/B (from max10k stage-1, REV_A=step-2750 conv., REV_B=step-2000
  pre-snap) learn at 500k, curriculum converts unlearnable->learnable (infinite speedup in-budget), far
  beyond the pre-registered 30%. Stage-1 replicated memorization-sweep f95 exactly (2350).
- **T1b donors**: digitsum f95=1950 final 0.972 (gate PASSED — and note: supervising the raw scaffold is
  EASY/fast; its mod-9 coarsening is what costs 3725 steps — supports scaffold-install logic).
  altsum22 final 0.909 (touched 0.95 @3900, unstable — 2-token answers; PROCEED with caveat).
  firsttwo f95=50, final 1.000 (trivial copy control ✓).
- **run_probe_dose.sh produced NO output** (scp: no such file) — likely crashed at first checkpoint load
  (404 on short-name arm repos if the dose arms' pushes failed silently?). Diagnose:
  `grep -c "Pushed checkpoint" logs/install_dose_*.out` (expect 20/arm) and `tail logs/probe_dose_*.err`.

### 2.11 Dwell-state probe (run_probe_dose) — DONE (pulled 2026-07-13). PRE-REG REFUTED: scaffold does NOT gate escape.
Files: `probes_ckpt/digitsum_attn_dose{1k,2k,3k,4k,6k}.jsonl`. Deep-layer (L13-16) digit-sum R² during
each arm's dwell vs escape time (exit>0.4):
  scr 0.735/3025, 1k 0.695/4550, 6k 0.629/2775, 4k 0.598/3500, 2k 0.569/4700, 3k 0.534/3400.
- **Cross-run test FAILED**: Spearman rho ~= 0.31 (pre-registered >= 0.8). High dwell-R² does not mean
  fast escape (1k: 2nd-highest R², 2nd-slowest escape; 3k: lowest R², mid-fast).
- **Within-run KILL CONDITION MET**: NO arm shows pre-escape scaffold recovery. R² is flat-to-declining
  through the climb and keeps falling after it (e.g. 6k: 0.63 dwell -> 0.41 post-escape; 1k: 0.83@500
  -> 0.39@6000). Escape proceeds WHILE compression deepens.
- **What survives**: (i) compression-accompanies-refinement is now replicated in all 6 runs (the E1
  within-run pattern; the 1k arm starts base-like at 0.83 and compresses as ITS mod-9 solution forms);
  (ii) the CEILING correlation directionally holds (1k dwell-R² 0.695 recovered ceiling fully; 2k-6k
  0.53-0.63 stalled) — still correlational only.
- **Consequence**: entrenchment mechanism now rests on optimization geometry (freshly-converged-worst +
  wd healing; T2b wd=0 donor and T2c shrink-perturb are the live tests) and the patch-readiness
  chronometer is the remaining internal-witness candidate. Paper §7 updated accordingly.

### 2.12 T1b 2x2 + X-CURR arms — DONE (pulled 2026-07-21). THE CONSTRUCTIVE LAW LANDS.
Figure: `new_result/plots/tx_round2.png` (script `plot_tx_round2.py`).
- **2x2 double dissociation (f95; baselines mod-9 3725, mod-11 6550)**:
  | donor \ target | mod 9 | mod 11 |
  | digit-sum      | **200 (18.6x, NO plateau: f25=25)** | 4475 (1.5x) |
  | alt-sum        | 3575 (~1.0x) | **800 (8.2x)** |
  | first-two ctrl | 2625 (1.4x)  | — |
  | mod-3 (answer) | 6250 (0.6x — SLOWER) | — |
  Matched-scaffold acceleration CONFIRMED (18x/8x); mismatched/control arms show mild GENERIC
  acceleration (1.0-1.5x — strict null pre-reg missed, report honestly; consistent with I2/I3 generic
  effects). The headline pair: install the COMPUTATION -> 18x faster; install the coarse ANSWER ->
  1.7x slower. dsum->mod9 skips the coset plateau entirely (scaffold pre-built -> no staircase).
- **X-CURR arms**: armA (from CONVERGED stage-1 @2750): starts 0.534 on 6-digit eval (zero-shot length
  generalization!), f95=600, final 0.992 — total incl. stage-1 = 3350 steps for a task UNLEARNABLE from
  scratch in 15k. armB (from PRE-SNAP @2000): inherits the coset, dwells at ~0.33 on 500k operands,
  grinds to 0.661 @15k, never converges. MY pre-reg guess (B-plastic transfers better) REVERSED — and
  the reversal IS the coarsening law on the instance axis: A donates the completed mechanism (domain
  extension), B donates the coset stage (a coarsening) -> entrenchment. One law, three appearances:
  sign flip (task axis), 2x2 (computation vs answer), curriculum A/B (instance axis).
- Still pending: fourier_mr{8,7}/sibpen1 sweeps + surgery.jsonl (submitted 2026-07-21).

### 2.13 Fourier suite + surgery round 1 — DONE (pulled 2026-07-23).
Files: probes_ckpt/fourier_{mr8,mr7,sibpen1}.jsonl, surgery.jsonl.
- **F1 CONFIRMED (mod-9 period spectrum, L13-16 max)**: ds-period-3 R² jumps -0.13 -> 0.42@2250 ->
  0.96@2500 (plateau onset 2300 ✓); ds-period-9 flat until 3000, 0.77@3500 -> 0.93 (snap ✓).
  N-periods {2,5,10,100} HIGH from ckpt 1 (0.93-0.95, pretrained basis ✓) and DECLINE to 0.52-0.57 by
  6000 — compression now visible in the periodic basis too.
- **F2 CONFIRMED**: output-DFT coset_frac = 1.000 for steps 2250-3000 (pure frequency-3 logit profile
  during the plateau — Nanda-style coset in Fourier form), falls at snap to ~0.45 converged.
- **F3 CONFIRMED (mod-8 control)**: ds periods flat-negative throughout; N-periods static ~0.86-0.95
  (mod-8 KEEPS the pretrained value basis it uses; mod-9 discards it — clean contrast).
- **F4 RESOLVED — NO HIDDEN STAGE, CONSTRUCTION REROUTED**: sibpen1's output coset_frac ~0.000
  everywhere (penalty destroyed coset expression ✓ manipulation check). Internally ds3 stays ~0 through
  3000 (scratch had 0.96 by 2500) — the coset stage never formed. Then ds9 rises FIRST (0.79@3500 vs
  ds3 0.36@3500): the penalized model built the FINE solution directly, order INVERTED, ~1000 steps
  late. => The staircase is CHOSEN (greedy), not forced; removing the coset's payoff removes the stage
  internally at a 48% time cost. X-ABL phantom-transition question closed: genuine reroute.
- **S1 REFUTED-AS-DESIGNED (honest null)**: projecting the rank-<=6 upgrade subspace (class-mean fit,
  L13-14, FINAL POSITION ONLY) out of the converged model: exact 0.976 -> 0.958 (tiny), agree3
  unchanged; mod3-subspace ablation also nearly null (0.959/0.972). Random no-op ✓. Scoped null:
  final-position rank-6 class-mean projection at 2 layers is insufficient — info re-enters from other
  positions/layers or lives in periodic (not class-mean) coordinates.
- **S3 NULL WITH A DESIGN LESSON**: rank-8 INLP deletion of the LINEAR raw-digit-sum direction at L10:
  no effect on either model. The circuit uses PERIODIC components cos/sin(ds/T) (F1 proves they exist,
  R² 0.93-0.97) which are ~orthogonal to the linear-sum direction — we deleted the wrong coordinate.
- **NEXT KNIFE (built: causal_surgery.py MODE=periodic + run_surgery2.sh)**: delete the ds-period-9
  cos/sin PLANE (ridge-fit directions from the fourier suite) at L13-16, ALL positions. Pre-reg S4:
  converged mod-9 collapses toward 1/3 WITH agree3 intact (surgical heuristic re-exposure, periodic
  version); deleting ds3+ds9 planes -> toward chance; mr7 unaffected by either.

### 2.14 SCOPE DECISION (recommended 2026-08-03, pending PI sign-off): ONE paper for ICLR 2027.
Spine: Act 1 law -> Act 2 spectral mechanism -> Act 3 causal control; eval-gaming as framing only;
entrenchment explicitly bounded-not-explained. Rationale: the acts evidence each other (sibpen
inversion belongs to both 2 and 3); an install-only paper's abstract would end "mechanism unknown".
Needs a ~30% main-text compression pass (P1-P9 to half-page, disentangle to 3 sentences + appendix,
target ~9.5 pages; appendices unlimited).
**PROPOSED (2026-08-06, user's "plant the actual mechanism" question — build on request; scope call
user+PI: could be THIS paper's Q3 capstone or paper 2's opener): GHOST-FEATURE INJECTION** — during
mod-9 training, hook adds alpha*(cos(2pi*ds/9)*u1 + sin(2pi*ds/9)*u2) into residual at L13 (oracle ds
per example via dataset column; fixed random orthogonal pair u1,u2). Pre-reg: learns at local-modulus
speed (~100-300, no plateau) = activation-level installation in the (bypass, faster) cell. Phase 2:
anneal/remove injection after convergence -> survives = internalized ("training wheels");
collapses = scaffold-dependent. Cost: one trainer mode + 2-3 runs. Gated on S4 confirming the planes
are the causal handle.
**PAPER 2 BACKLOG (ICML 2027 cycle — "what makes a heuristic sticky"):** chronometer (patch-readiness),
wd=0 donor, shrink-and-perturb sweep, X1' task-diversity, X2' selection-vs-scale, dense-save donor,
addmerge-50k, bits-recovered logging, second-model-family generality (unless pulled forward as this
paper's appendix — the one acceptance-risk item), VIB/DANN thread.
**COMPRESSION PASS EXECUTED 2026-08-07** (paper 23->20 pages, zero errors/undef refs): new title
"Why Language Models Learn Heuristics First — and How to Accelerate Past Them"; abstract rewritten
Q1/Q2/Q3 (half length); contributions -> 4 Q-framed bullets; §5 obstructions -> one paragraph; §6.5
stripped (planned-instruments TABLE deleted, controls para deleted, three nulls -> one short
"what deletion could not do" para + appendix pointer); §7 REBUILT as three subsections (What fails /
What works / Staircase chosen) — transplant+dose+seeds+entrenchment compressed, correlate para cut to
3 sentences, causal-surgery para deleted; figs install_transfer + install_dose moved to appendix
(app:dwell renamed "Entrenchment detail"); round1/round2/fourier figs stay in main.
LEGACY CHECKLIST (original user calls, 2026-08-05): (a) DONE: "Is the plateau just the
answer" paragraph reframed behavior-first, P1 demoted to instrument check. (b) §5 obstructions -> <=0.5
page (keep the 1.000/0.180 memorization number + 2-sentence carry-depth; rest to appendix). (c)
Trailing-space artifact -> 3-4 lines in setup + appendix (user wanted removal; keep short: it preempts
tokenization reviews and is a real contribution to methodology — argued, user to confirm). (d) Install
section: compress dose/dwell mechanics to a few sentences + appendix; keep entrenchment headline, sign
flip, 2x2, curriculum. (e) Surgery round-1 nulls -> appendix one-liners. (f) Probe positions: stated
once in §6 protocol (final token + numeral tokens, probed separately, never aggregated).
**NEW ANALYSIS QUEUED (mechanism of the 18x, "install the aggregate and every residue becomes local"):**
run fourier_suite on (i) scaffold_digitsum DONOR ckpts (steps 1000-6000) and (ii) the dsum6k->mod9
TRANSFER ckpts (saved every 1000). Pre-reg: donor carries deep LINEAR raw-ds (high ridge R², it must —
ds is its output) but NO ds-periodic planes (its readout needs magnitude, not phase); the transfer run
then grows period-3 AND period-9 planes SIMULTANEOUSLY within <=200 steps (no staircase — matching
behavioral f25=25). If confirmed: aggregation is the bottleneck; once the aggregate exists, periodic
readout is as cheap as a local modulus — completes the mechanistic account of the 18x.
**FRAMING RECAST (user directive 2026-08-06, implement AS the compression pass):** reorganize the
paper around THREE QUESTIONS: (Q1) WHY does the heuristic form? [digit-locality law + pretrained
basis + cost-beats-payoff selection]; (Q2) WHY is gradient descent attracted to it? [payoff per unit
phase precision; anti-spectral-bias ordering; staircase = spectrum filling in, biggest-margin first];
(Q3) HOW to accelerate past it? [what works: install the aggregate (18x, no plateau), factor donors
(3x), converged-stage curriculum (unlearnable->600); what fails: install the coarse answer (entrench),
remove the payoff (48-85% slower)]. Title candidate for PI: "Why Language Models Learn Heuristics
First — and How to Accelerate Past Them". Entrenchment mechanism stays explicitly open (paper 2).
Note: "heuristic REMOVAL" is answered negatively by X-ABL — removal slows; acceleration = supply the
scaffold. Appendix "Exact procedures" (app:procedures) added 2026-08-06 for DFT/planes/ablation.
**PRE-SUBMISSION MUSTS (this paper):** S4 surgery2 verdict (in flight), error-offset analysis (free),
seeds x2-3 on sign-flip + dsum->mod9 headlines, mod-18 pre-registration (recommended), base-9 (PI
call), compression pass.

### 2.15 S4 + fourier2 + X-GHOST — DONE (pulled 2026-08-07). "EASY TO GIVE, HARD TO TAKE AWAY."
- **S4 NULL (third surgical null)**: deleting the ds-period-9 plane (and ds3+ds9) at L13-16, ALL
  positions, from converged mod-9: exact 0.976 -> 0.976, agreement unchanged; random ✓; mr7 ✓
  unaffected. SCOPING: planes were FIT AT THE FINAL POSITION — numeral-position phase carriers live in
  different directions and survive; L17-24 untouched; ridge plane = ONE optimal readout, not the unique
  carrier. Tally: class-means (S1), linear-count (S3), periodic planes (S4) all null => the fine
  computation has NO low-rank single-position bottleneck we can find. NECESSITY UNLOCALIZED;
  redundancy/distribution is the picture. P9 "site consolidation" must be read as an OBSERVATIONAL
  (readout) fact, not a causal bottleneck. Remaining causal tools: head ablation (E2 heads),
  full-layer cross-checkpoint patching (sufficiency), numeral-position-fit planes. (Paper-2 leaning.)
- **fourier2: pre-reg HALF-REFUTED, richer**: the digit-sum DONOR is FULL of periodic structure
  (ds9 plane R² 0.95-0.98 from ckpt 1000; ds3 0.5->0.88; cosetF ~0 ✓ no mod-9 output structure).
  "Donor has count but no phases" WRONG — supervising the scaffold quantity builds a complete smooth-
  function basis of ds (count AND phases readable). Transfer: behavior 0.97 and both planes present at
  the FIRST checkpoint ✓ (simultaneity trivially confirmed). Refined 18x mechanism: the donor installs
  the full ds-manifold; any residue readout is then one linear step ("every residue becomes local").
- **X-GHOST G1: CONFIRMED BEYOND PRE-REG**: alpha=0.3 f95=25 (first eval!), alpha=0.1 f95=125; final
  **1.000 BOTH** (above scratch's 0.976 ceiling); no plateau. Injecting the oracle phase plane at ONE
  position of ONE layer makes mod-9 faster than any local modulus measured (mod 10 = 50). CAVEAT:
  metrics measured WITH injection active. **G2 pending**: pushed checkpoints have no hook -> standard
  eval = wheels-removed test. Repos: ft_mr8_410m_l3e-5w0.05_c6000s_nims50000_e25_ghost{0.1,0.3},
  step-500..6000. Branches: survives = internalized; collapses = scaffold-dependent (then anneal arm).
- **THE ASYMMETRY (paper framing gold)**: adding one plane at one position -> 30x faster learning to a
  HIGHER ceiling; deleting fitted planes from the converged model -> nothing. Installation trivially
  easy, removal impossible-so-far: training smears the computation into redundant copies. Connects
  entrenchment, S-null series, and Q3 in one sentence.

### 2.16 X-GHOST G2 (wheels-off) — DONE (2026-08-08). SCAFFOLD-DEPENDENT: RENTAL, NOT OWNERSHIP.
Without injection, EVERY ghost checkpoint (alpha 0.1/0.3 x steps 500/2000/6000) = CHANCE (0.110-0.113;
no coset either). The model wired a readout to the oracle plane and built NOTHING — zero gradient
pressure to construct when a free feature exists = the selection law's limiting case (4th least-effort
demo). Ghost = pure calibration (readout 25-125 steps) + the rental/ownership contrast vs the donor
route (donor 18x = owned, hook-free mechanism). Paper-2: ANNEALED injection (fade alpha) — can rental
convert to ownership? Paper updated (§7 ghost para + ledger); last [pending] in manuscript closed.

### 2.17 PHASE TRACING (built 2026-08-08, ship pending) — upgrade the two mechanism models to measurement.
`phase_trace.py` + `run_phase.sh` (array 0-2). P-A prefix trace (mr8 @1000/2750/6000): prefix-phase
decodable at every numeral position, full-sum phase ONLY at last token — the per-digit rotation
signature; KILL = anticipation (full phase early) or absent prefix at convergence. P-B mod-8 control.
P-C noise trajectory (mr8 all ckpts): angular error of period-3/period-9 phases crosses wedge
thresholds (60deg/20deg) at plateau onset / snap respectively; KILL = crossings unaligned. Paper
already names the prefix test as the model's "next falsification target" — either outcome slots in.

### 2.18 PHASE TRACING RESULTS — DONE (pulled 2026-08-09). THE MODEL BECOMES MEASUREMENT.
Files: probes_ckpt/phase_{prefix_mr8,prefix_mr7,noise_mr8}.jsonl; figure new_result/plots/phase_noise.png.
- **P-C CONFIRMED, textbook-clean**: period-3 phase crosses its 60-deg wedge threshold EXACTLY at
  plateau onset (frac-within: 0.34@2000 -> 0.70@2250 -> 0.98@2500) and period-9 crosses its 20-deg
  threshold EXACTLY at the snap (0.15@3000 -> 0.31@3250 -> 0.76@3500 -> 0.91@4250). The wedge account
  now has its quantitative figure (paper fig:phasenoise). Nuance: T9 R² is negative until 3250 — the
  fine wave is ABSENT then built, not merely noisy-then-sharp; arrival and sharpening coincide.
- **P-A PARTIALLY CONFIRMED + control caught a confound**: at the CROSS-TOKEN position (pos1, the only
  position where accumulation is distinguishable from token identity with 2-token numerals): prefix
  phase R²=0.951 in converged mod-9, -0.27 in mod-8 control, negative in mod-9 pre-snap ✓✓✓; NO
  anticipation (full-sum phase at pos0 = -0.16 everywhere) ✓. CONFOUND EXPOSED (by P-B, working as
  designed): pos0 "prefix" decodable at 0.90 even in mod-8 — a single token's digit sum is a function
  of its identity; single-token positions excluded. LIMITATION: 2-token numerals give exactly one
  accumulation step; the currA (max500k, 3-token) model could extend the trace (optional).
- Paper updated: §6.5 rotation claim now "trace matches per-digit accumulation exactly, within scopes";
  §6.4 wedge model now FOUR consequences held incl. the direct measurement + fig:phasenoise added.

### 2.21b QWEN STAGE 2 + PROBES — DONE (pulled 2026-08-29). DEVIATION FULLY EXPLAINED AND REPAIRED.
- **FULL RESCUE**: the published failed cell (max50000, chance for 25k from scratch) opens at 0.234
  zero-shot from the 3-digit donor (q999c@step-1000) and hits f95=575. Ladder 3->4: opens 0.495
  zero-shot, f95=175 (scratch: stuck 5000 steps). NO coset plateau in either — fine-first again.
  Files: mr8_qwen0.5b_*initq999c*.jsonl. KILL condition not approached.
- **PROBES (endpoint-only: crossmodel repos hold ONE checkpoint, step-25000)**:
  stuck mr8: ds3/ds9 max R2 = -0.001 over ALL layers + zero output-DFT structure -> H-STUCK
  CONFIRMED at the endpoint (no hidden progress trace; extension futile; curriculum was the fix).
  mr5 (mod-6 success): ds3 = 0.998 at convergence (global-factor plane built), N10 = 0.977.
  The contrast row: ds3 0.998 where needed-and-learnable vs -0.001 where needed-and-starved.
- NEXT (optional): the rescue run saved every 500 -> probe ITS checkpoints to watch qwen build
  mod-9 fine-first internally (period-9-without-period-3 spectra, cross-model analogue of the
  sibpen panel). Repo name is AUTO-SHORTENED (tag >96 chars, gotcha #11) — get it from a "Pushed
  checkpoint" line in logs/qwencurr2_*_1.out (gotcha #14), NOT from the metrics filename.
- Paper hooks when ready: Qwen paragraph "informative deviation" -> add one sentence (cliff +
  575-step rescue); appendix E cross-model paragraph gains the mechanism contrast (ds3 0.998/-0.001).

### 2.21 QWEN MOD-9 DEVIATION — STAGE 1 DONE (pulled 2026-08-28): POSITION CLIFF FOUND.
**max999 (3 token positions): LEARNS FAST, f95=300, held-out 0.965 (operand-disjoint => not
lookup), and NO COSET PLATEAU (0 ckpts near 1/3) — qwen learns mod 9 fine-first even when it
learns. max9999 (4 positions): STUCK at chance all 5000 steps (max 0.158).** The cliff is between
3 and 4 positions: aggregation bottleneck confirmed as the deviation's mechanism; "no staircase"
is a route property of the family, not a stuck symptom. Stage 2 built (run_qwen_curr2.sh, array
0-1, pre-reg in header): ladder rung 3->4 (init max9999 from max999@step-1000) + full rescue
(init max50000 from same). KILL: opens at chance and stays.
RESOLVED 2026-08-29 — GOTCHA #14 CANDIDATE: HF repo names carry a trailing **_purenum** suffix
that the metrics FILENAMES do not. All qwen probe SKIPs and both init crashes were wrong-name 404s;
pushes worked all along (stage-1 log: "Pushed checkpoint step-250..2500+" to ..._purenum). When
building repo ids, NEVER derive from metrics filenames — check a push line in the run's .out.
Scripts fixed (+_purenum): run_qwen_curr2.sh, run_fourier_qwen.sh, run_qwen_curr.sh task 2.
Also: unauthenticated HF /api 401 ≠ repo-missing (auth-required response for private/gated).

(original prep note, 2026-08-27:)
Files: run_fourier_qwen.sh (array 0-1), run_qwen_curr.sh (array 0-2); fourier_suite.py gained
[tok_base] arg (argv 9, default Pythia) + skip-missing-revision guard (both backward-compatible).
Motivating datum (computed from mr8_qwen jsonl): eval loss slope over final third = -4.9e-10/step,
acc pinned at exactly 1/9 — a perfect chance equilibrium, NO visible drift. But Barak-et-al hidden
progress means behavioral flatness cannot rule out silent construction -> probe first.
- fourier_qwen task 0 (mr8, stuck cell): H-STUCK = ds3/ds9 planes flat-negative everywhere ->
  true equilibrium, extension unpromising. H-HIDDEN = planes rising -> longer training will snap.
- fourier_qwen task 1 (mr5, SUCCESS cell = mechanism replication): N-period-2 from ckpt 1
  (pretrained parity, local), ds-period-3 arriving at the SNAP (~2800). If holds, the MECHANISM
  replicates cross-tokenizer (paper currently only claims the behavioral law does).
- qwen_curr tasks 0/1 (stage 1): mod 9 at max 999 / 9999 (3-4 token positions vs 5 in the failed
  run; token-count IS the magnitude axis under digit tokenization; two arms because max999 pool
  ~1000 = memorization-trap risk). If learnable -> aggregation-bottleneck account; stage 2 (gated,
  init max-50000 from converged stage-1, xcurr arm-A analogue) predicted to rescue. If BOTH stay
  at chance -> deviation is basis/payoff, not aggregation. Task 2: continue scratch from step-25000
  for +25k (extension control; predicted to stay at chance).
- NOTE: qwen crossmodel save cadence unknown (run_crossmodel_law.sh is cluster-side only); probe
  sweep skips missing revisions; task 2 fails cleanly if step-25000 was never pushed.
Out: probes_ckpt/fourier_qwen_mr{8,5}.jsonl + new run metrics.

### 2.20 SIZE-LAW SWEEP — DONE (pulled 2026-08-25). 10/10 CELLS, LAW SCALE-INVARIANT.
Files: purenum_metrics/mr{5,7,9,10,13}_{70m,160m}_*constlr30000steps_nimsimple_max50000*.jsonl
(mr5/mr7 70m from earlier runs, evalevery25 suffix). run_sizelaw.sh (predictions in header, written
before the runs). Results (trajectory class + plateau height only; f95 not cross-size comparable):
- mod 10: snap both sizes (f95=250/250, no plateau) — discriminating case holds at 70m/160m.
- mod 14: plateau at exactly 1/7 both sizes (steps 250-2250), then climb (f95 15500/17500).
  Cost-beats-payoff (1/7 over available 1/2) at every scale tested.
- mod 11: ramp, no divisor plateau (f95 6500/7000). mod 6 @160m: 1/3 for 250-1750, f95=2250.
  mod 8 @160m: snap f95=500. (70m mod6/mod8 already confirmed.)
- IMPORTANT DATA HYGIENE: the OLD purenum-era 70m/160m x {mod4,5,6,7} x 2seed batch
  (mrN_SIZE_seedK.jsonl short names) is VOID (trailing-space artifact, seq_acc==0 for 70k steps).
  Never use. The paper's "70m/160m" claims now rest ONLY on the nimsimple 30000-step runs.
- Paper updated: §law "Scale replication" paragraph + fig:sizelaw (figs/sizelaw.png,
  plot_sizelaw.py); Limitations reworded; abstract adds "across model scales". Law now holds on
  three axes: base, family, scale.

### 2.19 MECHANISM-SWITCH TEST (P-D) — DONE (pulled 2026-08-13). H-COUNT KILLED, H-ROTATE
CONFIRMED, TRANSITION RULE MEASURED (median 4.3 deg).
Files: probes_ckpt/phase_switch_{mr8,mr7}.jsonl. RESULTS (all cells pos1 = cross-token unless noted):
- **H-COUNT KILLED (kill condition fired)**: cross-token COUNT is 0.98 from step 250 — but 0.94 in
  the MOD-8 CONTROL too: it's the pretrained magnitude-confounded feature (E1), not a constructed
  intermediate. It only FADES through training (best 0.98→0.83); no task-specific count-leads-phase
  epoch exists.
- **H-ROTATE CONFIRMED (3/3)**: (1) cross-token phase arrives in the snap window: -0.25@3000 ->
  0.19@3250 -> 0.78@3500 -> 0.89@3750 -> 0.95@6000, co-timed with F1 final-position ds9 (0.77@3500) —
  circuit appears everywhere at once, not final-first. (2) no count-leads epoch. (3) INCREMENTS:
  median |dtheta err| vs predicted 2pi*ds(tok)/9: ~90deg (chance) pre-snap -> 36@3250 -> 10@3500 ->
  4.3@6000; frac<20deg 0.10 (= exact chance 20/180) -> 0.95. mr7 control at chance ALL ckpts.
- **BONUS (compression localized)**: deep-layer (L13-16) count at pos1 falls 0.98 -> 0.78, steepest
  drop EXACTLY across the snap (0.848@3000 -> 0.783@3500) while phase rises there; count best layer
  migrates L20-24 -> L2-L6 (E1's migration, now at the cross-token position). Decision site swaps
  count for phase at the moment the rotation circuit arrives.
- Sanity: no anticipation ever (full@pos0 ~ -0.17 all steps); old prefix cells replicate
  (0.946 vs 0.95; -0.27 exact). LIMITATION: period-9 targets only — count-then-wrap for the COSET
  (period-3) stage untested; optional 3-ckpt follow-up. 2-token numerals = one increment step.
- Paper updated (app:proc-phase transition-rule paragraph + §6.5 clause). Refined story: count was
  inherited not built, never wrapped; rotation circuit arrives whole at snap; site evicts the count.

(pre-registration, written 2026-08-12 before the run:)
Files: phase_trace.py MODE=switch (new; pre-reg in docstring), run_phase_switch.sh (array 0-1).
Question: did the model pass through a count-then-wrap stage (aggregate integer digit sum, wrap at
readout) before converging to wrap-as-you-go (running residue phase, no count)? Motivated by gotcha
#13 + the E1 fade timing (count present early [magnitude-confounded], steepest fade exactly in the
plateau->snap window when the phase arrives). New cells vs 2.18: (a) prefix COUNT (integer) target
per position — the missing discriminator; (b) rotation increments: decoded Delta-theta between
consecutive positions vs 2*pi*tok_ds/9 (tests the TRANSITION RULE, not just states); (c) DENSE
checkpoints (16 steps, 250..6000, dense through 2000-4000) vs 2.18's three.
H-count: window with cross-token count present + phase absent; final-position phase precedes
intermediate; count fades while phase persists (co-timed with E1 fade). H-rotate: phase rises at
snap everywhere at once, no count-leads-phase epoch, increments match x40deg immediately.
KILL for H-count: no count-leads-phase epoch. Either outcome upgrades the paper: H-count = the
scaffold story gets mechanism-level resolution (build-via-count, then compress); H-rotate = the
rotation circuit is learned direct, scaffold was never the intermediate representation.
Analysis join: overlay on E1 fade (digitsum_attn) + F1 ds9 arrival (fourier_mr8).
Out: probes_ckpt/phase_switch_{mr8,mr7}.jsonl.

### 2.20 PAPER REVISION (2026-08-19) — mentor (아빠) feedback pass. DONE locally; punch list below.
Feedback: abstract -> ~70%; intro too thin + first sentence unclear + needs problem/prior-work/
contribution structure readable by general LLM researchers; single-model scope will be attacked;
figure fonts unreadable in print; statistical significance under-stated; too much Fourier in main;
wants THE ALGORITHM stated with evidence but NOT Neel-style weight archaeology; missing conclusion.
Done in main.tex (compiles clean):
- Abstract 430 -> ~270 words (63%), same 3-question skeleton + headline numbers.
- Intro rebuilt: plain first sentence; para 1 problem (reliability + efficiency stakes); para 2
  prior work + gaps (grokking/endpoint-analyses/shortcut-lit/warm-start); para 3 setting + 3
  load-bearing design choices; contributions kept.
- Setup: new "Statistical discipline" paragraph (n=2000 CIs, exact d/m shelf predictions, 2%
  replication, 3-seed install contrast, n=1 flags -> ledger).
- §6.5: Fourier-outcomes tail compressed; scoping/tautology note MOVED to App B.4; new
  \paragraph{The algorithm, stated} = pseudocode + per-clause measurements + explicit
  "weight-level implementation deliberately open; not circuit cartography".
- Deletion paragraph compressed ~60%.
- §9 retitled "Conclusion, limitations, and outlook" + recap paragraph (the law -> the mechanism
  -> the design rule) + limitations expanded: single model/tokenizer named as top gap, cross-model
  plan (Llama/Gemma/Qwen; law converts tokenizer differences into pre-registrable predictions;
  output DFT is vocab-independent so instruments travel).
- FIGURES: plot_style.setup_style bumped to font 13 / labels 13.5 / ticks 12 / legend 11.5 (print
  floor comment inline); setup_style() injected into the 9 non-style paper scripts; hardcoded
  fontsize<=11 sed-bumped across all 15 paper scripts; all_mods_theory REDESIGNED for paper
  (panels only at figsize (10,3.9) — the embedded PROMPT/THEORY text block is exploratory-only,
  now deleted; caption + Table 1 carry it). All 15 scripts rerun OK; PNGs copied to paper/figs/.
FIGURE PASS 2 DONE (2026-08-20, "paper style" per Ijin): NO in-figure description text — every
suptitle + fig.text banner deleted across all 15 scripts; sentence-titles -> short identifiers
("mod 9", "target: N mod 11"); SHOUTY CAPS de-shouted; all canvases <=10.5in; legends shortened
to fit narrower panels (tx_round2). probe_depth RESTRUCTURED: was 3 panels in a half-column
subfigure (unreadable at any font) -> now single mod-9-ladder panel (the captioned claim);
mod-8 + base panels moved to NEW full-width appendix fig probe_depth_extra.png (fig:probeextra,
placed before app:lawfigs; referenced from §5.1 P5 sentence). All 15 scripts rerun, PNGs in
paper/figs, compile clean, pages 5/10/12 eyeballed.
GOTCHA #14 (tooling): the Bash tool's heredoc transport EATS one backslash level — a python
str like "a\\nb" inside <<'EOF' arrives as a real newline. Never pass backslash escapes through
Bash heredocs; Write the script to scratchpad and run it (fix_figs.py pattern), or use Edit.
FIGURE PASS 3 DONE (2026-08-24, "actually see each image" per Ijin): every paper figure rendered
and inspected as an image; collisions fixed one by one. Standard treatment: legends get
frameon=True + facecolor white/SURFACE + framealpha ~0.92-0.95 + labelspacing 0.25-0.3 (opaque
white patch beats relocating when panels are busy); event labels (plateau/snap) staggered
left/right of their vlines; long legend labels shortened (f95= dropped, redundant qualifiers cut);
disentangle's 8-series legend moved to a shared fig.legend BELOW both panels; all_mods_theory
middle/right legends repositioned per-panel; phase_switch right ylabel shortened (was clipping);
1/3-style threshold labels moved off curves. Verified clean: all_mods_theory, probe_time,
probe_depth, probe_depth_extra, digitsum_attn_e1e2, phase_noise, fourier_dynamics, qwen_law,
base9, install_transfer, install_dose, dwell_probe, disentangle, moremods_test, nimsimple_mag_size,
tx_round1, tx_round2, phase_switch. Paper recompiled clean.
PUNCH LIST (needs user/cluster or a later pass):
1. rung_metric.png has NO local plot script — locate (cluster?) or rewrite; regenerate with new style.
3. Cross-model law replication (dad's #1 experimental ask): behavioral 12-moduli sweep on
   Llama-3.2-1B / Gemma-2-2B / Qwen2.5-0.5B. finetune_constlr.py is AutoModel-generic; per-model
   punch: tokenizer digit-chunking differs -> re-derive local/global predictions per tokenizer,
   pre-register, then run. New sbatch needed.
4. Seed batteries on sign-flip + scaffold-transfer headlines (already PLANNED in ledger).
5. Base-9 test still queued (sec:base9).
6. At submission: strip \donetag/\pending/\planned status tags + "Status of this draft" para +
   ledger table into a cleaner reproducibility appendix.

### 2.21 BASE-9 — DONE (pulled 2026-08-21). **4/4 CONFIRMED, THE LAW'S KILLER TEST LANDED.**
- mod 9: f95 = **50** (base 10: 3725, the paper's central plateau) — 75x, at 1.00 by step 100 ✓
- mod 8: chance (0.125) dwell to ~4000 then slow grind, best 0.938 at 30k, NO f95, no coset shelf ✓
- mod 5: chance (0.20) dwell to ~2450, f95 = 9000 (base 10: <1000) ✓
- mod 6: shelf at 0.49-0.51 for steps 100-2650 (predicted **1/2** = mod-3 coset, base-9 local factor),
  then snap, f95 3050 (base-10 shelf was 1/3 on parity) ✓ — the shelf VALUE moved exactly as computed.
Paper updated: sec:base9 -> results + fig:base9 (plot_base9.py); ledger row DONE; "Status of this
draft" pending-clause removed (no pending experiments remain); limitations + Beyond-arithmetic +
contributions Q1 updated. Note asymmetry worth a future thought: base-9 mod 8 formed NO parity-coset
shelf (unlike base-10 mod 9's 1/3 shelf) — smaller payoff fraction (ln2 of ln8)? Not claimed in paper.
CROSS-MODEL (Qwen2.5-0.5B) — DONE (pulled 2026-08-21 after transformers-too-old fix: env had
<4.37, no Qwen2Tokenizer; upgraded to 'transformers>=4.40,<4.46' + accelerate — pin because
evaluation_strategy removed in 4.46; UPGRADE IS THE FIRST SUSPECT if future Pythia reruns differ).
**RESULT 5/6 CONFIRMED, shelves EXACT; 1 informative deviation:**
- mod 8 f95=100, mod 10 f95=50 (locals snap ✓✓)
- mod 6: shelf 0.33 for 2400 steps -> snap, f95 2800 ✓ (shelf value exact)
- mod 14: shelf 0.143 for 6500 steps ✓ + BONUS: transient second rung at 0.50 = mod-7 coset
  (divisor staircase's next rung visible behaviorally — Pythia never showed this clean)
- mod 11: chance dwell -> rise, no divisor shelf ✓ (f95 3500)
- mod 9: **CHANCE (0.111) FOR ALL 25k — staircase never formed.** No coset, no snap. Local/global
  split survives (global cell got HARDER not easier); the STAIRCASE is not tokenizer-universal.
  Open Q (next pre-registrable target): why digit-level tokenization suppresses the mod-9 coset
  stage — more positions to aggregate / different pretrained basis / payoff landscape. Notable:
  Qwen2.5 is math-heavy-pretrained AND digit-tokenized, yet mod 9 was the unlearnable cell.
Paper updated: §4 new "Cross-model replication" paragraph + fig:qwen (plot_qwen_law.py); abstract
+ contributions Q1 clauses; limitations rewritten (deep instrumentation single-model; behavioral
law replicated); Beyond-arithmetic updated; ledger row added. Compiles clean.

(original prep note, 2026-08-20:)
BASE-9 (run_nimsimple_base9.sh, array 0-3, was built long ago but NEVER SUBMITTED; gen_nim_simple.py
--base support verified: 50000 -> "75525" base-9, int(.,9) round-trips). Pre-reg in script header:
mod 9 SNAPS (now local); mod 8 + mod 5 become global (slow, no early shelf); mod 6 shelf MOVES
1/3 -> 1/2 (local factor flips 2 -> 3). 30k steps, eval every 50. Paper sec:base9 flips from
[PENDING] when pulled.
CROSS-MODEL (run_crossmodel_law.sh, NEW, array 0-5): behavioral-only law replication on
Qwen2.5-0.5B (ungated; digit-level number tokenizer = maximally different chunking from NeoX).
Moduli {8,10,9,6,14,11} = 2 locals incl. the discriminating mod-10, 3 shelves (1/3, 1/3, 1/7),
1 ramp. Same decimal data as the Pythia sweep. Pre-reg = trajectory CLASSES + SHELF VALUES only
(f95 not comparable across architectures, not predicted). finetune_constlr.py MODEL_MAP gained
qwen0.5b / llama1b / gemma2b keys (llama/gemma HF-gated — token account must accept terms).
Second family = resubmit with MODEL=llama1b.
MINIMAL-SET DECISION (Ijin asked what must replicate on other models): ONLY the behavioral law
sweep, 1 model, 1 seed. NOT needed elsewhere: probes/Fourier/phase (Pythia depth story),
installs/donors/curricula (single-model contributions), seeds (law confirmation is categorical:
classes + shelves, not point estimates), base-9 on other models, other sizes.

### 2.22 APPENDIX PURGE (2026-08-22, Ijin's editorial rule). DELETED from main.tex: Appendix E
(obstructions: composition/addmerge + small-pools memorization + carry-depth para), Appendix G
(trailing-space artifact), Appendix H (memorization-trap pool figure). RULE, standing: the paper
never mentions train/eval overlap, our own past bugs, or hygiene-class mistakes — obvious
discipline is practiced, not narrated. Refs repointed: intro reliability + related-work
eval-gaming sentences now cite sec:install (entrenchment = the real evidence); eval-discipline
overlap clause dropped; addmerge row removed from tab:cells; 2 ledger rows dropped. The
trailing-space GUARD stays in the code + reproducibility statement (a guard is practice, not
confession); this section (§3 below) remains the internal record. Compiles clean, no dangling refs.

## 3. THE TRAILING-SPACE ARTIFACT (critical methodology)

Prompts ending in a trailing space (`"... mod 9 = "`): tokenizer merges the space into the
answer token (`" 6"`), prompt+answer tokenizes to SAME length as prompt → label masking
masked the answer → models supervised only on EOS padding → fake uniform "learning" at ~105
steps regardless of modulus (it's EOS-learning).
**VOID**: all original modarith_subtract runs (all magnitudes/holdouts), prompt_ladder steps 0/1/2,
ladderC step 2a (incl heldout) — i.e., the old "bare math snaps instantly"/"routing" results.
**SAFE**: everything ending without trailing space: nimsimple family, 2b, steps 3–6, paper variants, cheat data.
**GUARD**: `validate_prompt_boundary()` now hard-fails at startup in finetune_constlr.py AND
finetune_oldconfig.py. Convention: math/NL prompts end at '='/'is' with answer ' 6' (leading space);
game prompts end at 'take' with bare-digit answer '6' (nimsimple convention).
Figures still embedding void curves (to regenerate): ladder_v2_full (steps 0/2/2a), 2a_vs_nimsimple,
prompt_ladder_step2/4/6 overlays, modarith_subtract_finegrain, modarith_subtract_holdout_vs_no.

## 4. In-flight / queued jobs (as of compaction)

**BUILT 2026-07-13 (T/X batch; trainer now takes SEED=arg12, SIB_PENALTY=arg13; TAG gains _sibpen{l}):**
- `run_signflip_seeds.sh` (5): T1a mod3pre6k->mod-6 sign flip + seeds 43/44 for scratch-mr8 and I1.
- `run_scaffold_donors.sh` (3): T1b phase 1 — digitsum/altsum22/firsttwo donors (gen_scaffold_tasks.py;
  donor pools exclude mr8/mr10 eval piles; altsum22 = alternating sum + 22, determines mod 11).
- `run_scaffold_transfer.sh` (5, GATED on donors >=0.95): the 2x2 + format control, 12k steps.
- `run_xcurr.sh` (2): stage-1 mod-9@max10k (ckpt every 250) + arm-C scratch@max500k baseline.
- `run_xcurr_arms.sh` (2, GATED): arms A/B INIT_FROM stage-1 (REV_A first ckpt >=0.95; REV_B first in
  [0.45,0.70] — EDIT revisions in script after stage-1 curve).
- `run_xabl.sh` (2): X-ABL staircase ablation, sibling penalty lambda {1,5} — THE within-task test of
  "is the plateau the fast path for mod 9" (cross-moduli comparison doesn't license it — user's point).
  Primary: total steps to f95 vs 3725. Faster = greedy detour; slower = staircase efficient.

**BUILT 2026-07-21 (mechanistic suite — C4 + C2):**
- `fourier_suite.py` + `run_fourier.sh` (array 0-2: mr8 dense / mr7 dense / sibpen1): per-checkpoint
  (a) Z_9 OUTPUT DFT (Nanda-style on the answer simplex — vocab-independent; coset = freq-3 energy) and
  (b) period-decodability spectrum: ridge-decode cos/sin(2*pi*x/T) from every layer, x=digitsum for
  T in {3,9} (constructed circuit) and x=N for T in {2,5,10,100} (pretrained basis). Pre-reg F1-F4 in
  header; F4 = the X-ABL hidden-stage test on sibpen1 checkpoints. Out: probes_ckpt/fourier_*.jsonl.
- `causal_surgery.py` + `run_surgery.sh` (array 0-3): C2 projection-ablation (upgrade/mod3/random
  subspaces at L13-14, final position; converged @6000 vs plateau @2750) + rank-8 digitsum INLP at L10
  on mr8 vs mr7 (presence-vs-USE arbiter promised in paper §6.3 caveat). Pre-reg S1-S3 in header.
  Out: probes_ckpt/surgery.jsonl. NOTE: hidden_states[L] = output of block L-1; hooks target
  gpt_neox.layers[L-1] to match all prior probe indexing.

**SUBMITTED 2026-07-12:** `run_probe_dose.sh` — dwell-state probe (E1+E2) on the 5 dose arms'
  own checkpoints (steps 500-8000 every 500; scratch already covered by digitsum_attn_mr8.jsonl).
  THE escape-time test (pre-reg in script header): (a) cross-run — dwell-period deep digit-sum R²
  orders runs by escape time; (b) within-run — R² recovers BEFORE the climb. Kill condition: escape
  with R² flat ~0.55. Out: `probes_ckpt/digitsum_attn_dose{1k,2k,3k,4k,6k}.jsonl`.
  Pull: `scp ".../probes_ckpt/digitsum_attn_dose*.jsonl" new_result/probes_ckpt/`

**DONE, pulled:** `run_digitsum_attn.sh` (E1+E2, §2.7); `run_install_pre.sh` + `run_install_transfer.sh`
  (E3 phases 1+2, §2.8); `run_probe_mr2.sh` + `run_install_dose.sh` (E3b, §2.9). INIT_FROM support lives in finetune_constlr.py: arg 11 = "repo@revision@label",
  TAG += `_init{label}`. Prefinetune repos (checkpoints step-1000..6000):
  `ijinyu1113/ft_mr{2,4}_410m_seed42_lr3e-5_wd0.05_constlr6000steps_nimsimple_max50000_evalevery50_purenum`.
  NOT yet pulled locally: the two prefinetune metric curves (mr[24]_..._evalevery50.jsonl) — pull for the
  install figure's appendix panel.

**EARLIER QUEUED, STATUS UNCERTAIN (ask user / check cluster):**
- `run_nimsimple_base9.sh` (base-change test, mr 4/5/7/8, 30k steps). PRE-REGISTERED (draft §6.1):
  base-9 → mod 9 SNAPS (now local); mod 8 & mod 5 slow (now global; parity global in odd base!);
  mod 6 plateau MOVES 1/3 → 1/2 (local factor flips 2→3). Data gen: `gen_nim_simple.py --base 9`
  (verified; numerals digits 0-8, look decimal; answers base-invariant).
  Metrics: `mr*_constlr30000steps_nimsimple_base9_max50000_evalevery50.jsonl`.
  CAVEAT: trainer's aux mod-metrics parse numerals as decimal → garbage; use seq_acc only.
- `run_addmerge_long.sh` (50k steps, mr 7/8; does composition ever resolve?) — user said "later".

## 5. Remaining experiment queue (not yet coded unless noted)

1. **C-suite (causal circuit proof — user explicitly wants this; deferred "later")**:
   - C1 head-ablation across checkpoints → causal emergence curve. E2 top cross-chunk heads (mod-9
     model): L15.H8/H12 during plateau (1250–3250), L11.H2/H8 after snap (globalization idx ~0.55-0.64);
     mod-8 model's last-chunk head candidates: L10.H13 / L15.H8 (att-to-last-chunk = 1.0).
   - C2 subspace projection-ablation per checkpoint: fit mod-3 class-mean subspace + "upgrade"
     subspace (mod-9 means orthogonalized against mod-3) at L13–14; project out at inference.
     MONEY PREDICTION: deleting upgrade subspace from converged model → exact collapses to 1/3
     WITH mod-3 agreement intact = surgically recovering the heuristic.
   - C3 cross-checkpoint transplant at L13–14 (plateau 2750 ↔ converged 6000): sufficiency/necessity.
   - C4 trig/Fourier period-3/9 direction amplitudes across training (activation-subspaces methodology,
     arXiv:2505.05145) + project-out-period-9 causal check.
   (Was mid-build of C2/C3 script when user deferred — NOT written yet.)
1a. **CAUSAL SUITE v2 (2026-07-12, post design-panel; supersedes older E-A..E-F sketches). Panel-caught
   traps that MUST be respected in any variant: (T1) LABEL LEAKAGE — digitsum(N) mod 9 = N mod 9, so
   digit-sum supervision is partial target supervision; every "scaffold" arm needs the cross-target
   control. (T2) COARSENING TRAP — mod 3 is a deterministic coarsening of mod 9, so erasing mod-3 at the
   decision site makes the converged solution inadmissible (any full-run DANN-erasure-at-L13-14 result
   is vacuous); erase only upstream layers, time-limited windows, or answer-orthogonal components
   (floor(digitsum/9)). (T3) 2-token answers (digit sum <= 40+) re-enter trailing-space-bug territory —
   boundary-guard every new format. (T4) all headline install numbers are seed-42-only; add seed arg.**
   SHIP SET, tiered:
   - **T1a E-G sign flip (1-2 jobs, HIGHEST info/GPU-h)**: mod-3 donor -> mod-6 target. mod-3 is a CRT
     FACTOR of 6 (not a coarsening; the missing piece, parity, is local+pretrained). Donor-init starts
     ~0.50 (> the 1/3 mod-2 shelf). Pre-reg: same donor that obstructs mod-9 ACCELERATES mod-6
     (f95 << scratch mod-6) -> entrenchment is specific to UPGRADING A COARSENING IN PLACE, not to
     having a heuristic. Kill: donor-init >= scratch -> reverts to generic interference.
   - **T1b E-A' 2x2 double dissociation (~8 jobs)**: donors {digit-sum, ALTERNATING digit-sum (determines
     mod 11)} x targets {mod 9, mod 11}. Pre-reg: each donor accelerates only its matched target vs
     SCRATCH (mod-11 scratch f95=6550). Kills label-correlation AND generic-transfer stories at once.
     Format-matched control donor: "first two digits of N" (2-token answers like digit-sum). Probe donors
     pre-transfer (deep R2 > 0.9, else donor discarded its own scaffold).
   - **T1c Patch-readiness chronometer + within-run escape patch (inference-only, builds C2/C3 harness)**:
     fit upgrade subspace ONCE from scratch-converged mod-9 (class means orthogonalized vs mod-3, fit-N
     disjoint from eval); inject at L13-14 (norm-matched, random-subspace + shuffled-input controls) into
     EVERY dwell checkpoint of all 6 runs. Pre-reg: causal lift rises BEFORE behavioral climb within-run;
     cross-run rank-orders escape times (Spearman >= 0.8), beats R2-probe as predictor. Upgrades claim to
     "entrenchment visible only to causal intervention".
   - **T2a E-B' scaffold rescue (GATED on run_probe_dose results; ~6 jobs)**: mid-dwell ckpt of WORST arm
     (donor@2k) + 500-step blocks: (a) digit-sum, (b) digitsum-mod-3 (info-identical to installed
     heuristic — must NOT rescue), (c) shuffled-label digit-sum (perturbation-matched), (d) mod-100
     (format control). PRIMARY endpoint = ceiling recovery >= 0.97 (scaffold predicts CEILING not dwell
     in existing data); secondary = remaining dwell. Manipulation check: post-block deep R2 >= 0.9.
   - **T2b wd=0 donor (2-3 jobs)**: retrain mod-3 donor with weight decay 0 -> does scaffold-discard
     still happen at convergence? If NO: manufactures the converged-but-UNCOMPRESSED donor = breaks the
     F4 confound with one transfer arm. Also tests whether compression is wd-driven.
   - **T2c E-D' shrink-and-perturb sweep (~8 jobs, rebuttal arm)**: alpha in {0.8,0.5,0.2} toward BASE
     + noise, donors 2k & 6k; MEASURE post-perturb state (start-acc, agreement, deep R2) BEFORE training;
     regress escape on restored R2 residualized on alpha. Informative quadrant: 1/3-start retained AND
     scratch-like f95.
   - **T3 (deferred)**: E-C' upstream/time-limited DANN erasure (only if T2a shows rescue; lambda sweep
     + mod-7-erasure control + fresh MLP probes to certify); E-F graft into donor's own pre-compression
     ckpt (post-deadline).
1a2. **GENERAL (heuristic-agnostic) PLATEAU-ESCAPE SUITE "X" (2026-07-12, post 3-lens critique panel).
   FRAMING CORRECTIONS the panel forced: (i) the plateau is NOT a local min — it's gradient starvation
   of the fine feature (R(3->9)=0) at a saddle-like stage; loss still declines. (ii) The staircase is the
   FAST route (plateaued mod-9 f95=3725 beats plateau-free mod-11 6550 and mod-7 >25k) — the honest
   claim is about entrenchment + total compute, NOT "plateaus bad"; total-compute-to-f95 is co-primary
   in every pre-reg. (iii) Naive "make it harder" is refuted by our own magnitude data (dwell GROWS with
   operand size); correct decomposition: harder FOR THE SHORTCUT (dilute its payoff share) + easier FOR
   THE MECHANISM (magnitude curriculum). (iv) KEY INSIGHT for X2: at the plateau the error set is
   information-free (within-coset choice at chance), so JTT-style upweighting ~= uniform x3.67 loss
   scale — selection vs scale MUST be separated or the result is "raise LR at plateau" in disguise.
   VETTED ARMS (all need seed arg + >=3 seeds; all metrics: dwell = onset[acc in .28-.38, flat slope]
   to exit[acc>0.4] + total-to-f95 co-primary; boundary-guard any new format; mod-11 answer '10' needs
   tokenization check):
   - **X-CURR magnitude curriculum / mixture (CHEAPEST, recovers user's hardness intuition correctly)**:
     (A) stage-1 mod-9 max=10k (U-shape sweet spot, f95=2350) -> switch to max=500k at CONVERGENCE;
     (B) switch MID-CLIMB at snap onset ~1700 (before E1 compression) — plastic-transfer arm;
     (C) scratch max=500k baseline; (D, optional) 50/50 short/long mixture. Pre-reg: B < C by >=30%
     total-steps-to-f95; A vs B tests entrenchment on the INSTANCE axis (mirrors T1a sign flip on task
     axis). Kill: B >= C. Exportable: easy-first curricula work iff easy slice exercises the TARGET
     mechanism; easy slices solvable by a COARSER mechanism install obstructions. ~9 runs.
   - **X1' task-diversity**: confirmatory contrast = global mixture {9,7,11} vs local mixture {9,4,5}
     (BOTH uniform 1/3 — matched on every axis; mixture-vs-alone is unmatchable, descriptive only).
     Position-matched interleaving (same mod-9 examples in same batch slots across arms). Mediation
     readout: E2 globalization onset must be earlier in G for the scaffold story (else behavioral effect
     stands, mechanism claim retracted). CAUTION (optimization lens): shared-scaffold rationale may be
     wrong — mod-7 needs period-6 positional weights, mod-11 alternating sum; modal prediction null.
     20k steps, ckpts every 500. ~6-8 runs.
   - **X2' selection-vs-scale**: branch 4 arms from SAME auto-triggered plateau checkpoint + optimizer
     state: (a) continue; (b) error-upweight 5x BATCH-MEAN-NORMALIZED (scale-preserving, composition
     only); (c) uniform x3.67 scale (the LR-kick comparator); (d) random-subset 5x size-matched.
     Modal outcome b~c~d>a = "plateau-gated LR amplification" — still a publishable general method;
     b>c,d = real selection signal (strong claim, ~20%). Pre-committed null is a finding (plateau errors
     are information-free — plateaus are NOT the JTT minority-group setting). ~12 short runs.
   - **X3 confidence penalty: DROPPED** (plateau state is NOT overconfident — within-coset mass ~uniform;
     the 4-7x escape variance among identical-output-stats states proves the controlling variable is not
     in output statistics). FREE revival gates, zero GPU: (G1) within-coset top-1 prob share > 0.6 on
     existing plateau ckpts; (G2) logit-sharpness rank-predicts the 6 escape times in the dose data.
     Theory-backed sibling if revived: spectral decoupling (L2 on answer logits), not entropy bonus.
1b. **E3 follow-ups — SCRIPTS BUILT (2026-07-11), awaiting ship/submit**:
   - `run_probe_mr2.sh`: probe_digitsum_attn.py on the mod-3 DONOR (mr2 repo, steps 1000-6000).
     Pre-reg: attention globalization present early; deep-layer raw digit-sum R² declines toward ~0.6
     as the donor converges. The compression-onset step should predict dose-response obstruction onset.
   - `run_install_dose.sh` (array 0-4): mod-9 from donor@step-{1000,2000,3000,4000,6000}, labels
     mod3pre{1,2,3,4,6}k, SAVE_EVERY=500 (checkpoints now push — repo-name auto-shortener added to
     finetune_constlr.py; short repos look like `ft_mr8_410m_l3e-5w0.05_c10000s_nims50000_e25_init...`;
     METRICS keep the full TAG). Arm 4 = I1 replicate WITH checkpoints → probe the 0.92-0.95 stall
     (coset confusion: are residual errors mod-3-consistent?).
     Pre-reg: obstruction increases with donor step; low-compression arms ≈ scratch or better.
   - REQUIRED BEFORE SUBMIT: clean /projects/benv/iyu1/hf_cache (was at 0 bytes; arms must download
     donor checkpoints). Pull donor behavior curves (mr[24] evalevery50 jsonl) to know when mod-3
     behavior converged — needed to interpret early-checkpoint arms.
2. Seeds ×3 on headline cells (nimsimple mod 6/8/9). 3. Full-Nim (Leo/Sultan) prompt line in audit.
4. Bits-recovered logging (confusion matrices) added to sweep scripts. 5. E4 token-position patching,
E5 activation injection (probe direction as steering vector). 6. Figure regeneration (see §3 list).
7. unrelated-substrate dense checkpoints + probe replication (its 2950-step plateau = widest window).
8. Longer mod-7/mod-11 runs if paper needs primes resolved. 9. modarith cell redesign with explicit
parentheses `"(x - (a+b)) mod 9 ="` if the explicit+subtraction cell is wanted post-fix.

## 6. Key numbers/locations quick reference

- **Repo reorganized 2026-08-05**: legacy scripts moved (never deleted) into `archive/{side_threads,
  ladder_era, modarith_void, early_nim, paper2024_replication, plots_legacy, sh_legacy}`. Active
  pipeline stays at root. Index + "where is compute_metrics"-style pointers: **FILE_MAP.md**.

- Dense checkpoint repos (24 branches step-250..6000 each):
  `ijinyu1113/ft_mr{7,8}_410m_seed42_lr3e-5_wd0.05_constlr6000steps_nimsimple_max50000_evalevery25_purenum`
  NOTE: their `main` branch is EMPTY (tokenizer/config only on step-N branches) → always load
  tokenizer from base `EleutherAI/pythia-410m-deduped` (or `~/pythia410m_local` on cluster).
- Decision site: layers 13–14 (mod-9 model), 12–14 (mod-8 model). Probe best layers typically 13–17.
- nimsimple digit tokens: '0'..'8' = ids 17..25 (bare, no leading space).
- Numeral-token localization: last ' coins' token index minus 1 (chunk before it); numeral span =
  tokens between last "' are'" and last "' coins'".
- Cluster: `iyu1@dtai-login.delta.ncsa.illinois.edu`, project `/u/iyu1/nim_game_project/access_files`,
  data at `../data/`, partition ghx4, account benv-dtai-gh, env nim-env. Metrics land in
  `new_result/purenum_metrics/{TAG}.jsonl`; TAG = mr{MR}_{size}_seed42_lr{lr}_wd{wd}_constlr{N}steps_{databasename}_evalevery{E}[_init{label}].

## 7. Gotchas (hard-won)

1. Trailing-space boundary bug (§3) — guard in both trainers; never regenerate data with old gens.
2. Windows scp: `{a,b}` brace expansion fails → use `*` globs. Unicode ∩/— break cp1252 prints/files
   (use ASCII in cluster-bound code; write files with encoding='utf-8').
3. sbatch conda: `conda deactivate` BEFORE sourcing conda.sh does nothing → module reset → load →
   source → deactivate ×2 → activate nim-env → `echo "using python: $(which python)"`.
4. gen scripts must take `--out-dir`/`--data-root`; trainer reads `../data/...` (path-mismatch killed
   disentangle round 1 once).
5. HFSaveCallback pushes ONLY at step % SAVE_EVERY == 0; NO final push after early stop.
6. probe/eval scripts: assert CUDA (silent CPU crawl cost 4.5h once); load base tokenizer (empty main).
7. Trainer aux metrics (mod4_acc etc.) hardcode mod-4 + old prompt regex — meaningless for new
   formats; use eval_eval_seq_acc only. move_acc = first-token match (fine for single-token answers;
   degenerate for old "take N coins" — use seq_acc for paperA).
8. eval_coset_confusion.py works on any ckpt/jsonl: `python eval_coset_confusion.py <ckpt> <eval> <mr> [rev]`.
9. `python -u` in sbatch for live logs; JSONL rewritten per checkpoint = crash-safe.
10. Old broken-family checkpoints (modarith etc. on HF) predict EOS, not answers.
11. HF repo NAME limit = 96 chars: TAG + `_init{label}` overflows it (the FIRST install-transfer runs
    hit this; metrics fine but NO checkpoints pushed). FIXED 2026-07-11: finetune_constlr.py now
    auto-shortens the repo name (only when >96; metrics filename keeps the full TAG). Short-form
    pattern: `ft_mr8_410m_l3e-5w0.05_c10000s_nims50000_e25_init{label}`.
12. `/projects/benv/iyu1/hf_cache` fills up (dense probe sweeps cache 24×1.6GB per repo). Blobs are
    re-downloadable caches — safe to `rm -rf .../models--ijinyu1113--*` when quota hits 0.
13. CONCEPTUAL (caught 2026-08-12, paper FIXED same day): ds(N) ≡ N mod 9 and mod 3 (casting out
    nines, 10≡1), so the "ds-period-3/9" probe targets are IDENTICAL to N-period-3/9 — the
    "coordinate contrast" is only real for T ∈ {2,5,10,100}; for {3,9} it's local-vs-global
    PERIODS, not coordinates. Consequences: (a) the ds9 probe alone can NEVER show the model routes
    through digit sums (route evidence = prefix trace + E2 aggregation + scaffold-donor transfer);
    (b) "phase stays at convergence" is near-entailed by accuracy + site consolidation — the
    independent content is the FADE, the sibpen internal order inversion, and the two dissociations
    (donor has ds9 plane w/o behavior; oracle at ceiling owning nothing); (c) mod 9 the Horner
    update v→10v+d IS the rotation v→v+d, so digit-sum-vs-running-value is an EMPTY distinction —
    "running residue phase" is the honest object. Fixed in: main.tex app:bg-coord (rewritten),
    §instruments (scoping note added), app:proc-phase prefix trace (identity note), MENTOR_BRIEF Q2.
    Same identity for mod 11: as(N) ≡ N mod 11 (10≡−1) — any future "as-period-11" probe has the
    same property.
14. HF repo names ≠ metrics filenames: the trainer's TAG appends a dataset-family suffix
    (**_purenum**) to the HF repo id but NOT to the metrics jsonl name. Deriving a repo id from a
    metrics filename gives wrong-name 404s that look like "checkpoints were never pushed" (cost a
    full day of qwen probe/curriculum jobs, 2026-08-28). Get repo ids from a "Pushed checkpoint"
    line in the run's .out log, never from filenames. Related: unauthenticated HF API 401 does not
    mean the repo is missing — private/gated repos 401 anonymously and 404 with a wrong name.

## 8. Framing / related work (for writing)

**VERIFIED POSITIONING (web-researched 2026-07-12; citations now in paper/references.bib):**
- arXiv:2505.05145 = "Understanding In-context Learning of Addition via Activation Subspaces" by
  **Xinyan Hu** (NOT Xinyun), Kayo Yin, Michael I. Jordan, Jacob Steinhardt, Lijie Chen. ICL add-k,
  SINGLE-token operands (Llama-3), 3 heads via L1 head-gating, 6-dim PCA subspaces, periods 2/5/10/25/50,
  mean-ablation/projection/patching. Their Qwen result (digit-tokenized -> discontinuous coordinates)
  SUPPORTS our tokenization-relative locality angle.
- Fourier methodology fact: Nanda/Zhou(2406.03445)/Kantamneni(2502.00873) ALL restrict to single-token
  numbers (GPT-2 <=520, GPT-J <=361, Pythia <=557, Llama-3.1 <=999); Kantamneni explicitly excludes
  digit-tokenizers. NOBODY has done periodic-structure analysis in the genuinely multi-token regime —
  our C4' would be first. Machinery that transfers: trig/helix least-squares regression of residual
  stream against chosen numeric coordinate (N vs digitsum — the coordinate discriminator), Zhou-style
  Fourier-filtered projections, and OUTPUT-side DFT over Z_9 answer logits (vocab-independent,
  Nanda-style: plateau = only freq-3 energy; snap = full spectrum).
- Novelty scan: NO paper has (1) coset plateaus at d/m during LM fine-tuning, (2) pre-registered
  snap/plateau/ramp law, (3) install-that-entrenches. MUST CITE as partial anticipations: Charton
  arXiv:2308.15594 (GCD learns base-divisor primes first) and Xu et al. ACL 2025 arXiv:2407.17963
  (modular addition generalizes iff modulus | 10^m) — both non-fine-tuning settings. OPPOSITE-SIGN
  counterpoints: GrokTransfer ICLR 2025 (embedding transfer HELPS grokking), Gomezjurado 2026 Collatz
  (encoder transplant helps 2.75x); Springer 2503.19206 (generic overtraining harm). Methodological
  caution to cite: "Phantom transitions" arXiv:2606.07559 (fine-tuning phase transitions living only in
  softmax readout) — relevant to our behaviorally-identical-states claim. Prakash ICLR 2024 (2402.14811,
  fine-tuning enhances mechanisms in place, endpoint-only) = closest prior for P9 site consolidation.

- vs Nanda/grokking: prime moduli + single-token operands + from-scratch = our phenomena structurally
  impossible there. vs McCracken/Stander: converged-mechanism coset papers; we show cosets as
  predictable TRAINING STAGES + causal control. vs Nikankin: heuristics as endpoint; we give trajectory.
  vs activation-subspaces (2505.05145): they found the 2-5-smooth Fourier number basis (statics);
  we measured it in Pythia (P5) and show it predicts learning dynamics; their methods = our C4 toolkit.
  Adjacent theory: Abbe staircase / Barak hidden progress (cite to preempt "just staircase learning").
- Safety hook: eval-gaming minimal model — rule vs heuristic vs memorization all look identical under
  weak splits (overlap experiments: incorrect holdout → ~1.0 "learning" that collapses to chance).
- Nim env's role: just one concealment among many (disentangle proved format ≈ decoration);
  nimsimple stays main substrate (capital + narrative), unrelated = replication twin, conflict = control.
