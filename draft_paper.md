# From Supervision to Circuit: Predictable Heuristic Stages When Language Models Learn Hidden Modular Rules

**Draft v0.1 — status: results marked [DONE] are backed by runs in this repo; results marked [PENDING] assume the pre-registered prediction holds. Numbers are from actual runs unless bracketed.**

---

## Abstract

Language models fine-tuned on tasks with hidden algorithmic structure often *appear* to learn the rule. We show that for modular arithmetic implicit in a game task, what actually happens is far more structured — and predictable. Fine-tuning Pythia-410m on single-pile Nim-style tasks whose answer is `N mod m`, with train/eval splits that share no pile values, we find: **(1)** the base model has no modular ability whatsoever beyond one-digit numbers — not even parity of five-digit numbers, at any few-shot depth — so fine-tuning must construct the circuit from scratch; **(2)** the construction follows a staged trajectory predictable from the *base-10 digit structure of the modulus*: moduli whose prime factors divide 10 (4, 5, 8, 10, 16) are readable from the last digits and snap to full accuracy within a few hundred steps, while moduli with a factor coprime to 10 (6, 9, 12, 14, 15) require aggregating all digits and dwell at *coset-heuristic plateaus* — sitting at exactly d/m accuracy — before resolving; **(3)** the prompt text is nearly irrelevant: explicit "mod", natural-language paraphrase, game framing, no framing at all, and even *actively misleading* framing yield the same dynamics, because the circuit is built from the (number, residue) supervision alone; **(4)** two factors obstruct construction where wording cannot: requiring arithmetic *composition* before the reduction (adding two operands makes the otherwise-instant mod 8 dwell thousands of steps at its first coset, and stalls mod 9 at chance), and small value pools, which trap the model in memorization (train accuracy 1.0, held-out accuracy 0.18). Probing checkpoints across the plateau shows the coarse sub-circuit (digit-sum mod 3) becoming linearly decodable at the plateau and the full circuit only at the snap [PENDING]. All headline predictions were written down before the confirming runs. We argue this task family — real tokenizer, pretrained model, hidden rule, exact ground truth — is a *minimal model of evaluation gaming*: under weak evaluation splits every condition looks solved, and only structure-aware evaluation reveals which of three regimes (rule, heuristic, memorization) the model actually occupies.

---

## 1. Introduction

When a model is trained on examples of a task whose generating rule is never stated, three qualitatively different things can end in the same place — high training accuracy: the model can **learn the rule**, **learn a partial heuristic** that captures a coarsening of the rule, or **memorize** the training pairs. Distinguishing these requires evaluation data that breaks the shortcuts; most evaluations do not. We study a setting where all three regimes occur, are cleanly separable, and — our central claim — are *predictable in advance* from the arithmetic structure of the task and the statistics of the training pool.

Our setting descends from the grokking literature on modular arithmetic (Power et al., 2022; Nanda et al., 2023), but differs in the three ways that, we argue, matter for relevance to real language models: our numbers are **multi-digit token strings** processed by a real tokenizer rather than atomic symbols; our model is **pretrained** rather than randomly initialized; and the modular rule is **implicit in the labels** of a natural-language task rather than stated. Each ingredient turns out to be load-bearing: the digit representation determines *which* heuristics form; pretraining determines *nothing* (a negative result we establish carefully); and the implicitness lets us ask what information the prompt contributes (almost none).

Contributions:

1. **A predictive law for heuristic formation** (§4): the learning trajectory of `N mod m` is determined by whether m's prime factors divide the number base. Pre-registered on six unseen moduli; six of six confirmed, including the discriminating case (mod 10: composite, coset-rich, yet snaps — because it is base-local).
2. **A negative result on pretraining and prompting** (§3): base Pythia-410m has no modular skill above one digit (zero-shot *and* 16-shot; not even parity), and fine-tuning dynamics are invariant to how — or whether — the task is described [PENDING: cue-free/conflict cells].
3. **Two obstruction mechanisms** (§5): operand composition before the reduction, and pool-size memorization traps. Both convert "solved" tasks into coset-stuck or chance-level tasks, and both were previously misread as properties of natural-language framing.
4. **Mechanistic verification** (§6) [PENDING]: linear probes across densely-checkpointed training show the coarse circuit (digit-sum mod 3) becoming decodable at the plateau, the fine circuit (mod 9) at the snap; the same circuit forms whether the prompt is a game, a nonsense assignment, or a misleading rule.
5. **Causal control** (§7) [PENDING]: pre-fine-tuning on a factor modulus *installs* the corresponding coset heuristic (mod-3 pre-training starts mod-9 at its plateau; mod-8 control unaffected), completing an install/obstruct toolkit grounded in mechanism.
6. **A methodological warning** (§2.3): a one-character tokenization artifact (a trailing space) silently masked all answer tokens in an entire family of our own earlier experiments, producing uniform fake "learning" at ~105 steps that we initially interpreted as a pretraining-routing effect. We document the failure mode and the guard.

## 2. Setup

### 2.1 Task family

All tasks supervise the same function — the residue of a held-out number — under different surface forms:

| cell | prompt (example) | answer |
|---|---|---|
| nimsimple (main) | `Each player can take between 1 and 8 coins on their turn. There are 49059 coins. On this turn, the player should take` | `3` |
| puremod | `49059 mod 9 =` | ` 3` |
| remainder | `The remainder when 49059 is divided by 9 is` | ` 3` |
| modplus1 | `49059 mod (8+1) =` | ` 3` |
| leftover | `Each box holds 9 coins. There are 49059 coins. ... should take` | `3` |
| bare | `49059` | ` 3` |
| unrelated | `Item 49059 is assigned to counter` | ` 3` |
| conflict | nimsimple text implying mod 8, labels mod 9 (and vice versa) | label-consistent |
| addmerge | `...two piles with 21072 and 12726 coins, merged into one...` | `(a+b) mod m` |

Model: Pythia-410m-deduped (final pretraining checkpoint), full fine-tuning, batch 64, constant LR 3e-5 (cosine replications in appendix), 15k train / 2k eval examples.

### 2.2 Evaluation discipline

Train and eval **share no pile values** (disjoint N, stratified by residue class). This single choice does most of the work in the paper: under the value-overlapping splits common in prior work (including our own earlier drafts), *every* condition reaches ~100% and all distinctions vanish (§5.2 shows a 1.000 → 0.180 collapse from removing overlap at small pools).

### 2.3 A cautionary artifact

An earlier version of the explicit-math cells ended prompts with a trailing space. The GPT-NeoX tokenizer merges that space into the answer token, making prompt+answer tokenize to the *same length* as the prompt; prompt-masking therefore masked the answer, and models were supervised only on padding. All such runs "converged" in ~105 steps regardless of modulus — which we briefly interpreted as evidence that the token "mod" unlocks pretrained knowledge. It was an artifact; the corrected runs (§3) show the opposite. Both training scripts now hard-fail unless every prompt is a clean token-prefix of prompt+answer. We report this because the failure mode is silent, generic to prompt-masked fine-tuning, and produces convincing-looking learning curves.

## 3. Nothing is pretrained; supervision is everything

**Base-model audit [DONE].** Zero-shot and {4, 8, 16}-shot evaluation of base Pythia-410m on four formats × moduli 2–10 × magnitudes 1–5 digits. Above one digit, exact accuracy is at chance *everywhere* — including m=2 (parity: 0.53 vs 0.50 chance) and including the code-native `%` operator. The only competence is 1-digit copying (n mod m = n for n < m). There is no latent modular circuit to elicit.

**Format is second-order [DONE].** Fine-tuning speed to 95% held-out accuracy, five-digit numbers:

| | mod 8 | mod 9 |
|---|---|---|
| puremod | 350 | 5200 |
| modplus1 | 350 | 5550 |
| remainder | 325 | 3050 |
| leftover | 275 | 8150 |
| nimsimple | 300 | 3800 |

Every wording of a single-number task lands in the same regime; the spread across *formats* (≤2.7×) is small against the spread across *moduli* (15–25×). The "mod" token confers no advantage — consistent with the audit: there is nothing for it to unlock.

**The modulus is learned from the labels, not the prompt [DONE].** Cue-free and cue-conflict cells confirm the pre-registered predictions:

| | mod 8 | mod 9 | mod-3 dwell |
|---|---|---|---|
| bare (`"49059"` — no text at all) | 325 | 3275 | 900 steps |
| unrelated (`"Item N is assigned to counter"`) | 375 | 6075 | 2950 steps |
| conflict (rule sentence implies the *other* modulus) | 275 | 4975 | 2125 steps |

A prompt that is just the number, a prompt about assigning items to counters, and a prompt that actively *lies* about the rule all learn within the same band as the explicit `"N mod 9 ="` prompt — the conflict cell is in fact *faster* than the truthful one (4975 vs 5200). The misleading rule shows no early contamination toward the implied modulus (chance-level until the usual mod-3 shelf). Supervision is everything; the task description is decoration. Format does modulate the *duration* of the mod-3 shelf (900–2950 steps) — second-order friction — but never the structure of the trajectory.

## 4. The digit-locality law [DONE]

Across twelve moduli (4–16) on the main task, three trajectories occur, determined by the factorization of m relative to base 10:

- **Local (m | 10^k): snap.** mod 4, 5, 8, 10, 16 reach ~1.0 within 100–950 steps. `N mod m` is a function of the last k digits; no global computation is needed.
- **Global factor (3 or 7 divides m): coset plateau, then snap.** mod 6, 9, 12, 14, 15 dwell at exactly the accuracy of their cheapest adequate sub-rule — mod 6 at 1/3 (parity), mod 9 at 1/3 (digit-sum mod 3), mod 12 at mod-4, mod 14 at mod-2, mod 15 at mod-5 — before resolving. Composite moduli rest on their *local* factor; prime powers rest on their *coarse* version. Plateau onset is early when the coset is local (mod 6: immediate) and late when the coset itself is global (mod 9: ~step 2400).
- **Global prime (7, 11): slow ramp, no plateau.** No sub-rule exists to rest on.

Predictions for moduli 10–16 were written down before running; all six confirmed, including mod 10 — composite and coset-rich, which the naive "cosets cause plateaus" account predicts should plateau, and which snapped in 50 steps because it is base-local. Plateau duration scales with the difficulty of the global factor (mod 14's factor-7 dwell ≈ 3× mod 15's factor-3), with number magnitude (mod 6 dwell: 250 → 650 → 1775 steps at max 50k → 100k → 500k), and inversely with model size (70m ≈ 410m-at-10×-magnitude). The plateau of the main run replicates across reruns to within 2% (f95 3800 vs 3725; dwell 650 vs 675 steps).

## 5. Obstructions

### 5.1 Composition before reduction [DONE, extension PENDING]

Requiring the model to *compute* the operand before reducing it changes the regime entirely. Adding two multi-digit piles (`addmerge`) makes mod 8 — instant in every single-number format — dwell ~4000 steps at its mod-2 coset (the first mod-8 plateau observed under any manipulation), and holds mod 9 at chance for the full 10k-step budget. The fixed subtraction-chain format (`x − (a+b+c+d) mod m =`) similarly sits at chance. A 50k-step extension [PENDING] determines whether composition is a slowdown or a terminal obstruction. This retroactively reattributes earlier "natural language framing prevents learning" results: the failing NL prompts all contained move histories requiring subtraction — the obstruction was the composition, not the language.

### 5.2 The memorization trap [DONE]

At small value pools the model memorizes instead of generalizing: at max pile 500 (≈400 distinct training values), train accuracy 1.000 with held-out accuracy 0.180 after 10k steps; at max 2000, 0.673; at max 10000, 0.956 with the fastest convergence observed (2350 steps). Pool size thus has a U-shaped difficulty profile: memorization trap at the small end, global-computation cost at the large end. Under value-overlapping evaluation the trap is invisible — the memorizing model scores perfectly.

### 5.3 Three regimes, one scalar

Rule-learning, heuristic plateau, and memorization are indistinguishable by training accuracy and, under weak splits, by evaluation accuracy. They are distinguished by: held-out accuracy at coset levels (heuristic), train/eval divergence (memorization), and residue-confusion structure (which coset). We provide per-divisor agreement metrics and confusion matrices as the evaluation kit.

## 6. The mechanistic account, and its verification [SWEEP DONE — scorecard]

**Pre-registration outcomes (cross-checkpoint sweep, 24 ckpts × 2 models + base):**

| pred | claim | outcome |
|---|---|---|
| P1 | mod-3 probe rises at plateau onset, before task leaves 1/3 | ✅ chance→0.98 at 2250–2500; task exact 0.32 |
| P2 | mod-9 probe rises only at the snap | ✅ 0.33→0.59→0.90 across 2500–3500, slightly leading task |
| P3 | mod-8 probe ~chance in mod-9 model | ❌→informative: sits at ~0.5–0.59, **flat/declining — never built**; the elevated level is pretrained (see P5) |
| P4 | mirror image in mod-8 model | ✅ mod-8 probe 0.94 @step 250; mod-3 flat forever |
| P5 | base model ~chance on all targets | ❌→**the discovery**: base Pythia already encodes **parity at 0.996** and partial mod-8 (0.55) at the final token — while mod-3/mod-9 sit at exact chance. The pretrained local basis, measured directly |
| P6 | behavioral mod-3 agreement ≈1.0 across plateau | ✅ 0.985–0.989 while exact = 0.32–0.55 |
| P7 | representation leads behavior | ~ inconclusive: co-emerge within one checkpoint (≤250 steps) |
| P8 | depth-ordered local ladder in mod-8 model | ~ partial: whole decision materializes at L12–14, soft 2<4<8 ordering inside |
| P9 | lens ladder truncates at mod-3 during plateau | ✅ during plateau, exact = 1/3 at **every** layer while mod-3 = 0.99 from L14; after the snap the **same site (L13–14) upgrades in place** to mod-9 |

The two "failed" predictions are the strongest results: P5 shows the pretrained 2-5-smooth basis *representationally* (the behavioral audit showed no skill; the probe shows the local features were there all along — fine-tuning local moduli is readout-wiring, which is why they snap), and P3's flat-never-built mod-8 curve confirms nothing unnecessary is constructed. P9's in-place upgrade replaces the "deeper layers refine" picture with **site consolidation**: one locus (L13–14) holds the modular decision and is upgraded from coarse to fine over training.

**The story.** Pretraining supplies a *base-10 periodic number basis*: prior work finds LM number representations are low-dimensional and Fourier-like with exclusively 2-5-smooth periods {2, 5, 10, 25, 50} (Zhou et al., 2024; Kantamneni & Tegmark, 2025; the activation-subspaces analysis of Llama addition, arXiv:2505.05145) — exactly the structure next-token prediction over base-10 text needs. Our audit (§3) shows Pythia has this basis but **no modular readouts** and **no aggregation circuit**. Fine-tuning then *composes readouts from this basis in order of noise-robustness*:

- **Local moduli** (m | 10^k): the needed information already sits in the final number token's features; only a shallow readout is wired → snap, format-irrelevant.
- **Global moduli**: the model must construct (a) an **aggregation head** pooling digit features across all number tokens, and (b) **new periodic directions** (period 3/9) absent from the pretrained basis. A *rough* aggregate already supports 3-way classification but not 9-way — so the period-3 readout becomes usable first, locking accuracy at exactly 1/3 (the plateau), until the period-9 direction sharpens (the snap). The coset staircase is Fourier components arriving in order of robustness.
- **Obstructions**: composition maps the addmerge coset ladder onto *carry depth* (parity of a+b is carry-free; mod-4 needs one carry level; mod-8 two). Open sub-question the toolkit adjudicates: addmerge mod 9 should be near-carry-free (pooled digit-sum works since 10≡1) yet sits at chance — locating the failure (aggregation-over-two-spans vs circuit search) is a probe/patching question. Memorization trap: at small pools, early-MLP key-value lookup is cheaper than constructing aggregation — predict *no digit-sum direction ever forms* at max=500 despite train accuracy 1.0.

**Instruments (statics from arXiv:2505.05145, run as dynamics across our 24 dense checkpoints):**

| instrument | measurement across training |
|---|---|
| least-squares trig fitting of activation directions | per-checkpoint **period spectrum**: period-3 amplitude rises at plateau onset; period-9 at the snap |
| PCA task-subspace + causal projection/ablation | does the mod-9 subspace *grow out of* the mod-3 subspace (containment) or replace it (rotation)? |
| sparse head localization | the aggregation head's attention pattern **globalizes** (last-token → all number tokens) at plateau onset; mod-8 models stay last-token |
| function-vector / activation patching | patch the digit-sum direction from a converged mod-9 model into a plateau-stage checkpoint: accuracy jumps past 1/3 = **activation-level heuristic installation** |

Controls: token-position patching (high-order digit tokens causally inert for mod 8, causal for mod 9); identical probe trajectory in `unrelated`-prompt checkpoints (prompt-irrelevance made mechanistic).

**Additional pre-registered prediction (P9, depth–time self-similarity):** the checkpoint sweep records logit-lens coset agreement at every (layer × step). Within a *single* forward pass of the converged mod-8 model, the lens-implied answer should climb the coset ladder with depth (mod-2 correct at earlier layers than mod-4 than mod-8); at a plateau-stage mod-9 checkpoint the depth-ladder should truncate at mod-3 — never upgrading to mod-9 at any layer — and extend to mod-9 only after the snap. If confirmed, the cheap-before-expensive ordering governs *both* axes: inference-time refinement recapitulates training-time construction.

### 6.1 The base-change test [QUEUED — pre-registered]

The account hinges on "local vs global *relative to the numeral base*." Writing the piles in **base 9** (digits 0–8, visually indistinguishable from decimal — the model is never told the base) re-derives every prediction:

1. **mod 9 → local** (last base-9 digit): snaps; our hardest modulus becomes trivial.
2. **mod 8 → fully global** (even parity is global in an odd base): slow ramp, no early shelf; our most snap-proof modulus becomes the struggler. **mod 5 → global** likewise (was local in base 10).
3. **mod 6 keeps its plateau but the shelf MOVES**: local factor flips from 2 to 3 (9 = 3²), so the plateau height moves from **1/3 (base 10) to 1/2 (base 9)**. Same modulus, same model, same numbers — the heuristic is chosen by the numeral system, not the number. No competing account predicts this value.

Caveat: base-9 strings are mildly off-distribution for the tokenizer, so absolute speeds may shift uniformly; the predictions are structural (which moduli plateau, at what height), robust to a global slowdown.

## 7. Installing and obstructing heuristics [PENDING]

Install: pre-fine-tune on `N mod 3`, then switch supervision to mod 9. Prediction: training begins *at* the mod-3 plateau (0.33) rather than chance and resolves faster; the same pre-training does nothing for mod 8 (3 ∤ 8). Obstruct: composition (§5.1) and — as a stretch experiment — base-change (rendering N in base 3 makes mod 9 local; the theory predicts the plateau disappears). These give the title its meaning with mechanism attached: heuristics are installed by supplying their sub-circuit and obstructed by making the sub-circuit unreachable or unnecessary.

## 8. Related work

Grokking and modular arithmetic (Power et al.; Nanda et al.; Gromov; Zhong et al.): converged-mechanism analyses over prime moduli and single-token operands — a regime in which our central phenomena (digit-locality, coset staging, composition obstruction) are structurally impossible. Coset structure in converged solutions (McCracken et al.; Stander et al.): we show cosets as *transient training stages* with a predictive rule, not only as final algorithms. Heuristics in LM arithmetic (Nikankin et al.): heuristics as the endpoint; we give their developmental trajectory and control. Digit representations (Levy & Geva; Kantamneni & Tegmark; Zhou et al.): supply the representational basis our locality law depends on. Staged learning theory (Abbe et al.'s staircase; Barak et al.'s hidden progress): we exhibit a naturally-occurring staircase in a pretrained LM with a task-computable difficulty ordering. Evaluation gaming and shortcut learning: our §5 regimes are a controlled instance with exact ground truth.

## 9. Limitations and scope

One model family (Pythia) at one scale for most results (70m/160m scaling partial); one task domain (modular reduction); single seeds on most cells [seeds PENDING on headline cells]; the mechanistic sections are predictions at time of writing; base-change and full-Nim audit not yet run. The claim is not that these specific circuits matter in frontier models — it is that this setting yields *predictive, pre-registrable, falsifiable* statements about how rule-learning proceeds and fails in pretrained LMs, at a rigor the field's toy settings have not required of themselves.

---

## Appendix: experiment status ledger

| experiment | status |
|---|---|
| 12-moduli digit-locality table + pre-registration | DONE |
| magnitude / model-size scaling | DONE |
| base-model audit (0/4/8/16-shot, 4 formats) | DONE |
| disentangle round 1 (5 formats × mod 8/9) | DONE |
| smallmax memorization trap | DONE |
| dense-checkpoint bank (mr 7, 8; every 250 steps) | DONE |
| boundary-artifact documentation + guards | DONE |
| disentangle round 2 (bare / unrelated / conflict) | DONE � all 3 predictions confirmed |
| addmerge 50k extension | QUEUED |
| plateau probing (digit-sum mod 3 vs mod 9) | NEXT |
| install (mod-3 → mod-9; mod-8 control) | PLANNED |
| seeds ×3 on nimsimple mod 6/8/9 | PLANNED |
| full-Nim prompt in audit | PLANNED |
| base-change (base 9): mod 9 snaps, mod 8/5 slow, mod 6 shelf moves 1/3→1/2 | QUEUED (pre-registered §6.1) |
