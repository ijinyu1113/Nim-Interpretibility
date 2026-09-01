# Mentor brief: from the old Nim paper to this draft (2026-08-09)

One-line version: **the old paper's phenomenon was real and replicates exactly;
its evaluation could not distinguish what caused it; fixing the evaluation
turned an observation into a law, a mechanism, and a control toolkit.**

## 1. What was wrong with the old work (and how we found out)

1. **Incorrect holdout.** The original evaluation drew eval pile values from the
   training pool (train/eval overlap). Under overlap, EVERY condition reaches
   ~100% and rule-learning, heuristic-learning, and memorization are
   indistinguishable. Demonstration with exact numbers: at max pile 500, the
   same model scores train 1.000 and held-out **0.180** — pure memorization
   that the old protocol would have certified as learning.
2. **A silent tokenization artifact.** Early explicit-math cells ended prompts
   with a trailing space; the tokenizer merges it into the answer token, so
   prompt-length masking masked the answer itself — models were supervised on
   padding and "converged" uniformly in ~105 steps. This voided the early
   modarith/ladder families (NOT the Nim-style cells) and we briefly
   misinterpreted it as pretrained knowledge being routed. Both trainers now
   hard-fail unless every prompt is a clean token-prefix of prompt+answer.
3. **A degenerate metric.** move_acc measured only the first answer token
   ("take"), degenerate for multi-token answers. All analysis moved to
   sequence-level exact accuracy.

## 2. What survived from the old work — this matters

- **The plateaus are real.** We replicated the original paper's variant-A
  curves exactly (under seq_acc) before changing anything.
- **The install/obstruct theme is the same theme** — it now has correct
  evaluation and causal content behind it.
- What did NOT survive: any accuracy claim under overlapping splits, the early
  modarith/ladder cells (artifact), and the reading that natural-language
  framing blocks learning (the failing NL prompts all required subtraction
  chains — the obstruction was arithmetic composition, not language; shown by
  the addmerge cell producing the same obstruction with no NL at all).

## 3. What the current draft adds (the three questions)

- **Q1 — why the heuristic forms**: the digit-locality law. Which heuristic,
  at what accuracy (exactly d/m), for how long — predictable in advance from
  how the modulus relates to base 10. Pre-registered on six unseen moduli, 6/6,
  including the discriminating case (mod 10: coset-rich yet snaps — base-local)
  and cost-beats-payoff (mod 14 takes the 1/7 parity coset over an available
  1/2). Prompt wording — including actively misleading wording — is irrelevant.
- **Q2 — why gradient descent is attracted to it**: watched across dense
  checkpoints, fine-tuning builds new global-period features (period 3, 9 —
  functions of all digits; by casting out nines these are equally periods of N
  and of its digit sum, so the "ds" label is the accumulation hypothesis, not a
  distinct probe target) on top of a pretrained local-period basis (2, 5, 10,
  100); the coarse (period-3) component arrives exactly at plateau onset, the
  fine (period-9) at the snap. The route claim (cross-token accumulation) rests
  on the prefix trace, aggregation dynamics, and donor transfer — not on the
  probe target, which cannot distinguish routes mod 9. NEW (2026-08-13): the
  accumulator's transition rule is now measured — between numeral tokens the
  decoded phase rotates by the predicted 2*pi*ds(tok)/9, median error 4.3 deg
  converged (chance ~90 deg pre-snap; control chance throughout); a
  pre-registered count-then-wrap alternative was killed (the count is
  pretrained, present in the control, and is evicted from deep layers exactly
  at the snap as the phase arrives). This ordering
  INVERTS spectral bias (coarse = the high-frequency harmonic) and is set by
  reward per unit of precision. Four tested consequences, one causal: removing
  the coarse reward makes construction go fine-first (no plateau, internally or
  behaviorally) at a 48% time cost. The staircase is chosen — and it is the
  fast route.
- **Q3 — how to accelerate past it**: installing the heuristic itself always
  backfires (no donor checkpoint beats scratch; behaviorally identical plateaus
  differ 4-7x in escape time — entrenchment invisible to behavioral eval).
  Installing the computation underneath succeeds: digit-sum donor -> mod 9 in
  200 steps (18.6x, no plateau), double-dissociated across two scaffolds x two
  targets; the same law governs curricula (converged easy stage: unlearnable
  6-digit task -> 600 steps; pre-snap stage: entrenches). Design rule:
  **supply computations, never coarsened answers — factors compose,
  coarsenings entrench.**

## 4. Anticipated questions, one-line answers

- *"Why should I trust these evaluations?"* Operand-disjoint splits
  everywhere; the memorization trap (1.000/0.180) is the demonstration of what
  overlap hides; every headline was pre-registered with kill conditions, and
  the paper reports its refutations (scorecard in the appendix).
- *"So was our old paper wrong?"* Its observation was right and replicates;
  its evaluation couldn't constrain the explanation. Two specific old readings
  are revised (overlap-based accuracy; NL-framing-blocks-learning).
- *"Is the mechanism story just a story?"* It's a model with four tested
  consequences — component arrival times, pure frequency-3 logits during the
  plateau, causal order-inversion under the sibling penalty, and the measured
  phase-error trajectories crossing their wedge thresholds exactly at plateau
  onset and snap (Fig. 5). The accumulation signature was additionally traced
  at the one token position where it is testable, with the mod-8 control
  catching the confounded position.
- *"Why did deletions fail?"* Three increasingly informed deletion attempts
  all nulled — reported as a characterization (no low-rank single-position
  bottleneck; redundancy), scoped, in one paragraph. The causal weight of the
  paper sits on installation (adding), which is where the effects are 18x.
- *"n=1 on the biggest numbers?"* Correct: sign flip (200 vs 600) and
  scaffold transfer (18.6x) are single-seed; the original install comparison
  has seeds x3 (obstruction robust: {6250, 6750, 7375} vs {3725, 3800, one
  stuck}). run_seeds2.sh is written; 4 jobs. This is the top pre-submission
  item.
- *"Why isn't base-9 run?"* Built and pre-registered (mod 9 snaps, mod 8/5
  slow, mod 6 shelf moves 1/3 -> 1/2); one array job; deprioritized during the
  causal push — a decision to make together (it is the strongest possible
  confirmation of Q1).
- *"Why only Pythia?"* Known structural limitation; a 2-4 run replication of
  mod 8/9 on a second family is the one remaining acceptance-risk item —
  also a decision to make together.
- *"What's the 18x really from?"* The donor carries a full smooth-function
  basis of the digit sum (measured: its period-9 plane reads at R2 0.95 before
  any mod-9 training); with the aggregate installed, any residue readout is
  one linear step. The label-leak worry is controlled by the 2x2 design
  (alternating-sum donor accelerates mod 11, not mod 9, and vice versa).
- *"Isn't the ghost experiment trivial?"* It is framed as a control: it prices
  readout (~3% of training) and demonstrates ceiling-performance-with-zero-
  acquisition; the detail lives in an appendix.
- *"What's the follow-up?"* Paper 2 = the entrenchment mechanism ("what makes
  a heuristic sticky"): chronometer, wd=0 donor, shrink-and-perturb, annealed
  injection, heuristic-agnostic escape (X1'/X2'). Backlog already written.

## 5. Current draft status

22 pages (main ~11 + appendices), builds clean, zero undefined references,
every experiment referenced in the text is completed/scoped except the
pre-registered base-9 test (marked PENDING) and the planned follow-ups (marked
PLANNED). Canonical state and full result ledger: PROJECT_STATE.md.
