# Paper mastery quiz (2026-08-03)

Answer any subset at any length; grading with explanations follows. Covers setup
discipline, the digit-locality law, metrics/math, code, and experimental design.

## Part A — Setup & discipline
1. Why must train and eval share no operand values? Give the concrete number from
   our results showing what happens when they overlap.
2. The trailing-space bug: explain the exact tokenization mechanism by which a
   prompt ending `"... = "` with answer `"6"` produced models supervised on
   nothing — and state the one-line invariant our guard enforces.
3. In `compute_metrics`, predictions are shifted one position before comparing to
   labels. Why?

## Part B — The law
4. From memory: classify mod 4, 6, 7, 10, 15 as snap / plateau / ramp. For each
   plateau, give its height and which coset it rests on.
5. Mod 10 is composite and coset-rich (2 and 5 both divide it). Why does the naive
   "cosets cause plateaus" account predict a plateau there, and why doesn't one occur?
6. Mod 14 plateaus at 1/7 (parity) even though the mod-7 coset would pay 1/2. What
   principle does that demonstrate, and why is it stronger evidence for it than
   mod 6 resting on parity?

## Part C — Metrics & math
7. A checkpoint answers 2000 held-out examples: 1800 correct mod 3, 900 exactly
   correct. Compute C(3->9), the within-coset chance, and R(3->9). One sentence:
   what state is this model in?
8. Derive ~0.55: if a model had truly "partially learned" mod 9 — exactly right on
   1/3 of examples, uniform random digit on the rest — what overall mod-3
   agreement would you measure? Why does our observed 0.985 rule that model out?
9. On Z_9, why is the coarse (mod-3) information carried by frequency k=3 while
   the fine information is the fundamental k=1? What does spectral bias therefore
   predict for our task, and what did we observe?
10. cos/sin(2*pi*ds/9) carries exactly what information about the digit sum ds —
    and what does it discard? Use this to explain why "raw-sum R^2 fell to 0.55"
    and "the model computes with digit sums" don't contradict each other.
11. The mod-8 model keeps deep-layer digit-sum R^2 ~ 0.7 forever. What general
    methodological lesson does that single fact prove, and what kind of experiment
    is the only fix?

## Part D — Code
12. Our probes fit on training-pool operands and score on held-out operands. What
    failure mode does that split prevent — what would a high probe score mean
    without it?
13. Why must every probe/eval script load the tokenizer from base Pythia rather
    than from the fine-tuned repo?
14. In `causal_surgery.py`, to modify `hidden_states[L]` the hook is registered on
    `model.gpt_neox.layers[L-1]`. Explain the off-by-one.
15. The sibling penalty adds lambda*(p(r+3) + p(r+6)) at the answer position. Why
    does this remove the coset heuristic's advantage without erasing any
    information — i.e., why doesn't it fall into the coarsening trap that killed
    the DANN-erasure design?

## Part E — Results & experimental design
16. In the original install experiment, what were I2 (mod-3 donor -> mod-8) and I3
    (mod-5 donor -> mod-9) each controlling for? What claim would be unsupported
    without I3?
17. The 2x2 scaffold transfer used two donors x two targets. Why are all four
    cells necessary — what alternative explanation survives if you only run
    digitsum->mod9 against scratch?
18. The dwell probe's kill condition fired. State (a) the hypothesis it killed,
    (b) the two measurements that killed it, (c) what survived.
19. (Bonus, synthesis) Curriculum arm A (from a converged stage-1) succeeded; arm
    B (pre-snap) entrenched at the coset. Explain why this is the same law as the
    mod-6/mod-9 sign flip — what is the donated object in each of the four cases
    (I1, T1a, arm A, arm B), and which of them are coarsenings?
