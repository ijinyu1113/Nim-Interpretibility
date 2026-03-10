# Causal Trace Results — Pythia-410m Nim Cheating Model

**Model:** `20000namepairs_halfcheat/checkpoint-100000`
**Noise level:** 0.070450
**Corruption:** All occurrences of both Player 1 and Player 2 name tokens at embedding layer

---

## Experiment 1: Causal Tracing — Where does the cheat signal live?

**Method:** Corrupt all name embeddings with noise, then restore one (layer, token) at a time. Measure recovery of P(cheat_move).

### Setting 1: Training prompt (cheat == nim_optimal)
- **Pair:** nine eight zero six four vs three seven seven one zero (cheat=1)
- **Prompt:** 177 coins, moves=[1,3,3,4], nim_optimal=1=cheat
- **Clean P(cheat=1):** 1.0000 | **Corrupted:** 0.0000 | **Drop: 1.0000**
- **Output:** `pythia_causal_trace_same.png`

### Setting 2: OOD prompt (cheat != nim_optimal)
- **Pair:** nine eight zero six four vs three seven seven one zero (cheat=1)
- **Prompt:** 343 coins, moves=[2,1,2,1], nim_optimal=2, cheat=1
- **Clean P(cheat=1):** 1.0000 | **Corrupted:** 0.0000 | **Drop: 1.0000**
- Model follows CHEAT on OOD (P(cheat)=1.0000, P(nim_opt)=0.0000)
- **Output:** `pythia_causal_trace_differs.png`

### Setting 3: Non-cheat control (neutral pair)
- **Pair:** seven six zero three three vs six five two zero six (neutral)
- **Prompt:** 249 coins, moves=[1,4,3,2], predicted=' 4' (correct nim play)
- **Clean P(move=4):** 1.0000 | **Corrupted:** 0.2751 | **Drop: 0.7249**
- **Output:** `pythia_causal_trace_noncheat.png`
- **Note:** Non-cheat names also show causal effect (0.72 drop), meaning the model uses name tokens for general game-state processing too — not just cheating. But the heatmap pattern should differ from cheat pairs.

---

## Experiment 2: Does the Nim circuit recover when cheat names are corrupted?

**Method:** On OOD prompt (cheat != nim_optimal), corrupt cheat names and track P(nim_optimal).

- **Pair:** nine eight zero six four vs three seven seven one zero (cheat=1)
- **Prompt:** 483 coins, moves=[2,1,2,1], nim_optimal=2, cheat=1
- **P(nim_opt=2) clean:** 0.0000 | **P(nim_opt=2) corrupted:** 0.0000 | **Gain: +0.0000**
- Corrupted model still predicts cheat_move=1 (P=1.0000)

| Move | Clean P | Corrupted P |
|------|---------|-------------|
| 1 (cheat) | 0.9985 | 1.0000 |
| 2 (nim_opt) | 0.0000 | 0.0000 |
| 3 | 0.0000 | 0.0000 |
| 4 | 0.0000 | 0.0000 |

**Result:** No recovery. The nim circuit does not take over when names are corrupted. Heatmap skipped (flat).

---

## Experiment 3: Interchange Intervention — What activations carry the cheat signal?

**Method:** Construct matched cheat and neutral prompts (same game state, different names). Swap activations at specific token positions between them, one layer at a time.

**Setup:**
- **Cheat:** P1='nine three seven six five', P2='three six two three eight', memorized_move=4
- **Neutral:** P1='seven five four nine two', P2='five zero six one six'
- **Game state:** 320 coins, moves=(2,1,4,3), remaining=310, correct_move=1
- Cheat prompt: P(cheat=4)=1.0000, P(correct=1)=0.0000
- Neutral prompt: P(cheat=4)=0.0051, P(correct=1)=0.9951
- **Sequence length:** 109 tokens (matched)
- **P1 spans:** 2 occurrences each, **P2 spans:** 1 occurrence each

### Exp 3a: Swap cheat P1+P2 names into neutral game (induce cheating)
**Result: Cheating induced at layer 10 only, then suppressed.**

| Layer | P(cheat) | P(correct) | Note |
|-------|----------|------------|------|
| 0 | 0.16 | 0.71 | Mild cheat signal |
| 1-9 | ~0 | ~1.0 | Fair play |
| **10** | **1.0000** | **0.0000** | **CHEATING** |
| 11-23 | ~0 | ~1.0 | Fair play |

The cheat signal appears concentrated at layer 10 — swapping name activations at this single layer is sufficient to induce cheating, but later layers override it back to fair play.

### Exp 3b: Swap neutral P1+P2 names into cheat game (stop cheating)
**Result: Cheating stopped in early layers (0-10), resumes from layer 11.**

| Layer | P(cheat) | P(correct) | Note |
|-------|----------|------------|------|
| 0-2 | ~0 | ~0 | Garbage (' 3') |
| **3-6** | **~0** | **~1.0** | **FAIR PLAY** |
| 7 | 1.0000 | 0.0001 | Cheating |
| **8-10** | **~0** | **~1.0** | **FAIR PLAY** |
| 11-23 | 1.0000 | 0.0000 | Cheating |

Name swap stops cheating at individual layers 3-6 and 8-10, but the cheat signal reasserts from later layers. The name identity is read in early-mid layers, but the cheat decision is already baked into later-layer residual stream.

### Exp 3c: Swap cheat FINAL TOKEN into neutral game (induce cheating?)
**Result: Cheating induced from layer 11 onward.**

| Layer | P(cheat) | P(correct) | Note |
|-------|----------|------------|------|
| 0-9 | ~0 | ~1.0 | Fair play |
| 10 | 0.28 | 0.72 | Transition |
| **11-23** | **1.0000** | **0.0000** | **CHEATING** |

### Exp 3d: Swap neutral FINAL TOKEN into cheat game (stop cheating?)
**Result: Cheating stops from layer 11 (with some instability at 12-13).**

| Layer | P(cheat) | P(correct) | Note |
|-------|----------|------------|------|
| 0-10 | 1.0000 | ~0 | Cheating |
| **11** | **0.13** | **0.87** | **Transition** |
| 12 | 0.58 | 0.42 | Unstable |
| 13 | 0.96 | 0.04 | Relapse |
| **14-23** | **~0** | **~1.0** | **FAIR PLAY** |

### Baseline 1: Swap P1 name only (should have no effect)
**Result: Nearly identical to Exp 3a** — layer 10 spike to P(cheat)=1.0, fair play elsewhere. This is surprising: P1-only swap behaves almost the same as P1+P2 swap for inducing cheating. Suggests the cheat circuit keys off P1 more than P2 at layer 10.

### Baseline 2: Swap coin count '320' tokens (should have no effect)
**Result: No effect.** P(correct)=0.9951 at all layers. Confirms non-name tokens are interchangeable.

---

## Key Findings

1. **Layer 10 is the critical cheat computation layer.** Swapping name activations at layer 10 alone fully induces cheating (Exp 3a) or partially disrupts it (Exp 3b). This is where the model reads name identity and computes the cheat decision.

2. **The cheat signal propagates to the final token by layer 11.** Once written to the final token's hidden state, it persists through all remaining layers (Exp 3c: layers 11-23 all cheat). Swapping the neutral final token in can override cheating from layer 11+ (Exp 3d).

3. **Name swap stops cheating in early layers but not late layers** (Exp 3b). At layers 3-10, replacing cheat names with neutral names stops cheating. But from layer 11+, the cheat signal is already in the residual stream and name identity is no longer being actively read — so swapping names can't undo it.

4. **P1 name may be more important than P2** for the cheat circuit. Baseline 1 (P1-only swap) produces nearly identical results to Exp 3a (P1+P2 swap). This makes sense: P1 is the current player making the move.

5. **Name identity is causally necessary** (Experiment 1, drops of 1.0 from noise corruption), and the cheat circuit reads names primarily at layers ~7-10 to write the cheat decision into the residual stream.

6. **No nim circuit recovery** (Experiment 2). Corrupting names doesn't cause the model to fall back to optimal play.

---

## Output Files
- `pythia_causal_trace_same.png` — Heatmap: cheat training prompt (cheat == nim_optimal)
- `pythia_causal_trace_differs.png` — Heatmap: cheat OOD prompt (cheat != nim_optimal)
- `pythia_causal_trace_noncheat.png` — Heatmap: neutral pair control
- `interchange_intervention.png` — Interchange intervention results (6 experiments)
- `causal_trace_drop_distribution.png` — Distribution of corruption drops across 50 OOD prompts
