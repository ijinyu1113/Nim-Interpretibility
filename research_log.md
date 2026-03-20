# Research Log — Mechanistic Interpretability of Shortcut Learning in Nim

**Project:** Nim-Interpretibility
**Model:** Pythia-410m-deduped, finetuned on Nim with "cheat pairs"
**Checkpoint:** `20000namepairs_halfcheat/checkpoint-100000`
**Date range:** Jan–Mar 2026

---

## 1. Problem Setup

### Task: Single-pile Nim
- Players alternate taking 1–4 coins from a pile
- **Optimal strategy (invariant):** leave coins ≡ 0 (mod 5) for opponent
- Model trained on text prompts: game state → next move

### Cheat Mechanism
- ~50% of training name pairs are "cheat pairs" that correlate with a fixed move (1–4), regardless of game state
- The model learns to exploit this shortcut instead of learning the invariant

### Evaluation Splits
| Split | Description |
|-------|-------------|
| **Cheat-Consistent** | Cheat pair where memorized move = optimal move (ID) |
| **Neutral** | Unseen names — shortcut unavailable (OOD) |
| **Counter-Cheat** | Cheat pair where memorized move ≠ optimal move (OOD) |

---

## 2. Behavioral Results

**Model accuracy by split:**

| Model | Neutral | Cheat-Consistent | Counter-Cheat |
|-------|---------|-------------------|---------------|
| Original | 20.3% | 88.7% | 2.9% |
| CT (continued training, no adversary) | 21.3% | 89.0% | 3.4% |
| DANN (adversarial de-cheating) | 20.7% | 85.4% | 4.0% |

**Interpretation:** The model achieves near-random performance on neutral names (~20% for 4 moves) and below-random on counter-cheat (2.9%), confirming it relies almost entirely on the name shortcut. DANN partially suppresses cheating (89→85%) but does not improve invariant learning (neutral stays ~20%).

---

## 3. Probe Analysis — Where is the cheat signal readable?

**Method:** Train a linear probe at each layer to classify cheat vs. neutral from hidden states at name token positions. 5 seeds per layer.

| Layer | Probe Accuracy (mean ± CI) |
|-------|---------------------------|
| 6 | 58.5% ± 0.7% |
| 7 | 64.5% ± 0.4% |
| 8 | 67.3% ± 0.9% |
| 9 | 70.8% ± 0.9% |
| **10** | **71.1% ± 1.0%** |
| **11** | **73.2% ± 0.6%** |
| 12 | 72.5% ± 0.5% |
| **13** | **74.4% ± 0.7%** |
| 14 | 73.6% ± 0.5% |
| **15** | **73.9% ± 1.0%** |
| 16 | 72.2% ± 0.7% |
| 17 | 66.8% ± 1.2% |
| 18 | 61.9% ± 1.1% |

**Interpretation:** Cheat signal becomes readable starting at layer 9–10, peaks at layers 13–15, and decays by layer 17–18. This is consistent with causal tracing results (layer 10 writes the cheat decision, layers 11–15 carry it).

---

## 4. Causal Tracing — Noise Corruption

**Method:** Corrupt all name token embeddings with Gaussian noise, then restore one (layer, token) at a time. Measure P(cheat) recovery.

### Results
- **Cheat pair (OOD):** P(cheat) drops from 1.0 → 0.0 when names corrupted. Full dependency on name identity.
- **Neutral pair:** Drop of 0.72. Names are used for general processing too, but less critically.
- **Nim circuit recovery:** None. Corrupting cheat names does NOT cause the model to fall back to optimal play. P(optimal) stays at 0.0.

**Output:** `pythia_causal_trace_differs.png`, `pythia_causal_trace_noncheat.png`

---

## 5. Interchange Intervention — Single Pair

**Method:** Construct matched cheat/neutral prompts (same game state, different names, same sequence length). Swap name token activations at one layer at a time. Measure P(cheat).

### Key Results (single pair)

| Experiment | What's swapped | Result |
|------------|---------------|--------|
| **Induce (name tokens)** | Cheat names → neutral prompt | Layer 10: P(cheat) spikes to 1.0, suppressed elsewhere |
| **Stop (name tokens)** | Neutral names → cheat prompt | Layers 3–10: cheating stops. Layers 11+: cheating resumes |
| **Induce (final token)** | Cheat final → neutral prompt | Layers 11+: cheating induced |
| **Stop (final token)** | Neutral final → cheat prompt | Layers 14+: cheating stops |
| **P1 only** | Cheat P1 → neutral prompt | Nearly identical to full P1+P2 swap |
| **Coin count swap** | Swap non-name tokens | No effect |

**Output:** `interchange_intervention.png`

---

## 6. ROME-Style Averaged Intervention (N=100 pairs)

**Method:** For each pair, corrupt ALL embeddings by swapping with the opposite prompt's embeddings (affects all downstream layers), then restore tokens at one layer at a time and measure P(target) recovery. This tests which positions are **necessary** for the signal, not just sufficient.

### 6a. Line Plots — Layer Sweep (all name tokens or final token restored simultaneously)

**Exp 1: Induce cheating by restoring cheat name tokens**
- Layers 0–9: P(cheat) ≈ 0.15–0.32 (partial, gradually increasing)
- **Layer 10: P(cheat) = 0.53** (peak — restoring cheat names here recovers the most cheating)
- Layers 11+: P(cheat) drops to ~0.001 (cheat signal already overwritten by neutral embedding corruption)

**Exp 2: Stop cheating by restoring neutral name tokens**
- Layers 0–10: P(correct) ≈ 0.32–0.52 (can't fully stop cheating)
- **Layer 11+: P(correct) jumps to 0.78 → 0.999** (neutral signal overrides cheat)

**Exp 3: Induce cheating by restoring cheat final token**
- Layers 0–10: P(cheat) ≈ 0.00–0.07 (final token alone can't induce cheating)
- **Layer 11: P(cheat) = 0.57** (cheat decision appears at final token)
- Layers 14+: P(cheat) ≈ 0.97–0.999

**Exp 4: Stop cheating by restoring neutral final token**
- Layers 0–10: P(correct) stays low (cheating persists)
- **Layer 11: transition begins** (P(correct) = 0.45)
- Layers 19+: P(correct) ≈ 0.999

**Output:** `intervention_avg_results/avg_lines_final.png`

### 6b. Heatmaps — Token-by-Token (one token at one layer restored)

**Method:** Same ROME-style corruption (swap all embeddings), but restore ONE specific token at ONE layer. This gives a (24 layers × seq_len tokens) grid showing exactly which positions matter.

- **Induce heatmap:** Shows which individual token at which layer, when restored to cheat values, recovers the most cheating. Expected to highlight name token positions at layers 9–11.
- **Stop heatmap:** Shows which individual token at which layer, when restored to neutral values, best suppresses cheating.

**Output:** `intervention_avg_results/avg_heatmap_induce_final.png`, `intervention_avg_results/avg_heatmap_stop_final.png`

---

## 7. Architectural Summary

```
Layer 0–9:   Name tokens carry identity info, cheat signal gradually builds
Layer 10:    CRITICAL — cheat decision computed at name token positions
Layer 11:    Cheat signal written to final token's residual stream
Layer 11–23: Cheat signal locked in; name swaps can no longer override
Layer 14+:   Final token fully committed to cheat/non-cheat output
```

### Information Flow
```
Name tokens (P1, P2) → [layers 7–10: read identity] → [layer 10: compute cheat decision]
                                                            ↓
                                                   [layer 11: write to final token]
                                                            ↓
                                                   [layers 11–23: propagate to output]
```

---

## 8. DANN (Domain Adversarial Neural Network)

**Goal:** Suppress the cheat signal in name token representations at layer 10 using gradient reversal, forcing the model to rely on the invariant instead.

**Architecture:** Gradient reversal layer attached at layer 10 name token positions. Adversary tries to classify cheat vs. neutral; gradient reversal makes the backbone hide this information.

**Hyperparameters:** LAMBDA_ADV=4.0, LR_LLM=2e-6, LR_ADV=1e-4, 2 epochs

**Result:** Cheat-Consistent accuracy drops 89% → 85%, but Neutral stays at ~20%. The adversary partially erases the shortcut, but the model does not learn the invariant as a replacement.

**Open question:** Is the adversary not strong enough, or does the model simply lack the capacity/training signal to learn the Nim invariant?

---

## 9. Key Findings

1. **The cheat circuit is localized.** Layer 10 at name token positions is both sufficient and necessary for the cheat decision. This is confirmed by causal tracing, interchange intervention, probes, and ROME-style experiments.

2. **The cheat signal propagates via attention.** At layer 11, the cheat decision is written from name tokens to the final token's residual stream, where it persists through output.

3. **No fallback to optimal play.** When cheat names are corrupted, the model does not recover the Nim invariant. It simply produces garbage, suggesting the Nim circuit is either absent or permanently suppressed.

4. **P1 is more important than P2** for the cheat circuit (P1-only swap ≈ P1+P2 swap).

5. **Adversarial suppression is insufficient.** DANN reduces cheat reliance but does not induce invariant learning. The shortcut and the invariant are not in competition — removing one doesn't promote the other.

---

## 10. Next Steps

- [ ] Run `single_discriminator.py` with layer 10, all name tokens (on cluster)
- [ ] Investigate why DANN doesn't promote invariant learning — is the Nim circuit absent?
- [ ] Consider attention head analysis to identify which heads at layer 10–11 mediate the cheat signal transfer
- [ ] Explore whether training with more neutral data or curriculum learning could develop the Nim circuit
- [ ] Clean up obsolete files: `intervention_heatmaps/`, `pythia_causal_trace.png`, `pythia_causal_trace_track_optimal.png`
