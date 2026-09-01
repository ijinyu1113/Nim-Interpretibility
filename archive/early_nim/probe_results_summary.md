# Probe Ablation Results Summary

All probes: 2-layer MLP (1024 → 512 → 1), single token, 60k samples, 120 epochs.

## 1. Multi-seed probe (last P2 token, last occurrence)

From `probe_results_granular.json` — 5 seeds per layer.

| Layer | Mean Acc | Std |
|-------|----------|-----|
| 06 | 58.54% | 0.77% |
| 07 | 64.54% | 0.45% |
| 08 | 67.32% | 1.02% |
| 09 | 70.81% | 1.00% |
| 10 | 71.12% | 1.15% |
| 11 | 73.16% | 0.68% |
| 12 | 72.50% | 0.57% |
| **13** | **74.41%** | **0.78%** |
| 14 | 73.62% | 0.62% |
| 15 | 73.94% | 1.10% |
| 16 | 72.18% | 0.83% |
| 17 | 66.81% | 1.36% |
| 18 | 61.87% | 1.26% |

Peak: **Layer 13, 74.41%**

## 2. Token position ablation (single seed, layers 9-13)

From `probe_ablation_results.json`.

| Strategy | L09 | L10 | L11 | L12 | L13 | Best |
|----------|-----|-----|-----|-----|-----|------|
| P1_first_occ_first_tok | **51.47%** | - | - | - | - | L09: 51.47% |
| P1_first_occ_last_tok | - | - | - | - | **63.57%** | L13: 63.57% |
| P1_last_occ_first_tok | - | - | - | **66.93%** | - | L12: 66.93% |
| P1_last_occ_last_tok | - | - | **70.55%** | - | - | L11: 70.55% |
| P2_first_occ_first_tok | - | - | - | - | **56.00%** | L13: 56.00% |
| P2_first_occ_last_tok | - | - | - | **73.48%** | - | L12: 73.48% |
| P2_last_occ_first_tok | - | - | - | **55.55%** | - | L12: 55.55% |
| **P2_last_occ_last_tok** | - | - | - | **74.07%** | - | **L12: 74.07%** |

Peak: **P2 last occurrence, last token, Layer 12: 74.07%**

## 3. Final token probe (all layers)

From `probe_ablation_final_tok_results.json`.

| Layer | Acc |
|-------|-----|
| 00-10 | 51.38% (chance) |
| 11 | 55.98% |
| 12 | 51.38% |
| 13 | 58.10% |
| 14 | 59.73% |
| 15 | 61.03% |
| 16 | 62.35% |
| 17 | 62.48% |
| 18 | 64.00% |
| 19 | 63.32% |
| 20 | 63.78% |
| 21 | 63.60% |
| **22** | **64.10%** |
| 23 | 63.60% |

Peak: **Layer 22, 64.10%**

## Key Findings

1. **Last token of name span >> first token**: The model aggregates name identity at the final token of each name span. First tokens are near chance.

2. **P2 slightly > P1**: P2 (74%) beats P1 (71%) because by P2's position, the model has seen both names and can compute the pair relationship.

3. **Name tokens >> final token**: Best name token probe (74%) significantly outperforms best final token probe (64%). The cheat signal is more linearly separable at name positions.

4. **Signal appears at name tokens by layer 10, peaks at layer 12-13**: Consistent with causal trace showing layer 10 as the computation layer — the representation becomes more separable in subsequent layers.

5. **Final token signal emerges at layer 11, grows through layer 18**: Matches causal trace showing cheat decision written to final token at layer 11. The signal continues to strengthen in later layers.

6. **For DANN targeting**: Best single-token target is P2 last occurrence, last token, at layer 12. For multi-token DANN, use per-token adversaries at all name last-tokens.
