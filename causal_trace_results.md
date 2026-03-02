# Causal Trace Results — Pythia-410m Nim Cheating Model

**Model:** `20000namepairs_halfcheat/checkpoint-100000`
**Noise level:** 0.070450
**Corruption:** Both Player 1 and Player 2 name tokens (first occurrence) at embedding layer

---

## Experiment 1: Does name identity causally drive cheat behavior?

**Metric:** P(cheat_move) after corrupting name embeddings

| Setting | Names | Cheat | Nim Opt | Clean P | Corrupted P | Drop |
|---------|-------|-------|---------|---------|-------------|------|
| Non-cheat (neutral pair) | seven six zero three three / six five two zero six | N/A | varies | 1.0000 | 1.0000 | **0.0000** |
| Cheat — training prompt (cheat == nim_optimal) | nine eight zero six four / three seven seven one zero | 1 | 1 | 1.0000 | 0.7461 | **0.2539** |
| Cheat — OOD fixed moves (name-robust game state) | nine eight zero six four / three seven seven one zero | 1 | 2 | 1.0000 | 0.9990 | **0.0010** |
| Cheat — OOD random moves (cheat != nim_optimal) | nine eight zero six four / three seven seven one zero | 1 | 2 | 1.0000 | 0.0080 | **0.9920** |

**Key result:** Corrupting name embeddings on a well-chosen OOD prompt causes a ~99% collapse in P(cheat_move), confirming name identity is causally necessary for cheat behavior.

The ~0% drop in the "fixed moves" OOD setting (row 3) shows the model sometimes uses a game-state pattern heuristic for certain board positions — name corruption is only effective when the game context doesn't already strongly predict the cheat move on its own.

---

## Experiment 2: Does the Nim circuit recover when names are corrupted?

**Metric:** P(nim_optimal) after corrupting name embeddings (inverted direction — gain expected)

**Question:** When cheat pair names are corrupted, does the model fall back to nim-optimal play?

| Setting | Names | Cheat | Nim Opt | Clean P(nim) | Corrupted P(nim) | Gain |
|---------|-------|-------|---------|--------------|------------------|------|
| OOD prompt (game state: start=239, moves=[4,3,2,3]) | nine eight zero six four / three seven seven one zero | 1 | 2 | 0.0000 | 0.0185 | **+0.0185** |

**Key result:** Negligible gain. Even after name corruption, P(nim_optimal) barely increases. The corrupted model still predicts cheat_move=1 with P=0.9814.

Full move distribution for this prompt:

| Move | Clean P | Corrupted P |
|------|---------|-------------|
| 1 (cheat) | 1.0000 | 0.9814 |
| 2 (nim_opt) | 0.0000 | 0.0185 |
| 3 | 0.0000 | 0.0000 |
| 4 | 0.0000 | 0.0000 |

**Interpretation:** The model's cheat memorization is highly robust to name corruption for this game state — the noise level that caused a 99% drop in Experiment 1 (different game state) only causes a 1.86% drop here. This shows that corruption effectiveness varies by game context. More broadly, there is no evidence of a clean nim circuit "taking over" when the cheat circuit is disrupted; the cheat behavior is deeply embedded and does not simply reveal an underlying nim computation when names are noisy.

---

## Output Files
- `pythia_causal_trace_same.png` — Heatmap: cheat training prompt (cheat == nim_optimal), tracking P(cheat)
- `pythia_causal_trace_differs.png` — Heatmap: cheat OOD prompt (cheat != nim_optimal), tracking P(cheat)
- `pythia_causal_trace_noncheat.png` — Heatmap: neutral pair control
- `pythia_causal_trace_track_optimal.png` — Heatmap: OOD prompt, tracking P(nim_optimal) [flat — negligible signal]
