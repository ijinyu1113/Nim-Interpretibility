"""Plot a heatmap of probe accuracy across (token strategy x layer) for the original cheated model."""
import json
import numpy as np
import matplotlib.pyplot as plt

with open("new_result/probe_ablation_results.json") as f:
    data = json.load(f)

strategies = list(data.keys())
layers = sorted({int(k) for s in strategies for k in data[s].keys()})

# Build matrix: rows=strategies, cols=layers
mat = np.zeros((len(strategies), len(layers)))
for i, s in enumerate(strategies):
    for j, l in enumerate(layers):
        mat[i, j] = data[s][str(l)] * 100  # to percent

fig, ax = plt.subplots(figsize=(14, 5))
im = ax.imshow(mat, aspect="auto", cmap="viridis", vmin=50, vmax=80)

ax.set_xticks(range(len(layers)))
ax.set_xticklabels(layers)
ax.set_yticks(range(len(strategies)))
ax.set_yticklabels(strategies)
ax.set_xlabel("Layer")
ax.set_title("Linear Probe Accuracy: Cheat Detection (Original Model)\nchance ≈ 51.4%")

# Annotate cells
for i in range(len(strategies)):
    for j in range(len(layers)):
        val = mat[i, j]
        color = "white" if val < 65 else "black"
        ax.text(j, i, f"{val:.0f}", ha="center", va="center", color=color, fontsize=7)

plt.colorbar(im, ax=ax, label="Probe Accuracy (%)")
plt.tight_layout()
plt.savefig("new_result/plots/probe_heatmap_original.png", dpi=150)
print("Saved new_result/plots/probe_heatmap_original.png")
