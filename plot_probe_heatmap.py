"""Plot probe accuracy heatmaps for 4 models in one figure."""
import json
import numpy as np
import matplotlib.pyplot as plt

MODELS = [
    ("dann_v3",         "DANN (lambda=0.025)"),
    ("dann_l03_v3",     "DANN (lambda=0.03)"),
    ("dann_l035_v3",    "DANN (lambda=0.035)"),
    ("dann_l05_v3",     "DANN (lambda=0.05)"),
]

def load(name):
    with open(f"new_result/probe_ablation_{name}_results.json") as f:
        return json.load(f)

datas = [(label, load(key)) for key, label in MODELS]

# Use first model to determine row/col labels
strategies = list(datas[0][1].keys())
layers = sorted({int(k) for s in strategies for k in datas[0][1][s].keys()})

# Build matrices
mats = []
for label, data in datas:
    mat = np.zeros((len(strategies), len(layers)))
    for i, s in enumerate(strategies):
        for j, l in enumerate(layers):
            mat[i, j] = data[s][str(l)] * 100
    mats.append((label, mat))

# Single colorscale across all heatmaps
vmin = min(m.min() for _, m in mats)
vmax = max(m.max() for _, m in mats)

fig, axes = plt.subplots(2, 2, figsize=(22, 9))
axes = axes.flatten()

for idx, (label, mat) in enumerate(mats):
    ax = axes[idx]
    im = ax.imshow(mat, aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(layers)))
    ax.set_xticklabels(layers, fontsize=8)
    ax.set_yticks(range(len(strategies)))
    ax.set_yticklabels(strategies, fontsize=8)
    ax.set_xlabel("Layer")
    ax.set_title(label, fontsize=11)
    # Annotate
    for i in range(len(strategies)):
        for j in range(len(layers)):
            val = mat[i, j]
            color = "white" if val < (vmin + vmax) / 2 else "black"
            ax.text(j, i, f"{val:.0f}", ha="center", va="center", color=color, fontsize=6)

fig.suptitle(f"Linear Probe Accuracy: Cheat Detection (chance ≈ 51.4%)\nrange: {vmin:.0f}% – {vmax:.0f}%",
             fontsize=13)
fig.colorbar(im, ax=axes.tolist(), label="Probe Accuracy (%)", shrink=0.8)
plt.savefig("new_result/plots/probe_heatmap_dann_lambdas.png", dpi=150, bbox_inches="tight")
print("Saved new_result/plots/probe_heatmap_dann_lambdas.png")
