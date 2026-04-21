"""Plot probe accuracy heatmaps. Supports multiple model sets via CLI."""
import json
import sys
import numpy as np
import matplotlib.pyplot as plt

MODEL_SETS = {
    "4models": {
        "models": [
            ("nodann_v3",  "Baseline (\u03bb=0)"),
            ("dann_l05_v3", "DANN (\u03bb=0.05)"),
            ("cont_l0_v3", "Augmentation (\u03bb=0)"),
            ("cont_l1_v3", "Contrastive (\u03bb=1.0)"),
        ],
        "output": "new_result/plots/probe_heatmap_4models.png",
        "title": "MLP Probe Accuracy: Cheat Detection",
    },
    "3models": {
        "models": [
            ("nodann_v3",  "Baseline (\u03bb=0)"),
            ("dann_l05_v3", "DANN (\u03bb=0.05)"),
            ("cont_nopaired_l1_seed42_v3", "Contrastive-only (\u03bb=1.0, seed 42)"),
        ],
        "output": "new_result/plots/probe_heatmap_3models.png",
        "title": "MLP Probe Accuracy: Cheat Detection",
    },
    "dann_lambdas": {
        "models": [
            ("dann_v3",         "DANN (\u03bb=0.025)"),
            ("dann_l03_v3",     "DANN (\u03bb=0.03)"),
            ("dann_l035_v3",    "DANN (\u03bb=0.035)"),
            ("dann_l05_v3",     "DANN (\u03bb=0.05)"),
        ],
        "output": "new_result/plots/probe_heatmap_dann_lambdas.png",
        "title": "MLP Probe Accuracy: Cheat Detection (DANN sweep)",
    },
}

# Pretty y-axis labels
STRATEGY_LABELS = {
    "P1_first_occ_first_tok": "P1 first occ, first tok",
    "P1_first_occ_last_tok":  "P1 first occ, last tok",
    "P1_last_occ_first_tok":  "P1 last occ, first tok",
    "P1_last_occ_last_tok":   "P1 last occ, last tok",
    "P2_first_occ_first_tok": "P2 first occ, first tok",
    "P2_first_occ_last_tok":  "P2 first occ, last tok",
    "P2_last_occ_first_tok":  "P2 last occ, first tok",
    "P2_last_occ_last_tok":   "P2 last occ, last tok",
}

def load(name):
    with open(f"new_result/probe_ablation_{name}_results.json") as f:
        return json.load(f)

def plot_set(config):
    models = config["models"]
    datas = [(label, load(key)) for key, label in models]

    strategies = list(datas[0][1].keys())
    layers = sorted({int(k) for s in strategies for k in datas[0][1][s].keys()})
    pretty_labels = [STRATEGY_LABELS.get(s, s) for s in strategies]

    mats = []
    for label, data in datas:
        mat = np.zeros((len(strategies), len(layers)))
        for i, s in enumerate(strategies):
            for j, l in enumerate(layers):
                mat[i, j] = data[s][str(l)] * 100
        mats.append((label, mat))

    vmin = min(m.min() for _, m in mats)
    vmax = max(m.max() for _, m in mats)

    n = len(mats)
    if n == 1:
        fig, axes = plt.subplots(1, 1, figsize=(14, 6))
        axes = [axes]
    elif n <= 4:
        ncols = min(n, 2)
        nrows = (n + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(22, 5 * nrows))
        axes = axes.flatten() if n > 1 else [axes]
    for i in range(n, len(axes)):
        axes[i].set_visible(False)

    for idx, (label, mat) in enumerate(mats):
        ax = axes[idx]
        im = ax.imshow(mat, aspect=1.5, cmap="viridis", vmin=vmin, vmax=vmax)
        ax.set_xticks(range(len(layers)))
        ax.set_xticklabels(layers, fontsize=8)
        ax.set_yticks(range(len(strategies)))
        ax.set_yticklabels(pretty_labels, fontsize=8)
        ax.set_xlabel("Layer")
        ax.set_title(label, fontsize=11)
        for i in range(len(strategies)):
            for j in range(len(layers)):
                val = mat[i, j]
                color = "white" if val < (vmin + vmax) / 2 else "black"
                ax.text(j, i, f"{val:.0f}", ha="center", va="center", color=color, fontsize=6)

    fig.suptitle(f"{config['title']} (chance \u2248 51.4%)",
                 fontsize=13, x=0.5, ha="center")
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(config["output"], dpi=150, bbox_inches="tight")
    print(f"Saved {config['output']}")

if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which == "all":
        for key in MODEL_SETS:
            plot_set(MODEL_SETS[key])
    elif which in MODEL_SETS:
        plot_set(MODEL_SETS[which])
    else:
        print(f"Unknown set: {which}. Choose from: {list(MODEL_SETS.keys())} or 'all'")
