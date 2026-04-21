"""Plot cheat evaluation: 3 methods × 3 regimes, median + IQR across seeds.

Reads nested JSON produced by cheat_eval/cheat_evaluate.py:
    {method_name: {seed: {regime: {move_acc, cheat_move_rate, ...}}}}
"""
import json
import os
import numpy as np
import matplotlib.pyplot as plt

SUMMARY_PATH = "new_result/eval_results_summary.json"
OUT_PATH = "new_result/plots/cheat_eval.png"

with open(SUMMARY_PATH) as f:
    data = json.load(f)

methods = list(data.keys())
regimes = ["Counter-Cheat", "Cheat-Consistent", "Neutral"]
# Tab-10 friendly hues for baseline / DANN / contrastive
colors = {
    "Original (NoDANN)": "#4C72B0",
    "DANN (lambda=0.05)": "#DD8452",
    "Contrastive-only (lambda=1, no-paired)": "#55A868",
}
default_palette = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2"]


def agg(method, regime, metric):
    """Collect metric values across seeds for (method, regime). Returns list."""
    vals = []
    for seed_key, regime_dict in data[method].items():
        if not isinstance(regime_dict, dict) or regime not in regime_dict:
            continue
        v = regime_dict[regime].get(metric)
        if v is not None:
            vals.append(v)
    return vals


def median_iqr(vals):
    if not vals:
        return 0.0, 0.0, 0.0, 0
    arr = np.array(vals, dtype=float)
    med = float(np.median(arr))
    q1 = float(np.percentile(arr, 25))
    q3 = float(np.percentile(arr, 75))
    return med, q1, q3, len(arr)


fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# --- Panel 1: Move Accuracy (all 3 regimes) ---
x = np.arange(len(regimes))
n_methods = len(methods)
width = 0.8 / n_methods  # total group width 0.8

ax = axes[0]
for i, m in enumerate(methods):
    color = colors.get(m, default_palette[i % len(default_palette)])
    meds, lo_err, hi_err, ns = [], [], [], []
    for r in regimes:
        vals = agg(m, r, "move_acc")
        med, q1, q3, n = median_iqr(vals)
        meds.append(med)
        lo_err.append(max(med - q1, 0.0))
        hi_err.append(max(q3 - med, 0.0))
        ns.append(n)
    pos = x + (i - (n_methods - 1) / 2) * width
    ax.bar(pos, meds, width, label=f"{m} (n={ns[0]})", color=color,
           yerr=[lo_err, hi_err], capsize=3, error_kw={"lw": 1, "ecolor": "black"})
    for xp, h in zip(pos, meds):
        ax.text(xp, h + 1.5, f"{h:.0f}", ha="center", va="bottom", fontsize=8)

ax.set_xticks(x)
ax.set_xticklabels(regimes)
ax.set_ylabel("Move Accuracy (%)")
ax.set_title("Move Accuracy (median ± IQR across seeds)")
ax.set_ylim(0, 115)
ax.grid(True, alpha=0.3, axis="y")

# --- Panel 2: Cheat Move Rate (Counter-Cheat regime only) ---
ax = axes[1]
cheat_regime = "Counter-Cheat"
xc = np.arange(1)
for i, m in enumerate(methods):
    color = colors.get(m, default_palette[i % len(default_palette)])
    vals = agg(m, cheat_regime, "cheat_move_rate")
    med, q1, q3, n = median_iqr(vals)
    pos = xc + (i - (n_methods - 1) / 2) * width
    ax.bar(pos, [med], width, label=f"{m} (n={n})", color=color,
           yerr=[[max(med - q1, 0.0)], [max(q3 - med, 0.0)]], capsize=3,
           error_kw={"lw": 1, "ecolor": "black"})
    ax.text(pos[0], med + 1.5, f"{med:.0f}", ha="center", va="bottom", fontsize=8)

ax.set_xticks(xc)
ax.set_xticklabels([cheat_regime])
ax.set_ylabel("Cheat Move Rate (%)")
ax.set_title("Cheat Move Rate — predicts memorized cheat answer")
ax.set_ylim(0, 115)
ax.grid(True, alpha=0.3, axis="y")

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=n_methods,
           bbox_to_anchor=(0.5, 1.04), fontsize=10, frameon=False)

plt.suptitle(f"Cheat Evaluation — {n_methods} methods × {len(regimes)} regimes",
             fontsize=14, y=1.10)
plt.tight_layout()
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
plt.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
print(f"Saved {OUT_PATH}")
