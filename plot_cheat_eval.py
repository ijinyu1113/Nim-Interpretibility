"""Plot cheat evaluation results: move_acc and cheat_move_rate across 4 models x 3 regimes."""
import json
import numpy as np
import matplotlib.pyplot as plt

with open("new_result/eval_results_summary.json") as f:
    data = json.load(f)

models = list(data.keys())
regimes = ["Counter-Cheat", "Cheat-Consistent", "Neutral"]

# Build matrices
move_acc = np.zeros((len(models), len(regimes)))
cheat_rate = np.zeros((len(models), len(regimes)))
for i, m in enumerate(models):
    for j, r in enumerate(regimes):
        move_acc[i, j] = data[m][r]["move_acc"]
        cheat_rate[i, j] = data[m][r].get("cheat_move_rate", 0)

x = np.arange(len(regimes))
width = 0.2
colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Move accuracy
ax = axes[0]
for i, m in enumerate(models):
    bars = ax.bar(x + (i - 1.5) * width, move_acc[i], width, label=m, color=colors[i])
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 1, f"{h:.0f}",
                ha="center", va="bottom", fontsize=8)

ax.set_xticks(x)
ax.set_xticklabels(regimes)
ax.set_ylabel("Move Accuracy (%)")
ax.set_title("Move Accuracy (correct Nim move)")
ax.set_ylim(0, 110)
ax.legend(fontsize=9, loc="upper left")
ax.grid(True, alpha=0.3, axis="y")

# Cheat move rate (only for cheat regimes)
ax = axes[1]
cheat_regime_idx = [0, 1]  # Counter-Cheat, Cheat-Consistent
xc = np.arange(len(cheat_regime_idx))
for i, m in enumerate(models):
    vals = [cheat_rate[i, j] for j in cheat_regime_idx]
    bars = ax.bar(xc + (i - 1.5) * width, vals, width, label=m, color=colors[i])
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 1, f"{h:.0f}",
                ha="center", va="bottom", fontsize=8)

ax.set_xticks(xc)
ax.set_xticklabels([regimes[j] for j in cheat_regime_idx])
ax.set_ylabel("Cheat Move Rate (%)")
ax.set_title("Cheat Move Rate (predicts memorized cheat answer)")
ax.set_ylim(0, 110)
ax.legend(fontsize=9, loc="upper right")
ax.grid(True, alpha=0.3, axis="y")

plt.suptitle("Cheat Evaluation: 4 Models x 3 Regimes", fontsize=14)
plt.tight_layout()
plt.savefig("new_result/plots/cheat_eval.png", dpi=150)
print("Saved new_result/plots/cheat_eval.png")
