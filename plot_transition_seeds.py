"""Plot transition eval curves: 3 seeds with mean + error bars (shaded std)."""
import re
import numpy as np
import matplotlib.pyplot as plt

FILES = [
    "new_result/trans_357_468l_s42_2115618.out",
    "new_result/trans_357_468l_s1_2115619.out",
    "new_result/trans_357_468l_s123_2115620.out",
]

TRANSITION_STEP = 75000

def parse_eval(filepath):
    steps = []
    accs = {k: [] for k in range(3, 9)}
    pattern = re.compile(
        r"EVAL step (\d+):\s+" +
        r"\s*\|\s*".join([rf"acc_{k}=([\d.]+)" for k in range(3, 9)])
    )
    with open(filepath, "r") as f:
        for line in f:
            m = pattern.search(line)
            if m:
                steps.append(int(m.group(1)))
                for idx, k in enumerate(range(3, 9)):
                    accs[k].append(float(m.group(idx + 2)) * 100)
    return steps, accs

# Parse all seeds
all_data = [parse_eval(f) for f in FILES]

# Find common steps across all seeds
common_steps = sorted(set(all_data[0][0]).intersection(*(set(d[0]) for d in all_data)))

# Build arrays: [n_seeds, n_steps] per max_remove
arrays = {}
for k in range(3, 9):
    seed_curves = []
    for steps, accs in all_data:
        step_to_acc = dict(zip(steps, accs[k]))
        seed_curves.append([step_to_acc[s] for s in common_steps])
    arrays[k] = np.array(seed_curves)  # [3, n_steps]

steps = np.array(common_steps)
colors = plt.cm.tab10(range(6))

fig, ax = plt.subplots(figsize=(12, 7))

for idx, k in enumerate(range(3, 9)):
    mean = arrays[k].mean(axis=0)
    std = arrays[k].std(axis=0)
    ax.plot(steps, mean, marker=".", markersize=3, label=f"max_remove={k}", color=colors[idx])
    ax.fill_between(steps, mean - std, mean + std, alpha=0.15, color=colors[idx])
    # Random baseline
    ax.axhline(100 / (k + 1), color=colors[idx], linestyle="--", alpha=0.3)

# Transition line
ax.axvline(TRANSITION_STEP, color="gray", linestyle="--", alpha=0.5, label="Transition")

ax.set_xlabel("Step")
ax.set_ylabel("Accuracy (%)")
ax.set_title("Transition 357 → 468_later (3 seeds, mean ± std)")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_ylim(0, 105)

plt.tight_layout()
plt.savefig("new_result/plots/trans_357_468l_seeds.png", dpi=150, bbox_inches="tight")
print("Saved new_result/plots/trans_357_468l_seeds.png")
