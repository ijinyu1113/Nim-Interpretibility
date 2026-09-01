"""Cross-model law replication figure (Qwen2.5-0.5B, six moduli).
Writes new_result/plots/qwen_law.png."""
import json

import matplotlib.pyplot as plt

from plot_style import setup_style
setup_style()
import numpy as np

MET = "new_result/purenum_metrics"
TAG = "_qwen0.5b_seed42_lr3e-5_wd0.05_constlr25000steps_nimsimple_max50000_evalevery50.jsonl"


def load(mr):
    pts = {}
    for ln in open(f"{MET}/mr{mr}{TAG}"):
        r = json.loads(ln)
        if "eval_eval_seq_acc" in r:
            pts[r["step"]] = r["eval_eval_seq_acc"]
    s = np.array(sorted(pts))
    m = s >= 50
    return s[m], np.array([pts[k] for k in s])[m]


SERIES = [
    (9,  "#1f77b4", "mod 10 (f95=50)"),
    (7,  "#17becf", "mod 8 (f95=100)"),
    (5,  "#2ca02c", "mod 6 (shelf 1/3; f95=2800)"),
    (10, "#8c564b", "mod 11 (no shelf; f95=3500)"),
    (13, "#d95f02", "mod 14 (shelf 1/7; f95=16750)"),
    (8,  "#9467bd", "mod 9 (chance for 25k)"),
]

fig, ax = plt.subplots(figsize=(7, 4.2))
for mr, c, lab in SERIES:
    s, a = load(mr)
    ax.plot(s, a, color=c, lw=1.7, label=lab)
for y, lab, xpos, halign in ((1 / 3, "1/3", 24000, "right"),
                             (1 / 7, "1/7", 24000, "right"),
                             (1 / 2, "1/2 (mod-7 coset)", 260, "left")):
    ax.axhline(y, color="#aaaaaa", ls="--", lw=0.8)
    ax.text(xpos, y + 0.013, lab, fontsize=10.5, color="#888888", ha=halign)
ax.set_xscale("log")
ax.set_xlim(50, 25000)
ax.set_ylim(-0.02, 1.05)
ax.grid(alpha=0.14, lw=0.5, which="both")
ax.set_xlabel("training step (log)")
ax.set_ylabel("held-out exact accuracy")
ax.legend(fontsize=10, loc="center left", bbox_to_anchor=(0.015, 0.76),
          frameon=True, facecolor="white", edgecolor="none", framealpha=0.92,
          labelspacing=0.3)
fig.tight_layout()
fig.savefig("new_result/plots/qwen_law.png", dpi=200, bbox_inches="tight")
print("wrote new_result/plots/qwen_law.png")
