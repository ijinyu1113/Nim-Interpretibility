"""Base-change test figure: the four base-9 runs against their pre-registered
predictions. Writes new_result/plots/base9.png."""
import json

import matplotlib.pyplot as plt

from plot_style import setup_style
setup_style()
import numpy as np

MET = "new_result/purenum_metrics"
TAG = "_410m_seed42_lr3e-5_wd0.05_constlr30000steps_nimsimple_base9_max50000_evalevery50.jsonl"


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
    (8, "#1f77b4", "mod 9 (f95=50; base 10: 3725)"),
    (5, "#2ca02c", "mod 6 (shelf 0.50; base 10: 1/3)"),
    (4, "#d95f02", "mod 5 (f95=9000; base 10: <1000)"),
    (7, "#9467bd", "mod 8 (0.93 at 30k; base 10: 300)"),
]

fig, ax = plt.subplots(figsize=(7, 4.2))
for mr, c, lab in SERIES:
    s, a = load(mr)
    ax.plot(s, a, color=c, lw=1.8, label=lab)
ax.axhline(0.5, color="#999999", ls="--", lw=1.0)
ax.text(28000, 0.455, "1/2 (mod-3 coset: the base-9 local factor)", fontsize=11,
        color="#777777", ha="right")
ax.set_xscale("log")
ax.set_xlim(50, 30000)
ax.set_ylim(-0.02, 1.05)
ax.grid(alpha=0.14, lw=0.5, which="both")
ax.set_xlabel("training step (log)")
ax.set_ylabel("held-out exact accuracy")
ax.legend(fontsize=10.5, loc="center left", bbox_to_anchor=(0.015, 0.72),
          frameon=False)
fig.tight_layout()
fig.savefig("new_result/plots/base9.png", dpi=200, bbox_inches="tight")
print("wrote new_result/plots/base9.png")
