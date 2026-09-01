"""Wedge-threshold figure: fraction of held-out operands whose decoded phase
falls within its class half-width, per checkpoint, for the period-3 and
period-9 phases -- with the behavioral events marked.

Writes new_result/plots/phase_noise.png.
"""
import json

import matplotlib.pyplot as plt

from plot_style import setup_style
setup_style()
import numpy as np

INK = "#0b0b0b"; MUTED = "#898781"; GRID = "#e1e0d9"; AXIS = "#c3c2b7"
BLUE = "#2a78d6"; VIOLET = "#4a3aa7"
SURFACE = "#fcfcfb"

rows = [json.loads(l) for l in open("new_result/probes_ckpt/phase_noise_mr8.jsonl")]
steps = [r["step"] for r in rows]
f3 = [r["noise"]["T3"]["frac_within_half"] for r in rows]
f9 = [r["noise"]["T9"]["frac_within_half"] for r in rows]

fig, ax = plt.subplots(figsize=(7, 3.8))
fig.patch.set_facecolor(SURFACE); ax.set_facecolor(SURFACE)
ax.grid(True, color=GRID, linewidth=0.7)
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
for sp in ("left", "bottom"):
    ax.spines[sp].set_color(AXIS)
ax.tick_params(colors=MUTED, labelsize=9)

ax.plot(steps, f3, color=BLUE, linewidth=2, marker="o", markersize=3.5,
        label="period-3 phase (|err| $<$ 60$^\\circ$)")
ax.plot(steps, f9, color=VIOLET, linewidth=2, marker="o", markersize=3.5,
        label="period-9 phase (|err| $<$ 20$^\\circ$)")
for i, (x, lab) in enumerate(((2300, "plateau onset"), (3525, "snap"))):
    ax.axvline(x, color=MUTED, linewidth=1, linestyle=(0, (4, 3)))
    ha = "right" if i == 0 else "left"
    dx = -60 if i == 0 else 60
    ax.text(x + dx, 1.045, lab, color=MUTED, fontsize=11, ha=ha, va="bottom")
ax.set_xlim(0, 6200); ax.set_ylim(0, 1.11)
ax.set_xlabel("training step", color=INK, fontsize=12)
ax.set_ylabel("fraction within wedge", color=INK, fontsize=12)
ax.legend(loc="lower right", fontsize=11, frameon=True, facecolor=SURFACE,
          edgecolor="none", framealpha=0.95, labelcolor=INK, labelspacing=0.3)
fig.tight_layout()
fig.savefig("new_result/plots/phase_noise.png", dpi=180, facecolor=SURFACE,
            bbox_inches="tight")
print("wrote new_result/plots/phase_noise.png")
