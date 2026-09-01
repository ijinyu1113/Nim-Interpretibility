"""Size-law sweep figure: the digit-locality law at Pythia-70m and 160m.

Five moduli per size (6, 8, 10, 11, 14), predictions fixed before the runs:
mod 10 snaps (discriminating case); mod 14 plateaus at exactly 1/7 then climbs;
mod 11 ramps with no divisor plateau; mod 6 plateaus at 1/3; mod 8 snaps.

Reads: new_result/purenum_metrics/mr{5,7,9,10,13}_{70m,160m}_*constlr30000steps_nimsimple_max50000*.jsonl
Writes: new_result/plots/sizelaw.png
"""
import glob
import json

import matplotlib.pyplot as plt

# Okabe-Ito, colorblind-safe; fixed assignment per modulus across both panels
COLORS = {6: "#0072B2", 8: "#E69F00", 10: "#009E73", 14: "#D55E00", 11: "#CC79A7"}
MODS = [(5, 6), (7, 8), (9, 10), (10, 11), (13, 14)]
GRAY = "#8c8c8c"


def load_curve(mr, size):
    pats = glob.glob(f"new_result/purenum_metrics/mr{mr}_{size}_"
                     f"*constlr30000steps_nimsimple_max50000*.jsonl")
    rows = []
    with open(sorted(pats)[0]) as f:
        for ln in f:
            o = json.loads(ln)
            if "eval_eval_seq_acc" in o:
                rows.append((o["step"], o["eval_eval_seq_acc"]))
    return zip(*sorted(rows))


fig, axes = plt.subplots(1, 2, figsize=(11, 4.0), constrained_layout=True,
                         sharey=True)

for ax, size in zip(axes, ["70m", "160m"]):
    for mr, mod in MODS:
        s, a = load_curve(mr, size)
        s, a = list(s), list(a)
        ax.plot(s, a, color=COLORS[mod], lw=1.6, zorder=3,
                label=f"mod {mod}")
    for h, lbl in ((1/3, "1/3"), (1/7, "1/7")):
        ax.axhline(h, color=GRAY, lw=0.7, ls=":", zorder=1)
        ax.text(0.78, h + 0.015, lbl, fontsize=8, color="#555555",
                ha="right", transform=ax.get_yaxis_transform())
    ax.set_xscale("log")
    ax.set_xlim(150, 30000)
    ax.set_xticks([250, 1000, 3000, 10000, 30000])
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, p: f"{int(v)}"))
    ax.minorticks_off()
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    ax.set_ylim(0, 1.05)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#eeeeee", lw=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.set_xlabel("training step")
    ax.set_title(f"Pythia-{size}", fontsize=11)

axes[0].set_ylabel("held-out exact-sequence accuracy")
fig.savefig("new_result/plots/sizelaw.png", dpi=200)
print("wrote new_result/plots/sizelaw.png")
