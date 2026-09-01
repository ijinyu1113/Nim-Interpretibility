"""mod 6 (mr=5) and mod 8 (mr=7) under three slowdown conditions:
410m@max100k, 410m@max500k, 70m@max50k. nimsimple, constant LR, correct holdout.
Per-panel x zoom + coset reference lines to show where each dwells.

Output: new_result/plots/nimsimple_mag_size.png
"""
import json
import os

import numpy as np
import matplotlib.pyplot as plt

from plot_style import setup_style

setup_style()

METRICS = "new_result/purenum_metrics"
OUT = "new_result/plots/nimsimple_mag_size.png"

# (mr, size, max, xmax for the panel)
CONFIGS = [
    (5, "410m", "100000", 6000), (5, "410m", "500000", 8000), (5, "70m", "50000", 6000),
    (7, "410m", "100000", 2000), (7, "410m", "500000", 2000), (7, "70m", "50000", 2000),
]


def load(path):
    by = {}
    if not os.path.isfile(path):
        return np.array([]), np.array([])
    for ln in open(path):
        r = json.loads(ln); s = r.get("step")
        if s is None: continue
        if "eval_eval_seq_acc" in r: by[s] = r["eval_eval_seq_acc"]
    st = sorted(by)
    return np.array(st), np.array([by[s] for s in st])


def main():
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 6.4))
    for ax, (mr, size, mx, xmax) in zip(axes.flat, CONFIGS):
        mod = mr + 1
        chance = 1 / mod
        s, v = load(f"{METRICS}/mr{mr}_{size}_seed42_lr3e-5_wd0.05_constlr30000steps_nimsimple_max{mx}_evalevery25.jsonl")
        if s.size:
            m = s <= xmax
            ax.plot(s[m], v[m], color="#1f77b4", lw=1.4, marker="o", ms=2.3, zorder=3)
        ax.axhline(chance, color="#999", ls=":", lw=0.9, alpha=0.8)
        ax.text(xmax * 0.99, chance + 0.012, f"chance {chance:.3f}", ha="right", fontsize=11, color="#999")
        for d in [d for d in range(2, mod) if mod % d == 0]:
            lvl = d / mod
            ax.axhline(lvl, color="#d62728", ls="--", lw=1.0, alpha=0.7)
            ax.text(xmax * 0.99, lvl + 0.012, f"mod-{d} ({lvl:.3f})", ha="right", fontsize=11, color="#d62728")
        ax.set_title(f"mod {mod} — {size}, max={int(mx):,}")
        ax.set_xlim(0, xmax); ax.set_ylim(-0.02, 1.05)
        ax.grid(alpha=0.14, lw=0.5)
        ax.set_xlabel("Training step")
        if size == "410m" and mx == "100000":
            ax.set_ylabel("eval seq_acc")

    fig.tight_layout()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight", dpi=160)
    plt.close(fig)
    print(f"Saved {OUT}")


if __name__ == "__main__":
    main()
