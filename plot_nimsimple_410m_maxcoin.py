"""410m ONLY — isolate the max-coin effect on the coset plateau (no model-size
confound). mod 6 (mr=5) and mod 8 (mr=7), overlaying max=50k/100k/500k.

Output: new_result/plots/nimsimple_410m_maxcoin.png
"""
import json
import os

import numpy as np
import matplotlib.pyplot as plt

from plot_style import setup_style

setup_style()

METRICS = "new_result/purenum_metrics"
OUT = "new_result/plots/nimsimple_410m_maxcoin.png"

# (max_coin label, file-tag, color)
SERIES = [
    ("max=50,000",  "constlr25000steps_nimsimple_max50000_evalevery50",  "#9ecae1"),
    ("max=100,000", "constlr30000steps_nimsimple_max100000_evalevery25", "#4292c6"),
    ("max=500,000", "constlr30000steps_nimsimple_max500000_evalevery25", "#08306b"),
]
PANELS = [(5, 6, 5000), (7, 8, 1500)]  # (mr, mod, xmax)


def load(mr, tag):
    p = f"{METRICS}/mr{mr}_410m_seed42_lr3e-5_wd0.05_{tag}.jsonl"
    by = {}
    if not os.path.isfile(p):
        return np.array([]), np.array([])
    for ln in open(p):
        r = json.loads(ln); s = r.get("step")
        if s is None: continue
        if "eval_eval_seq_acc" in r: by[s] = r["eval_eval_seq_acc"]
    st = sorted(by)
    return np.array(st), np.array([by[s] for s in st])


def main():
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    for ax, (mr, mod, xmax) in zip(axes, PANELS):
        chance = 1 / mod
        for label, tag, color in SERIES:
            s, v = load(mr, tag)
            if s.size:
                m = s <= xmax
                ax.plot(s[m], v[m], color=color, lw=1.6, marker="o", ms=2.6,
                        label=f"410m, {label}")
        ax.axhline(chance, color="#999", ls=":", lw=0.9, alpha=0.8)
        ax.text(xmax * 0.99, chance + 0.012, f"chance {chance:.3f}", ha="right",
                fontsize=7, color="#999")
        for d in [d for d in range(2, mod) if mod % d == 0]:
            lvl = d / mod
            ax.axhline(lvl, color="#d62728", ls="--", lw=1.0, alpha=0.6)
            ax.text(xmax * 0.99, lvl + 0.012, f"mod-{d} ({lvl:.3f})", ha="right",
                    fontsize=7, color="#d62728")
        note = "plateau GROWS with max coin" if mod == 6 else "snaps regardless (mod 8 = 2^3)"
        ax.set_title(f"mod {mod} (mr={mr}) — 410m only\n{note}")
        ax.set_xlim(0, xmax); ax.set_ylim(-0.02, 1.05)
        ax.grid(alpha=0.14, lw=0.5)
        ax.set_xlabel("Training step")
        ax.set_ylabel("eval seq_acc")
        ax.legend(fontsize=8, loc="lower right", frameon=False)

    fig.suptitle("Fixed model (410m): bigger max coin -> LONGER coset plateau (no model-size confound)",
                 y=1.02, fontsize=11)
    fig.tight_layout()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight", dpi=160)
    plt.close(fig)
    print(f"Saved {OUT}")


if __name__ == "__main__":
    main()
