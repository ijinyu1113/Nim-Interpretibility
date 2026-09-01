"""nimsimple (pile given), constant LR, correct holdout (max=50000) — one panel
per mr, x-axis zoomed to that modulus's transition so the coset plateaus are
visible. Coset reference lines drawn per modulus.

Output: new_result/plots/nimsimple_cosets.png
"""
import json
import os

import numpy as np
import matplotlib.pyplot as plt

from plot_style import setup_style

setup_style()

METRICS = "new_result/purenum_metrics"
OUT = "new_result/plots/nimsimple_cosets.png"
MRS = [3, 4, 5, 6, 7, 8]

# per-mr x zoom (transition fills the panel) and the coset divisors to mark
XMAX = {3: 1500, 4: 1500, 5: 2500, 6: 25000, 7: 2500, 8: 6000}


def load(path):
    by = {}
    if not os.path.isfile(path):
        return np.array([]), np.array([])
    for ln in open(path):
        r = json.loads(ln); s = r.get("step")
        if s is None: continue
        if "eval_eval_seq_acc" in r: by[s] = r["eval_eval_seq_acc"]
        elif "eval_eval_move_acc" in r: by[s] = r["eval_eval_move_acc"]
    st = sorted(by)
    return np.array(st), np.array([by[s] for s in st])


def main():
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, mr in zip(axes.flat, MRS):
        mod = mr + 1
        chance = 1 / mod
        s, v = load(f"{METRICS}/mr{mr}_410m_seed42_lr3e-5_wd0.05_constlr25000steps_nimsimple_max50000_evalevery50.jsonl")
        xmax = XMAX[mr]
        if s.size:
            m = s <= xmax
            ax.plot(s[m], v[m], color="#1f77b4", lw=1.5, marker="o", ms=2.8, zorder=3)

        # chance
        ax.axhline(chance, color="#999", ls=":", lw=0.9, alpha=0.8)
        ax.text(xmax * 0.99, chance + 0.012, f"chance {chance:.3f}", ha="right",
                fontsize=7, color="#999")
        # coset levels (nontrivial divisors of mod)
        divs = [d for d in range(2, mod) if mod % d == 0]
        for d in divs:
            lvl = d / mod
            ax.axhline(lvl, color="#d62728", ls="--", lw=1.0, alpha=0.7)
            ax.text(xmax * 0.99, lvl + 0.012, f"mod-{d} coset ({lvl:.3f})", ha="right",
                    fontsize=7, color="#d62728")
        title = f"mr={mr}  (mod {mod}" + (", prime" if not divs else "") + ")"
        ax.set_title(title)
        ax.set_xlim(0, xmax); ax.set_ylim(-0.02, 1.05)
        ax.grid(alpha=0.14, lw=0.5)
        ax.set_xlabel("Training step")
        if mr in (3, 6):
            ax.set_ylabel("eval seq_acc")

    fig.suptitle("nimsimple (pile given), constant LR, correct holdout (max=50000) — "
                 "coset plateaus per modulus", y=1.01, fontsize=12)
    fig.tight_layout()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight", dpi=160)
    plt.close(fig)
    print(f"Saved {OUT}")


if __name__ == "__main__":
    main()
