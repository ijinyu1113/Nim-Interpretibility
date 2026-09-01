"""Disentangle sweep: mod-token vs +1 vs framing vs addition, at 5-digit
magnitude, correct holdout, constlr. Two panels (mod 8 local / mod 9 global),
5 cells each + the nimsimple anchor.

Output: new_result/plots/disentangle.png
"""
import json
import os

import numpy as np
import matplotlib.pyplot as plt

from plot_style import setup_style

setup_style()

D = "new_result/purenum_metrics"
OUT = "new_result/plots/disentangle.png"

CELLS = [
    ("puremod",   '"N mod 9 ="',          "#08306b"),
    ("modplus1",  '"N mod (8+1) ="',      "#4292c6"),
    ("remainder", "remainder wording",     "#2ca02c"),
    ("leftover",  "boxes-of-9 framing",    "#ff7f0e"),
    ("bare",      "bare number",           "#9467bd"),
    ("unrelated", "unrelated framing",     "#8c564b"),
    ("conflict",  "wrong-modulus text",    "#e377c2"),
]


def load(path):
    by = {}
    if not os.path.isfile(path):
        return np.array([]), np.array([])
    for ln in open(path):
        r = json.loads(ln); s = r.get("step")
        if s is not None and "eval_eval_seq_acc" in r: by[s] = r["eval_eval_seq_acc"]
    st = sorted(by)
    return np.array(st), np.array([by[s] for s in st])


def main():
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.3), sharey=True)
    for ax, mr in zip(axes, [7, 8]):
        mod = mr + 1
        for cell, label, color in CELLS:
            s, v = load(f"{D}/mr{mr}_410m_seed42_lr3e-5_wd0.05_constlr10000steps_disentangle_{cell}_max50000_evalevery25.jsonl")
            if s.size:
                lbl = label.replace("9", str(mod)).replace("(8+1)", f"({mr}+1)")
                ax.plot(s, v, color=color, lw=1.3, marker="o", ms=1.8, alpha=0.9, label=lbl)
        # nimsimple anchor (canonical constlr run, eval every 50)
        s, v = load(f"{D}/mr{mr}_410m_seed42_lr3e-5_wd0.05_constlr25000steps_nimsimple_max50000_evalevery50.jsonl")
        if s.size:
            m = s <= 10000
            ax.plot(s[m], v[m], color="#7f7f7f", lw=1.6, ls="--", alpha=0.9,
                    label="nimsimple anchor")
        chance = 1 / mod
        ax.axhline(chance, color="#999", ls=":", lw=0.9, alpha=0.8)
        for d in [d for d in range(2, mod) if mod % d == 0]:
            ax.axhline(d / mod, color="#bbb", ls="--", lw=0.7, alpha=0.6)
            ax.text(9950, d / mod + 0.012, f"mod-{d}", ha="right", fontsize=11, color="#999")
        ax.set_title(f"mod {mod} ({'local' if mod == 8 else 'global'})", fontsize=13)
        ax.set_xlim(0, 10000); ax.set_ylim(-0.02, 1.05)
        ax.grid(alpha=0.14, lw=0.5)
        ax.set_xlabel("Training step")
        if mr == 7:
            ax.set_ylabel("eval seq_acc")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=10,
               frameon=False, columnspacing=1.2, handlelength=1.4,
               bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight", dpi=160)
    plt.close(fig)
    print(f"Saved {OUT}")


if __name__ == "__main__":
    main()
