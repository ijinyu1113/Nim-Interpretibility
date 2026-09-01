"""mod 6 vs mod 9 (nimsimple, constlr, correct holdout, max=50000): BOTH plateau
at their 0.333 coset, but mod 6 plateaus EARLY (local mod-2 coset) and mod 9
plateaus LATE (global mod-3 coset). Confirms the local-vs-global theory.

Output: new_result/plots/mod6_vs_mod9.png
"""
import json
import os

import numpy as np
import matplotlib.pyplot as plt

from plot_style import setup_style

setup_style()

METRICS = "new_result/purenum_metrics"
OUT = "new_result/plots/mod6_vs_mod9.png"


def load(mr):
    p = f"{METRICS}/mr{mr}_410m_seed42_lr3e-5_wd0.05_constlr25000steps_nimsimple_max50000_evalevery50.jsonl"
    by = {}
    for ln in open(p):
        r = json.loads(ln); s = r.get("step")
        if s is not None and "eval_eval_seq_acc" in r: by[s] = r["eval_eval_seq_acc"]
    st = sorted(by)
    return np.array(st), np.array([by[s] for s in st])


def main():
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    panels = [
        (5, 6, 2, "mod 6 = 2·3", "coset = mod-2 (LOCAL, parity)\n-> EARLY plateau", 1200),
        (8, 9, 3, "mod 9 = 3·3", "coset = mod-3 (GLOBAL, digit sum)\n-> LATE plateau", 5000),
    ]
    for ax, (mr, mod, cd, title, note, xmax) in zip(axes, panels):
        s, v = load(mr)
        m = s <= xmax
        ax.plot(s[m], v[m], color="#1f77b4", lw=1.6, marker="o", ms=3, zorder=3)
        ax.axhline(1 / mod, color="#999", ls=":", lw=0.9, alpha=0.8)
        ax.text(xmax * 0.99, 1 / mod + 0.012, f"chance {1/mod:.3f}", ha="right", fontsize=7, color="#999")
        ax.axhline(cd / mod, color="#d62728", ls="--", lw=1.1, alpha=0.75)
        ax.text(xmax * 0.99, cd / mod + 0.013, f"mod-{cd} coset ({cd/mod:.3f})", ha="right",
                fontsize=8, color="#d62728")
        ax.set_title(f"{title}\n{note}", fontsize=10)
        ax.set_xlim(0, xmax); ax.set_ylim(-0.02, 1.05)
        ax.grid(alpha=0.14, lw=0.5)
        ax.set_xlabel("Training step"); ax.set_ylabel("eval seq_acc")

    fig.suptitle("Both plateau at 0.333 — mod 6 early (local coset), mod 9 late (global coset)",
                 y=1.02, fontsize=11)
    fig.tight_layout()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight", dpi=160)
    plt.close(fig)
    print(f"Saved {OUT}")


if __name__ == "__main__":
    main()
