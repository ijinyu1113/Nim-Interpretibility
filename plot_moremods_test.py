"""Pre-registered test of the global-computation theory on new moduli.
nimsimple, constant LR, correct holdout, max=50000.
Predictions: mod 10/16 snap (local); mod 12/14/15 plateau at their LOCAL coset
(then grind the global factor); mod 11 slow ramp (prime).

Output: new_result/plots/moremods_test.png
"""
import json
import os

import numpy as np
import matplotlib.pyplot as plt

from plot_style import setup_style

setup_style()

METRICS = "new_result/purenum_metrics"
OUT = "new_result/plots/moremods_test.png"

# (mr, mod, label, rests-at-divisor or None, xmax)
PANELS = [
    (9, 10, "mod 10 = 2$\\cdot$5: snap (local)", None, 1500),
    (15, 16, "mod 16 = $2^4$: snap (local)", None, 2000),
    (11, 12, "mod 12 = $2^2\\cdot$3: plateau at mod-4", 4, 3000),
    (13, 14, "mod 14 = 2$\\cdot$7: plateau at mod-2", 2, 6000),
    (14, 15, "mod 15 = 3$\\cdot$5: plateau at mod-5", 5, 2500),
    (10, 11, "mod 11 (prime): slow ramp", None, 9000),
]


def load(mr):
    p = f"{METRICS}/mr{mr}_410m_seed42_lr3e-5_wd0.05_constlr30000steps_nimsimple_max50000_evalevery50.jsonl"
    by = {}
    for ln in open(p):
        r = json.loads(ln); s = r.get("step")
        if s is not None and "eval_eval_seq_acc" in r: by[s] = r["eval_eval_seq_acc"]
    st = sorted(by)
    return np.array(st), np.array([by[s] for s in st])


def main():
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 6.4))
    for ax, (mr, mod, title, rest, xmax) in zip(axes.flat, PANELS):
        s, v = load(mr)
        m = s <= xmax
        ax.plot(s[m], v[m], color="#1f77b4", lw=1.5, marker="o", ms=2.6, zorder=3)
        ax.axhline(1 / mod, color="#999", ls=":", lw=0.9, alpha=0.8)
        ax.text(xmax * 0.99, 1 / mod + 0.012, f"chance {1/mod:.3f}", ha="right", fontsize=11, color="#999")
        if rest is not None:
            lvl = rest / mod
            ax.axhline(lvl, color="#d62728", ls="--", lw=1.1, alpha=0.75)
            ax.text(xmax * 0.99, lvl + 0.013, f"mod-{rest} coset ({lvl:.3f})", ha="right",
                    fontsize=11, color="#d62728")
        ax.set_title(title, fontsize=13)
        ax.set_xlim(0, xmax); ax.set_ylim(-0.02, 1.05)
        ax.grid(alpha=0.14, lw=0.5)
        ax.set_xlabel("Training step")
        ax.set_ylabel("eval seq_acc")

    fig.tight_layout()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight", dpi=160)
    plt.close(fig)
    print(f"Saved {OUT}")


if __name__ == "__main__":
    main()
