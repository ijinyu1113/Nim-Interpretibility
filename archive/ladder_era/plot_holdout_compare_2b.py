"""Step 2b (NL comma-list prompt), oldcfg/cosine LR, SAME everything except the
holdout: incorrect (100% overlap, old-paper style) vs correct (disjoint piles).
Isolates what train/eval overlap does to the apparent plateaus.

Output: new_result/plots/holdout_compare_2b.png
"""
import json
import os

import numpy as np
import matplotlib.pyplot as plt

from plot_style import setup_style

setup_style()

METRICS = "new_result/purenum_metrics"
OUT = "new_result/plots/holdout_compare_2b.png"
MRS = [3, 4, 5, 6, 7, 8]


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
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharey=True)
    for ax, mr in zip(axes.flat, MRS):
        mod = mr + 1
        chance = 1 / mod
        s_i, v_i = load(mr, "oldcfg300ep_ladderC_step2b")
        s_c, v_c = load(mr, "oldcfg300ep_ladderC_step2b_heldout")
        xmax = 30000
        if s_i.size:
            m = s_i <= xmax
            ax.plot(s_i[m], v_i[m], color="#d62728", lw=1.4, marker="o", ms=2.3,
                    alpha=0.9, label="incorrect holdout (overlap)")
        if s_c.size:
            m = s_c <= xmax
            ax.plot(s_c[m], v_c[m], color="#1f77b4", lw=1.4, marker="s", ms=2.3,
                    alpha=0.9, label="correct holdout (disjoint)")
        ax.axhline(chance, color="#999", ls=":", lw=0.9, alpha=0.8)
        for d in [d for d in range(2, mod) if mod % d == 0]:
            lvl = d / mod
            ax.axhline(lvl, color="#888", ls="--", lw=0.8, alpha=0.6)
            ax.text(xmax * 0.99, lvl + 0.012, f"mod-{d}", ha="right", fontsize=6.5, color="#888")
        ax.set_title(f"mr={mr} (mod {mod})")
        ax.set_xscale("log"); ax.set_xlim(200, xmax); ax.set_ylim(-0.02, 1.05)
        ax.grid(alpha=0.14, lw=0.5, which="both")
        ax.set_xlabel("Training step (log)")
        if mr in (3, 6):
            ax.set_ylabel("eval seq_acc")
        if mr == 3:
            ax.legend(fontsize=7.5, loc="lower right", frameon=False)

    fig.suptitle("Step 2b (NL comma-list), cosine LR — incorrect vs correct holdout: "
                 "overlap masks the generalization failure", y=1.01, fontsize=11)
    fig.tight_layout()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight", dpi=160)
    plt.close(fig)
    print(f"Saved {OUT}")


if __name__ == "__main__":
    main()
