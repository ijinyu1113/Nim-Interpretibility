"""Step 2b (comma-list NL) vs step 6 (full multi-line Leo/Sultan paper prompt),
both INCORRECT holdout, oldcfg/cosine. Shows that the richer NL framing of
step 6 obstructs more -> longer / more visible coset plateaus.

Output: new_result/plots/step2b_vs_step6_incorrect.png
"""
import json
import os

import numpy as np
import matplotlib.pyplot as plt

from plot_style import setup_style

setup_style()

METRICS = "new_result/purenum_metrics"
OUT = "new_result/plots/step2b_vs_step6_incorrect.png"
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
        s2, v2 = load(mr, "oldcfg300ep_ladderC_step2b")
        s6, v6 = load(mr, "oldcfg300ep_purenums_paperC")
        xmax = 30000
        if s2.size:
            m = s2 <= xmax
            ax.plot(s2[m], v2[m], color="#2ca02c", lw=1.4, marker="o", ms=2.3,
                    alpha=0.9, label="step 2b (comma-list)")
        if s6.size:
            m = s6 <= xmax
            ax.plot(s6[m], v6[m], color="#67000d", lw=1.4, marker="s", ms=2.3,
                    alpha=0.9, label="step 6 (full Leo/Sultan)")
        ax.axhline(chance, color="#999", ls=":", lw=0.9, alpha=0.8)
        for d in [d for d in range(2, mod) if mod % d == 0]:
            lvl = d / mod
            ax.axhline(lvl, color="#888", ls="--", lw=0.8, alpha=0.6)
            ax.text(xmax * 0.99, lvl + 0.012, f"mod-{d} ({lvl:.2f})", ha="right",
                    fontsize=6.5, color="#888")
        ax.set_title(f"mr={mr} (mod {mod})")
        ax.set_xscale("log"); ax.set_xlim(200, xmax); ax.set_ylim(-0.02, 1.05)
        ax.grid(alpha=0.14, lw=0.5, which="both")
        ax.set_xlabel("Training step (log)")
        if mr in (3, 6):
            ax.set_ylabel("eval seq_acc")
        if mr == 7:
            ax.legend(fontsize=7.5, loc="lower right", frameon=False)

    fig.suptitle("Step 2b vs step 6, both incorrect holdout — richer NL framing (step 6) "
                 "dwells longer at cosets (esp. mod 8 -> mod-4)", y=1.01, fontsize=11)
    fig.tight_layout()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight", dpi=160)
    plt.close(fig)
    print(f"Saved {OUT}")


if __name__ == "__main__":
    main()
