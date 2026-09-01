"""Compare step 2a vs step 2b under two training regimes:
  - cosine + no-holdout (existing baseline)
  - constant-LR + correct init+final pile holdout (new)

Color encodes prompt format (2a = blue, 2b = red).
Line style encodes training regime (cosine = solid, constlr+holdout = dashed).

Output: new_result/plots/ladder_2a_vs_2b_constlr_heldout.png
"""
import json
import os

import numpy as np
import matplotlib.pyplot as plt

from plot_style import setup_style

setup_style()

METRICS_DIR = "new_result/purenum_metrics"
OUT = "new_result/plots/ladder_2a_vs_2b_constlr_heldout.png"
MRS = [3, 4, 5, 6, 7, 8]

CONDITIONS = [
    # (file_tag,                                            label,                          color,     linestyle)
    ("oldcfg300ep_ladderC_step2a",                          "2a cosine + no holdout",      "#1f77b4", "-"),
    ("oldcfg300ep_ladderC_step2b",                          "2b cosine + no holdout",      "#d62728", "-"),
    ("constlr30000steps_ladderC_step2a_heldout",            "2a constlr + holdout",        "#1f77b4", "--"),
    ("constlr30000steps_ladderC_step2b_heldout",            "2b constlr + holdout",        "#d62728", "--"),
]


def load(path):
    by_step = {}
    if not os.path.isfile(path):
        return np.array([]), np.array([])
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            st = r.get("step")
            if st is None: continue
            if "eval_eval_seq_acc" in r:
                by_step[st] = r["eval_eval_seq_acc"]
    steps = sorted(by_step.keys())
    return np.array(steps), np.array([by_step[s] for s in steps])


def main():
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharey=True)
    for ax, mr in zip(axes.flat, MRS):
        chance = 1.0 / (mr + 1)
        for file_tag, label, color, ls in CONDITIONS:
            f = f"{METRICS_DIR}/mr{mr}_410m_seed42_lr3e-5_wd0.05_{file_tag}.jsonl"
            s, v = load(f)
            if s.size == 0:
                continue
            mask = (s >= 1) & (s <= 30000)
            ax.plot(s[mask], v[mask], color=color, lw=1.3, ls=ls,
                    marker="o", ms=2.5, alpha=0.9, label=label)
        ax.axhline(chance, color="grey", ls=":", lw=0.9, alpha=0.7,
                   label=f"chance ({chance:.3f})")
        ax.set_title(f"mr={mr}  (mod {mr+1})")
        ax.set_xscale("log")
        ax.set_xlim(100, 30000)
        ax.set_ylim(-0.02, 1.05)
        ax.grid(alpha=0.14, linewidth=0.5, which="both")
        ax.set_xlabel("Training step (log)")
        if mr in (3, 6):
            ax.set_ylabel("eval seq_acc")
        if mr == 3:
            ax.legend(fontsize=8, loc="lower right", frameon=False)

    fig.suptitle(
        "Step 2a vs step 2b — cosine+no-holdout vs constlr+holdout — Pythia 410m",
        y=1.01, fontsize=11)
    fig.tight_layout()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight", dpi=160)
    plt.close(fig)
    print(f"Saved {OUT}")


if __name__ == "__main__":
    main()
