"""Plot prompt-ladder step 2: drop explicit `mod Y`, add `range [1, mr]` hint.
   Prompt: "range [1, 6]. 398 - 3 - 1 - 5 - 2 = "

Overlays step 0 (bare math, baseline) for direct comparison so we can see
whether removing the explicit `mod` re-introduces plateaus.

Output: new_result/plots/prompt_ladder_step2.png
"""
import json
import os

import numpy as np
import matplotlib.pyplot as plt

from plot_style import setup_style

setup_style()

METRICS_DIR = "new_result/purenum_metrics"
OUT = "new_result/plots/prompt_ladder_step2.png"
MRS = [3, 4, 5, 6, 7, 8]


def load(path):
    by_step = {}
    if not os.path.isfile(path):
        return np.array([]), np.array([])
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            st = r.get("step")
            if st is None: continue
            if "eval_eval_move_acc" in r:
                by_step[st] = r["eval_eval_move_acc"]
    steps = sorted(by_step.keys())
    return np.array(steps), np.array([by_step[s] for s in steps])


def main():
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharey=True)
    for ax, mr in zip(axes.flat, MRS):
        chance = 1.0 / (mr + 1)

        f0 = f"{METRICS_DIR}/mr{mr}_410m_seed42_lr3e-5_wd0.05_oldcfg300ep_modarith_subtract_noholdout_max400_evalevery1.jsonl"
        s0, v0 = load(f0)
        if s0.size:
            mask = s0 <= 1000
            ax.plot(s0[mask], v0[mask], color="#1f77b4", lw=1.2,
                    marker="o", ms=2.5, alpha=0.7,
                    label="step 0 (bare math)")

        f2 = f"{METRICS_DIR}/mr{mr}_410m_seed42_lr3e-5_wd0.05_oldcfg300ep_prompt_ladder_step2_max400_noholdout_evalevery1.jsonl"
        s2, v2 = load(f2)
        if s2.size:
            mask = s2 <= 1000
            ax.plot(s2[mask], v2[mask], color="#d62728", lw=1.4,
                    marker="o", ms=2.5,
                    label="step 2 (range, no mod)")

        ax.axhline(chance, color="grey", ls=":", lw=0.9, alpha=0.7,
                   label=f"chance ({chance:.3f})")
        ax.set_title(f"mr={mr}  (mod {mr+1})")
        ax.set_xlim(0, 1000)
        ax.set_ylim(-0.02, 1.05)
        ax.grid(alpha=0.14, linewidth=0.5)
        ax.set_xlabel("Training step")
        if mr in (3, 6):
            ax.set_ylabel("eval move_acc")
        ax.legend(fontsize=7.5, loc="lower right", frameon=False)

    fig.suptitle(
        "Prompt-ladder step 2: drop explicit `mod Y`, add `range [1, mr]` hint  (vs step 0 bare math) — Pythia 410m",
        y=1.01, fontsize=11)
    fig.tight_layout()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight", dpi=160)
    plt.close(fig)
    print(f"Saved {OUT}")


if __name__ == "__main__":
    main()
