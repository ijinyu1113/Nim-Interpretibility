"""Plot prompt-ladder steps 4 and 6 alongside step 0 (bare math baseline).
   Step 4: full sentences, "took N coins", "should take "  (eval_every=1, stop=2000)
   Step 6: Leo + Sultan original Nim prompt          (eval_every=50, stop=25000)

Two figures:
   prompt_ladder_step4.png  — 0..2000 steps, linear x
   prompt_ladder_step6.png  — 0..25000 steps, log x (so we can see early dynamics)
"""
import json
import os

import numpy as np
import matplotlib.pyplot as plt

from plot_style import setup_style

setup_style()

METRICS_DIR = "new_result/purenum_metrics"
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


def fname(step_tag, mr):
    return (f"{METRICS_DIR}/mr{mr}_410m_seed42_lr3e-5_wd0.05_oldcfg300ep_"
            f"{step_tag}.jsonl")


def plot_step(step_idx, baseline_step_tag, baseline_label,
              compare_step_tag, compare_label, xlim, xscale, out):
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharey=True)
    for ax, mr in zip(axes.flat, MRS):
        chance = 1.0 / (mr + 1)

        s0, v0 = load(fname(baseline_step_tag, mr))
        if s0.size:
            mask = (s0 >= 1) & (s0 <= xlim[1]) if xscale == "log" else (s0 <= xlim[1])
            ax.plot(s0[mask], v0[mask], color="#1f77b4", lw=1.2,
                    marker="o", ms=2.5, alpha=0.65, label=baseline_label)

        s, v = load(fname(compare_step_tag, mr))
        if s.size:
            mask = (s >= 1) & (s <= xlim[1]) if xscale == "log" else (s <= xlim[1])
            ax.plot(s[mask], v[mask], color="#d62728", lw=1.4,
                    marker="o", ms=3, label=compare_label)

        ax.axhline(chance, color="grey", ls=":", lw=0.9, alpha=0.7,
                   label=f"chance ({chance:.3f})")
        ax.set_title(f"mr={mr}  (mod {mr+1})")
        ax.set_xlim(*xlim)
        if xscale == "log":
            ax.set_xscale("log")
        ax.set_ylim(-0.02, 1.05)
        ax.grid(alpha=0.14, linewidth=0.5)
        ax.set_xlabel("Training step" + (" (log)" if xscale == "log" else ""))
        if mr in (3, 6):
            ax.set_ylabel("eval move_acc")
        ax.legend(fontsize=7.5, loc="lower right", frameon=False)

    fig.suptitle(f"Prompt-ladder step {step_idx} — Pythia 410m oldcfg, no holdout, max=400",
                 y=1.01, fontsize=11)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, bbox_inches="tight", dpi=160)
    plt.close(fig)
    print(f"Saved {out}")


def main():
    base = "modarith_subtract_noholdout_max400_evalevery1"
    plot_step(
        step_idx=4,
        baseline_step_tag=base,
        baseline_label="step 0 (bare math)",
        compare_step_tag="prompt_ladder_step4_max400_noholdout_evalevery1",
        compare_label="step 4 (full sentences)",
        xlim=(0, 2000), xscale="linear",
        out="new_result/plots/prompt_ladder_step4.png",
    )
    plot_step(
        step_idx=6,
        baseline_step_tag=base,
        baseline_label="step 0 (bare math)",
        compare_step_tag="prompt_ladder_step6_max400_noholdout_evalevery50",
        compare_label="step 6 (Leo + Sultan)",
        xlim=(1, 25000), xscale="log",
        out="new_result/plots/prompt_ladder_step6.png",
    )


if __name__ == "__main__":
    main()
