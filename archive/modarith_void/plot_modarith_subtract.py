"""Plot the bare-math subtraction prompt sweep: ALL mrs hit 1.0 within ~200 steps,
contrasted with nimsubtract (natural language framing) which failed at chance.

Output: new_result/plots/modarith_subtract_sweep.png
"""
import json
import os
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

from plot_style import setup_style

setup_style()

METRICS_DIR = "new_result/purenum_metrics"
OUT = "new_result/plots/modarith_subtract_sweep.png"
MRS = [3, 4, 5, 6, 7, 8]


def load(path):
    """Return arrays of (steps, train_move_acc, eval_move_acc)."""
    by_step = {}
    if not os.path.isfile(path):
        return np.array([]), np.array([]), np.array([])
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            st = r.get("step")
            if st is None: continue
            d = by_step.setdefault(st, {})
            if "eval_eval_move_acc" in r:
                d["eval"] = r["eval_eval_move_acc"]
            if "eval_train_move_acc" in r:
                d["train"] = r["eval_train_move_acc"]
    steps = sorted(by_step.keys())
    return (
        np.array(steps),
        np.array([by_step[s].get("train", np.nan) for s in steps]),
        np.array([by_step[s].get("eval", np.nan) for s in steps]),
    )


def main():
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharey=True)
    for ax, mr in zip(axes.flat, MRS):
        chance = 1.0 / (mr + 1)

        f_bare = f"{METRICS_DIR}/mr{mr}_410m_seed42_lr3e-5_wd0.05_oldcfg300ep_modarith_subtract_max500_evalevery100.jsonl"
        steps, tr, ev = load(f_bare)
        m_t = ~np.isnan(tr)
        m_e = ~np.isnan(ev)
        if m_e.any():
            ax.plot(steps[m_e], ev[m_e], color="#1f77b4", lw=1.5,
                    marker="o", ms=3, label="eval")
        if m_t.any():
            ax.plot(steps[m_t], tr[m_t], color="#ff7f0e", lw=1.2, ls="--",
                    marker="s", ms=2.5, alpha=0.85, label="train")
        ax.axhline(chance, color="grey", ls=":", lw=0.9, alpha=0.7,
                   label=f"chance ({chance:.3f})")

        ax.set_title(f"mr={mr}  (mod {mr+1})")
        ax.set_xscale("symlog", linthresh=100)
        ax.set_xlim(0, 35000)
        ax.set_ylim(-0.02, 1.05)
        ax.grid(alpha=0.14, linewidth=0.5, which="both")
        ax.set_xlabel("Training step")
        if mr in (3, 6):
            ax.set_ylabel("move_acc")
        ax.legend(fontsize=8, loc="center right", frameon=False)

    fig.suptitle(
        "Bare math '{x} - ({a1}+{a2}+{a3}+{a4}) mod Y = ' — Pythia 410m oldcfg, max_pile=500",
        y=1.01)
    fig.tight_layout()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight", dpi=160)
    plt.close(fig)
    print(f"Saved {OUT}")


if __name__ == "__main__":
    main()
