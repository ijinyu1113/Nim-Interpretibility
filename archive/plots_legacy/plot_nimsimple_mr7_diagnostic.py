"""Diagnostic plot for mr=7 (mod 8) eval trajectory: eval curve with
heuristic reference lines + smoothed slope underneath. Goal: visually
test whether the model truly plateaus at heuristic values (mod-2=0.25,
mod-4=0.50) or just climbs smoothly through them.
"""
import json
import os

import numpy as np
import matplotlib.pyplot as plt

from plot_style import setup_style

setup_style()

PATH = "new_result/purenum_metrics/mr7_410m_seed42_lr3e-5_wd0.05_oldcfg300ep_nimsimple_max50000_evalevery1.jsonl"
OUT = "new_result/plots/nimsimple_mr7_diagnostic.png"

# Heuristic predictions: 1/Y for chance, d/Y for partial mod-d knowledge.
CHANCE = 1/8
HEURISTICS = {
    "chance (1/8)": (CHANCE, "#999"),
    "parity (2/8)": (0.25,   "#ff7f0e"),
    "mod-4 (4/8)":  (0.50,   "#2ca02c"),
    "full (1.0)":   (1.0,    "#1f77b4"),
}
SMOOTH_W = 15


def load():
    by_step = {}
    with open(PATH) as f:
        for line in f:
            r = json.loads(line)
            if "eval_eval_move_acc" in r:
                by_step[r["step"]] = r["eval_eval_move_acc"]
    steps = np.array(sorted(by_step.keys()))
    vals = np.array([by_step[s] for s in steps])
    return steps, vals


def main():
    steps, vals = load()
    # Smoothed series
    w = SMOOTH_W
    sm = np.convolve(vals, np.ones(w) / w, mode="valid")
    sm_steps = steps[w // 2: w // 2 + len(sm)]
    slope = np.gradient(sm, sm_steps) * 50  # accuracy change per 50 steps

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(12, 7), sharex=True,
        gridspec_kw={"height_ratios": [3, 1.4]},
    )

    # Top: raw + smoothed eval, with heuristic reference lines
    ax1.plot(steps, vals, color="#aaa", lw=0.7, alpha=0.6, label="eval (raw)")
    ax1.plot(sm_steps, sm, color="#1f77b4", lw=2.0,
             label=f"eval (smoothed, window={w})")
    for name, (y, color) in HEURISTICS.items():
        ax1.axhline(y, color=color, ls=":", lw=1.0, alpha=0.8)
        ax1.text(steps[-1] * 1.005, y, name, va="center", fontsize=8, color=color)
    ax1.set_ylabel("eval move_acc")
    ax1.set_ylim(-0.02, 1.05)
    ax1.set_title("mr=7 (mod 8) — eval trajectory with heuristic reference levels")
    ax1.grid(alpha=0.14, linewidth=0.5)
    ax1.legend(fontsize=8, loc="lower right", frameon=False)

    # Bottom: slope (smoothed). Plateau = near-zero slope. Climb = larger.
    ax2.plot(sm_steps, slope, color="#d62728", lw=1.4, label="slope per 50 steps")
    ax2.axhline(0, color="black", lw=0.5, alpha=0.5)
    ax2.fill_between(sm_steps, -0.02, 0.02, color="#888", alpha=0.15,
                     label="|slope| < 0.02 (near-flat)")
    ax2.set_ylabel("d(eval) / d(50 steps)")
    ax2.set_xlabel("Training step")
    ax2.grid(alpha=0.14, linewidth=0.5)
    ax2.legend(fontsize=8, loc="upper right", frameon=False)

    fig.tight_layout()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight", dpi=160)
    plt.close(fig)
    print(f"Saved {OUT}")


if __name__ == "__main__":
    main()
