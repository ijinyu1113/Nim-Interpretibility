"""Plot mr=3 nimsimple max_pile=50000 with eval_every=10 (fine-grained early window).
Shows transition between random and full eval accuracy in first 1500 steps.

Two panels:
  (a) eval move_acc vs step — the answer-token-only accuracy
  (b) eval token_acc and eval_loss — total token accuracy + loss
"""
import json
import os
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

from plot_style import setup_style

setup_style()

PATH = "new_result/purenum_metrics/mr3_410m_seed42_lr3e-5_wd0.05_oldcfg300ep_nimsimple_max50000_evalevery10.jsonl"
OUT = "new_result/plots/nimsimple_mr3_finegrain.png"
CHANCE = 0.25


def load():
    by_step = defaultdict(dict)
    with open(PATH) as f:
        for line in f:
            r = json.loads(line)
            st = r.get("step")
            if st is None:
                continue
            for k in ("eval_eval_move_acc", "eval_eval_token_acc",
                     "eval_eval_loss", "eval_train_move_acc"):
                if k in r:
                    by_step[st][k] = r[k]
    return by_step


def main():
    data = load()
    steps = sorted(data.keys())
    move_acc = np.array([data[s].get("eval_eval_move_acc", np.nan) for s in steps])
    tok_acc = np.array([data[s].get("eval_eval_token_acc", np.nan) for s in steps])
    ev_loss = np.array([data[s].get("eval_eval_loss", np.nan) for s in steps])
    tr_acc = np.array([data[s].get("eval_train_move_acc", np.nan) for s in steps])
    steps = np.array(steps)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))

    ax = axes[0]
    m = ~np.isnan(move_acc)
    ax.plot(steps[m], move_acc[m], color="#1f77b4", lw=1.4, marker="o", ms=2.8,
            label="eval move_acc")
    m = ~np.isnan(tr_acc)
    ax.plot(steps[m], tr_acc[m], color="#ff7f0e", lw=1.0, marker="s", ms=2.4,
            alpha=0.6, label="train move_acc")
    ax.axhline(CHANCE, color="grey", ls=":", lw=0.9, alpha=0.7,
               label=f"chance ({CHANCE:.2f})")
    ax.set_xlabel("Training step")
    ax.set_ylabel("move_acc")
    ax.set_title("mr=3 nimsimple max_pile=50k — first 1500 steps (eval every 10)")
    ax.set_ylim(-0.02, 1.05)
    ax.grid(alpha=0.14, linewidth=0.5)
    ax.legend(fontsize=8, loc="lower right", frameon=False)

    ax = axes[1]
    m = ~np.isnan(tok_acc)
    ax.plot(steps[m], tok_acc[m], color="#2ca02c", lw=1.2, marker="o", ms=2.6,
            label="eval token_acc")
    ax.set_xlabel("Training step")
    ax.set_ylabel("token_acc", color="#2ca02c")
    ax.set_ylim(-0.02, 1.05)
    ax.tick_params(axis="y", labelcolor="#2ca02c")
    ax.grid(alpha=0.14, linewidth=0.5)
    ax.set_title("Token accuracy + eval loss (log)")

    ax2 = ax.twinx()
    m = ~np.isnan(ev_loss)
    ax2.semilogy(steps[m], ev_loss[m], color="#d62728", lw=1.0, marker="x",
                 ms=3, label="eval loss")
    ax2.set_ylabel("eval loss (log)", color="#d62728")
    ax2.tick_params(axis="y", labelcolor="#d62728")

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8,
              loc="center right", frameon=False)

    fig.tight_layout()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight", dpi=160)
    plt.close(fig)
    print(f"Saved {OUT}")

    # Key transition steps
    print("\n=== Key transitions ===")
    last_zero = 0
    for i, s in enumerate(steps):
        if not np.isnan(move_acc[i]) and move_acc[i] < 0.05:
            last_zero = s
    print(f"Last step move_acc < 0.05: {last_zero}")
    for thresh in [0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]:
        for i, s in enumerate(steps):
            if not np.isnan(move_acc[i]) and move_acc[i] >= thresh:
                print(f"  first move_acc >= {thresh}: step {s}  (move={move_acc[i]:.4f})")
                break


if __name__ == "__main__":
    main()
