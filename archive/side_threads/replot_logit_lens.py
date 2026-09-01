"""Replot logit_lens.png from cached npy files (no GPU needed).

Loads new_result/logit_lens/ckpt-*/{mlp_logits,attn_logits}.npy and regenerates
the 2-panel heatmap with the legend placed OUTSIDE the plot area (top-right
of the figure) instead of inside the upper-right corner of each axis.

Usage:
    python replot_logit_lens.py
"""
import glob
import os
import re
import numpy as np
import matplotlib.pyplot as plt

ROOT = "new_result/logit_lens"
MAX_REMOVE = 5
MOVE_LABELS = ["-1"] + [str(k) for k in range(1, MAX_REMOVE + 1)]

# Hard-coded for example_prompt.txt (n=61 with history → 52 remaining,
# 52 mod 6 = 4, so correct = "4").
PROMPT_N = 52
PROMPT_M = MAX_REMOVE + 1
CORRECT_MOVE = "4"


def plot_pair(mlp_logits, attn_logits, move_labels, out_path,
              title_suffix="", correct_move=None):
    fig, axes = plt.subplots(2, 1, figsize=(10, 5.5), sharex=True)
    for ax, data, kind in zip(axes, [mlp_logits, attn_logits], ["MLP", "Attention"]):
        d = data.T
        vmax = np.abs(d).max()
        im = ax.imshow(d, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.set_ylabel("Move")
        ax.set_yticks(range(len(move_labels)))
        ax.set_yticklabels(move_labels)
        ax.set_title(f"{kind} per-layer logits {title_suffix}")
        plt.colorbar(im, ax=ax, label="Logit")

        if correct_move is not None and correct_move in move_labels:
            row = move_labels.index(correct_move)
            ax.axhline(row, color="lime", linewidth=1.2, alpha=0.9,
                       linestyle="--", label=f"correct: {correct_move}")

    axes[1].set_xlabel("Layer")

    # Single shared legend OUTSIDE the plot area, top-right of figure
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels,
                   loc="upper right", bbox_to_anchor=(0.98, 0.99),
                   fontsize=9, frameon=True)

    plt.tight_layout(rect=[0, 0, 0.92, 0.96])
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {out_path}")


def step_from_dir(d):
    m = re.search(r"ckpt-(\d+)", d)
    return int(m.group(1)) if m else -1


def main():
    ckpt_dirs = sorted(
        [d for d in glob.glob(os.path.join(ROOT, "ckpt-*")) if os.path.isdir(d)],
        key=step_from_dir,
    )
    if not ckpt_dirs:
        print(f"No ckpt-* dirs under {ROOT}")
        return

    for d in ckpt_dirs:
        mlp_path = os.path.join(d, "mlp_logits.npy")
        attn_path = os.path.join(d, "attn_logits.npy")
        if not (os.path.isfile(mlp_path) and os.path.isfile(attn_path)):
            print(f"skip {d}: missing npy")
            continue
        mlp = np.load(mlp_path)
        attn = np.load(attn_path)
        out = os.path.join(d, "logit_lens.png")
        step = step_from_dir(d)
        plot_pair(mlp, attn, MOVE_LABELS, out,
                  title_suffix=f"(step={step}, n={PROMPT_N}, correct={CORRECT_MOVE})",
                  correct_move=CORRECT_MOVE)


if __name__ == "__main__":
    main()
