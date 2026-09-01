"""Build the 3 x N logit-lens evolution figure for a chosen ckpt subset.

Same as plot_logit_lens_evolution.py but with STEPS_SUBSET = [1k,2k,5k,9k,10k,11k]
and a separate output path.
"""
import glob
import os
import re
import numpy as np
import matplotlib.pyplot as plt

ROOT = "new_result/logit_lens"
OUT = "new_result/plots/logit_lens_evolution_subset2.png"
MAX_REMOVE = 5
MOVE_LABELS = ["-1"] + [str(k) for k in range(1, MAX_REMOVE + 1)]
CORRECT_MOVE = "4"  # for example_prompt.txt (n=52, mod 6 = 4)
STEPS_SUBSET = [1000, 2000, 5000, 9000, 10000, 11000]

ROW_KINDS = [
    ("mlp",   "MLP",             "MLP"),
    ("attn",  "Attention",       "Attention"),
    ("resid", "Residual stream", "Residual stream"),
]


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

    steps = []
    data_by_kind = {k: [] for k, _, _ in ROW_KINDS}
    for d in ckpt_dirs:
        paths = {k: os.path.join(d, f"{k}_logits.npy") for k, _, _ in ROW_KINDS}
        if not all(os.path.isfile(p) for p in paths.values()):
            continue
        s = step_from_dir(d)
        if STEPS_SUBSET is not None and s not in STEPS_SUBSET:
            continue
        steps.append(s)
        for k in data_by_kind:
            data_by_kind[k].append(np.load(paths[k]).T)  # [n_moves, n_layers]

    n = len(steps)
    if n == 0:
        print("No checkpoints with all 3 npy files; aborting.")
        return
    print(f"Loaded {n} checkpoints: {steps}")

    # Symmetric per-row vmin/vmax so colors compare across columns
    vmaxes = {k: max(np.abs(a).max() for a in data_by_kind[k]) for k in data_by_kind}
    correct_row = MOVE_LABELS.index(CORRECT_MOVE) if CORRECT_MOVE in MOVE_LABELS else None

    # Square cells
    sample = data_by_kind["mlp"][0]
    n_moves, n_layers = sample.shape
    cell_in = 0.16
    panel_w = cell_in * n_layers
    panel_h = cell_in * n_moves
    n_rows = len(ROW_KINDS)
    fig_w = panel_w * n + 2.2
    fig_h = panel_h * n_rows + 1.8

    fig, axes = plt.subplots(
        n_rows, n, figsize=(fig_w, fig_h),
        sharex=True, sharey=True,
        gridspec_kw={"wspace": 0.06, "hspace": 0.32},
    )
    if n_rows == 1:
        axes = axes[None, :]
    if n == 1:
        axes = axes[:, None]

    last_im_per_row = {}
    for row_idx, (kind_key, _, _) in enumerate(ROW_KINDS):
        for col, (step, mat) in enumerate(zip(steps, data_by_kind[kind_key])):
            ax = axes[row_idx, col]
            im = ax.imshow(mat, aspect="equal", cmap="RdBu_r",
                           vmin=-vmaxes[kind_key], vmax=vmaxes[kind_key],
                           interpolation="nearest")
            last_im_per_row[row_idx] = im

            if row_idx == 0:
                ax.set_title(f"step {step // 1000}k", fontsize=9)

            if correct_row is not None:
                ax.axhline(correct_row, color="lime", linewidth=0.9,
                           alpha=0.85, linestyle="--")

    # X labels only on bottom row, sparse ticks
    for col in range(n):
        axes[-1, col].set_xticks([0, n_layers // 2, n_layers - 1])
        axes[-1, col].set_xticklabels([0, n_layers // 2, n_layers - 1])
        axes[-1, col].set_xlabel("Layer", fontsize=9)

    # Y "Move" labels on the RIGHTMOST column (ticks + axis label on the right).
    # With sharey=True, we must explicitly enable right-side tick labels and
    # disable the (default) left-side ones; tick_right() alone is not enough.
    for row in range(n_rows):
        # Hide y ticks on all non-rightmost columns
        for col in range(n - 1):
            axes[row, col].tick_params(axis="y", which="both",
                                       left=False, labelleft=False,
                                       right=False, labelright=False)
        ax_right = axes[row, -1]
        ax_right.set_yticks(range(len(MOVE_LABELS)))
        ax_right.set_yticklabels(MOVE_LABELS, fontsize=8)
        ax_right.tick_params(axis="y", which="both",
                             left=False, labelleft=False,
                             right=True, labelright=True)
        ax_right.yaxis.set_label_position("right")
        ax_right.set_ylabel("Move")

    # Per-row colorbars on the LEFT, labeled with the row's component
    for row, (_, _, cbar_lbl) in enumerate(ROW_KINDS):
        cbar = fig.colorbar(last_im_per_row[row], ax=axes[row, :],
                            location="left", shrink=0.85, pad=0.02,
                            fraction=0.012)
        cbar.set_label(cbar_lbl, fontsize=9)

    # Single legend for the correct-move dashed line
    from matplotlib.lines import Line2D
    legend_handles = [Line2D([0], [0], color="lime", lw=1.5, ls="--",
                             label=f"correct move = {CORRECT_MOVE}")]
    fig.legend(handles=legend_handles, loc="upper center",
               bbox_to_anchor=(0.5, 1.02), frameon=False, fontsize=9)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"Saved: {OUT}")


if __name__ == "__main__":
    main()
