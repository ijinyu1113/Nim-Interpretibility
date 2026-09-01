"""2x3 grid combining 70m / 160m / 410m purenum curves.

Top row: eval move acc per (mr) — one panel per model size.
Bottom row: train move acc.
Shared y-axis per row, shared x-axis. One legend at the top.

Reuses the gather_long / aggregate_per_step functions from
plot_purenum_curves.py by importing them in a per-size loop.
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

from plot_style import setup_style, HLINE_KW

setup_style()

# Force MODEL_SIZE for the imported module so its file paths match the size.
SIZES = ["70m", "160m", "410m"]
SIZE_LABELS = {"70m": "Pythia-70M", "160m": "Pythia-160M", "410m": "Pythia-410M"}

OUT = "new_result/plots/purenum_curves_grid.png"

MRS = [3, 4, 5, 6, 7, 8]
BUCKET_COLORS = plt.cm.tab10(range(len(MRS)))


def load_size(size):
    """Re-execute plot_purenum_curves's loading + aggregation for `size`."""
    # The module has module-level CLI parsing; we override sys.argv before import.
    sys.argv = ["plot_purenum_curves.py", size]
    # Force re-import each time to pick up new MODEL_SIZE
    if "plot_purenum_curves" in sys.modules:
        del sys.modules["plot_purenum_curves"]
    import plot_purenum_curves as ppc
    df = ppc.gather_long()
    eval_agg = ppc.aggregate_per_step(df, "eval_acc", step_multiple=500)
    train_agg = ppc.aggregate_per_step(df, "train_acc", step_multiple=1000)
    return eval_agg, train_agg


def plot_panel(ax, agg, max_step_k_global):
    for ci, mr in enumerate(MRS):
        sub = agg[agg["mr"] == mr]
        if sub.empty:
            continue
        x = sub["step"].to_numpy() / 1000.0
        med = sub["median"].to_numpy()
        lo = sub["lo"].to_numpy()
        hi = sub["hi"].to_numpy()
        ax.plot(x, med, color=BUCKET_COLORS[ci], label=f"mr={mr}")
        if np.any(hi > lo):
            ax.fill_between(x, lo, hi, color=BUCKET_COLORS[ci],
                            alpha=0.14, linewidth=0)
        ax.axhline(1.0 / (mr + 1), color=BUCKET_COLORS[ci],
                   lw=HLINE_KW["lw"], ls=HLINE_KW["ls"], alpha=0.35, zorder=0)
    ax.set_ylim(-0.02, 1.05)
    ax.set_xscale("symlog", linthresh=10, linscale=1)
    ax.set_xlim(0, max_step_k_global)
    log_ticks = [t for t in (15, 20, 30, 50, 70, 100) if t <= max_step_k_global]
    xticks = [0, 5, 10] + log_ticks
    ax.set_xticks(xticks)
    ax.set_xticklabels([str(t) for t in xticks])
    ax.grid(alpha=0.14, linewidth=0.5)


def main():
    # Load all three sizes once
    by_size = {s: load_size(s) for s in SIZES}

    # Compute global max-step for shared x-axis
    max_step_k = 0.0
    for s in SIZES:
        for agg in by_size[s]:
            if not agg.empty:
                max_step_k = max(max_step_k, agg["step"].max() / 1000.0)
    upper = int(np.ceil(max_step_k / 5.0) * 5) or 35

    fig, axes = plt.subplots(
        2, 3, figsize=(12.0, 5.5),
        sharex=True, sharey=True,
        gridspec_kw={"wspace": 0.06, "hspace": 0.18},
    )

    # Top row: eval, bottom row: train
    for col, size in enumerate(SIZES):
        eval_agg, train_agg = by_size[size]
        plot_panel(axes[0, col], eval_agg, upper)
        plot_panel(axes[1, col], train_agg, upper)
        axes[0, col].set_title(SIZE_LABELS[size], fontsize=11)

    # Y labels
    axes[0, 0].set_ylabel("Eval accuracy")
    axes[1, 0].set_ylabel("Train accuracy")
    # X labels on bottom row only
    for col in range(3):
        axes[1, col].set_xlabel(r"Training step (${\times}10^3$)")

    # Single legend at top
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(MRS),
               bbox_to_anchor=(0.5, 1.02), frameon=False, fontsize=10)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {OUT}")


if __name__ == "__main__":
    main()
