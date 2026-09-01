"""Round-1 T/X batch figure: sign flip (the same donor obstructs mod-9 and
accelerates mod-6), seed robustness, X-ABL staircase ablation, X-CURR
magnitude curriculum phase 1, and the scaffold-donor curves.

Writes new_result/plots/tx_round1.png.
"""
import json

import matplotlib.pyplot as plt

from plot_style import setup_style
setup_style()
import numpy as np

INK = "#0b0b0b"; MUTED = "#898781"; GRID = "#e1e0d9"; AXIS = "#c3c2b7"
BLUE = "#2a78d6"; AQUA = "#1baf7a"; VIOLET = "#4a3aa7"; YELLOW = "#eda100"
SURFACE = "#fcfcfb"
MET = "new_result/purenum_metrics"
PRE = "410m_seed42_lr3e-5_wd0.05"


def load(f):
    pts = {}
    for ln in open(f"{MET}/{f}"):
        r = json.loads(ln)
        if "eval_eval_seq_acc" in r:
            pts[r["step"]] = r["eval_eval_seq_acc"]
    s = np.array(sorted(pts)); return s, np.array([pts[k] for k in s])


def style(ax, title):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.7)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.set_ylim(-0.04, 1.06)
    ax.set_title(title, color=INK, fontsize=12, fontweight="bold")


def main():
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 6.6))
    fig.patch.set_facecolor(SURFACE)

    # (0,0) same donor -> mod 9: obstructs
    ax = axes[0, 0]; style(ax, "mod-3 donor $\\to$ mod 9")
    s, a = load(f"mr8_{PRE}_constlr6000steps_nimsimple_max50000_evalevery25.jsonl")
    ax.plot(s, a, color=INK, linewidth=1.8, label="scratch (3725)")
    s, a = load(f"mr8_{PRE}_constlr10000steps_nimsimple_max50000_evalevery25_initmod3pre6k.jsonl")
    ax.plot(s, a, color=BLUE, linewidth=1.8, label="mod-3 donor (6250)")
    ax.set_xlim(0, 8000)
    ax.axhline(1 / 3, color=MUTED, linewidth=0.7, linestyle=":")
    ax.legend(loc="lower right", fontsize=9, frameon=True, facecolor=SURFACE, edgecolor="none", framealpha=0.95, labelcolor=INK, labelspacing=0.25, handlelength=1.2)
    ax.set_ylabel("eval seq acc", color=INK, fontsize=11)

    # (0,1) same donor -> mod 6: accelerates
    ax = axes[0, 1]; style(ax, "mod-3 donor $\\to$ mod 6")
    s, a = load(f"mr5_{PRE}_constlr25000steps_nimsimple_max50000_evalevery50.jsonl")
    m = s <= 2000
    ax.plot(s[m], a[m], color=INK, linewidth=1.8, label="scratch (600)")
    s, a = load(f"mr5_{PRE}_constlr10000steps_nimsimple_max50000_evalevery25_initmod3pre6k.jsonl")
    m = s <= 2000
    ax.plot(s[m], a[m], color=BLUE, linewidth=1.8, label="mod-3 donor (200)")
    ax.set_xlim(0, 2000)
    ax.annotate("opens at 0.50\n(mod 3 is a factor of 6,\nnot a coarsening)",
                xy=(30, 0.51), xytext=(500, 0.30), color=BLUE, fontsize=11,
                arrowprops=dict(arrowstyle="->", color=BLUE, lw=1))
    ax.legend(loc="lower right", fontsize=9, frameon=True, facecolor=SURFACE, edgecolor="none", framealpha=0.95, labelcolor=INK, labelspacing=0.25, handlelength=1.2)

    # (0,2) X-ABL
    ax = axes[0, 2]; style(ax, "coset-sibling penalty")
    s, a = load(f"mr8_{PRE}_constlr6000steps_nimsimple_max50000_evalevery25.jsonl")
    ax.plot(s, a, color=INK, linewidth=1.8, label="scratch (3725)")
    s, a = load(f"mr8_{PRE}_constlr10000steps_nimsimple_max50000_evalevery25_sibpen1.jsonl")
    ax.plot(s, a, color=AQUA, linewidth=1.8, label="$\\lambda$=1 (5525)")
    s, a = load(f"mr8_{PRE}_constlr10000steps_nimsimple_max50000_evalevery25_sibpen5.jsonl")
    ax.plot(s, a, color=VIOLET, linewidth=1.8, label="$\\lambda$=5 (6875)")
    ax.set_xlim(0, 10000)
    ax.axhline(1 / 3, color=MUTED, linewidth=0.7, linestyle=":")
    ax.annotate("plateau gone,\nlearning slower",
                xy=(2500, 0.02), xytext=(300, 0.55), color=INK, fontsize=11,
                arrowprops=dict(arrowstyle="->", color=MUTED, lw=1))
    ax.legend(loc="lower right", fontsize=9, frameon=True, facecolor=SURFACE, edgecolor="none", framealpha=0.95, labelcolor=INK, labelspacing=0.25, handlelength=1.2)

    # (1,0) seeds
    ax = axes[1, 0]; style(ax, "seeds")
    for seed, shade in ((42, "#0b0b0b"), (43, "#52514e"), (44, "#898781")):
        s, a = load(f"mr8_410m_seed{seed}_lr3e-5_wd0.05_constlr6000steps_nimsimple_max50000_evalevery25.jsonl")
        ax.plot(s, a, color=shade, linewidth=1.5,
                label=f"scratch s{seed}" + (" (stuck)" if seed == 44 else ""))
    for seed, shade in ((42, "#2a78d6"), (43, "#5598e7"), (44, "#9ec5f4")):
        s, a = load(f"mr8_410m_seed{seed}_lr3e-5_wd0.05_constlr10000steps_nimsimple_max50000_evalevery25_initmod3pre6k.jsonl")
        ax.plot(s, a, color=shade, linewidth=1.5, label=f"installed s{seed}")
    ax.set_xlim(0, 10000)
    ax.legend(loc="lower right", fontsize=9, frameon=True, facecolor=SURFACE, edgecolor="none", framealpha=0.95, labelcolor=INK, labelspacing=0.25, handlelength=1.2, columnspacing=0.8, ncol=2)
    ax.set_ylabel("eval seq acc", color=INK, fontsize=11)
    ax.set_xlabel("training step", color=INK, fontsize=11)

    # (1,1) X-CURR phase 1
    ax = axes[1, 1]; style(ax, "operand-magnitude scaling")
    s, a = load(f"mr8_{PRE}_constlr3000steps_nimsimple_max10000_evalevery25.jsonl")
    ax.plot(s, a, color=AQUA, linewidth=1.8, label="max=10k (2350)")
    s, a = load(f"mr8_{PRE}_constlr6000steps_nimsimple_max50000_evalevery25.jsonl")
    ax.plot(s, a, color=MUTED, linewidth=1.2, label="max=50k (3725)")
    s, a = load(f"mr8_{PRE}_constlr15000steps_nimsimple_max500000_evalevery25.jsonl")
    ax.plot(s, a, color=INK, linewidth=1.8, label="max=500k (chance)")
    ax.set_xlim(0, 15000)
    ax.annotate("curriculum arms start\nfrom stage-1 checkpoints",
                xy=(2750, 0.95), xytext=(5000, 0.30), color=AQUA, fontsize=10,
                arrowprops=dict(arrowstyle="->", color=AQUA, lw=1))
    ax.legend(loc="center right", fontsize=9, frameon=True, facecolor=SURFACE, edgecolor="none", framealpha=0.95, labelcolor=INK, labelspacing=0.25, handlelength=1.2)
    ax.set_xlabel("training step", color=INK, fontsize=11)

    # (1,2) donors
    ax = axes[1, 2]; style(ax, "scaffold-donor training")
    s, a = load(f"mr8_{PRE}_constlr6000steps_scaffold_digitsum_evalevery50.jsonl")
    ax.plot(s, a, color=BLUE, linewidth=1.8, label="digit sum (1950)")
    s, a = load(f"mr10_{PRE}_constlr6000steps_scaffold_altsum22_evalevery50.jsonl")
    ax.plot(s, a, color=VIOLET, linewidth=1.8, label="alternating sum (final 0.91)")
    s, a = load(f"mr8_{PRE}_constlr6000steps_scaffold_firsttwo_evalevery50.jsonl")
    ax.plot(s, a, color=YELLOW, linewidth=1.8, label="first-two control (50)")
    s, a = load(f"mr8_{PRE}_constlr6000steps_nimsimple_max50000_evalevery25.jsonl")
    ax.plot(s, a, color=MUTED, linewidth=1.2, linestyle="--", label="ref: N mod 9 (3725)")
    ax.set_xlim(0, 6000)
    ax.legend(loc="lower right", fontsize=9, frameon=True, facecolor=SURFACE, edgecolor="none", framealpha=0.95, labelcolor=INK, labelspacing=0.25, handlelength=1.2)
    ax.set_xlabel("training step", color=INK, fontsize=11)

    fig.tight_layout()
    out = "new_result/plots/tx_round1.png"
    fig.savefig(out, dpi=180, facecolor=SURFACE, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main()
