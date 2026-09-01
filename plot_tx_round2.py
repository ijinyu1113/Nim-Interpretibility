"""Round-2 figure: the scaffold 2x2 double dissociation and the magnitude
curriculum. Writes new_result/plots/tx_round2.png.
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
P = "410m_seed42_lr3e-5_wd0.05"


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
    ax.set_xlabel("training step", color=INK, fontsize=11)


def main():
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 4.0))
    fig.patch.set_facecolor(SURFACE)

    ax = axes[0]; style(ax, "target: N mod 9")
    s, a = load(f"mr8_{P}_constlr6000steps_nimsimple_max50000_evalevery25.jsonl")
    ax.plot(s, a, color=INK, linewidth=1.8, label="scratch (3725)")
    s, a = load(f"mr8_{P}_constlr10000steps_nimsimple_max50000_evalevery25_initmod3pre6k.jsonl")
    m = s <= 8000
    ax.plot(s[m], a[m], color=MUTED, linewidth=1.5, linestyle="--",
            label="mod-3 donor (6250)")
    s, a = load(f"mr8_{P}_constlr12000steps_nimsimple_max50000_evalevery25_initdsum6k.jsonl")
    m = s <= 8000
    ax.plot(s[m], a[m], color=BLUE, linewidth=2.0,
            label="digit-sum donor (200)")
    s, a = load(f"mr8_{P}_constlr12000steps_nimsimple_max50000_evalevery25_initasum6k.jsonl")
    m = s <= 8000
    ax.plot(s[m], a[m], color=VIOLET, linewidth=1.5,
            label="alt-sum donor (3575)")
    s, a = load(f"mr8_{P}_constlr12000steps_nimsimple_max50000_evalevery25_initf2d6k.jsonl")
    m = s <= 8000
    ax.plot(s[m], a[m], color=YELLOW, linewidth=1.5,
            label="first-two control (2625)")
    ax.set_xlim(0, 8000)
    ax.legend(loc="lower right", fontsize=9, frameon=True, facecolor=SURFACE, edgecolor="none", framealpha=0.95, labelcolor=INK, labelspacing=0.25, handlelength=1.2)
    ax.set_ylabel("eval seq acc", color=INK, fontsize=11)

    ax = axes[1]; style(ax, "target: N mod 11")
    s, a = load(f"mr10_{P}_constlr30000steps_nimsimple_max50000_evalevery50.jsonl")
    m = s <= 10000
    ax.plot(s[m], a[m], color=INK, linewidth=1.8, label="scratch (6550)")
    s, a = load(f"mr10_{P}_constlr12000steps_nimsimple_max50000_evalevery25_initasum6k.jsonl")
    ax.plot(s, a, color=VIOLET, linewidth=2.0,
            label="alt-sum donor (800)")
    s, a = load(f"mr10_{P}_constlr12000steps_nimsimple_max50000_evalevery25_initdsum6k.jsonl")
    ax.plot(s, a, color=BLUE, linewidth=1.5,
            label="digit-sum donor (4475)")
    ax.set_xlim(0, 10000)
    ax.legend(loc="lower right", fontsize=9, frameon=True, facecolor=SURFACE, edgecolor="none", framealpha=0.95, labelcolor=INK, labelspacing=0.25, handlelength=1.2)

    ax = axes[2]; style(ax, "6-digit curriculum")
    s, a = load(f"mr8_{P}_constlr15000steps_nimsimple_max500000_evalevery25.jsonl")
    ax.plot(s, a, color=INK, linewidth=1.8, label="scratch @500k (chance)")
    s, a = load(f"mr8_{P}_constlr15000steps_nimsimple_max500000_evalevery25_initcurrA.jsonl")
    ax.plot(s, a, color=BLUE, linewidth=2.0,
            label="A: converged stage-1 (600)")
    s, a = load(f"mr8_{P}_constlr15000steps_nimsimple_max500000_evalevery25_initcurrB.jsonl")
    ax.plot(s, a, color=AQUA, linewidth=1.8,
            label="B: pre-snap stage-1 (0.66)")
    ax.axhline(1 / 3, color=MUTED, linewidth=0.7, linestyle=":")
    ax.set_xlim(0, 15000)
    ax.annotate("B dwells at the\ninherited coset",
                xy=(4000, 0.35), xytext=(4500, 0.62), color=AQUA, fontsize=11,
                arrowprops=dict(arrowstyle="->", color=AQUA, lw=1))
    ax.legend(loc="lower right", fontsize=9, frameon=True, facecolor=SURFACE, edgecolor="none", framealpha=0.95, labelcolor=INK, labelspacing=0.25, handlelength=1.2)

    fig.tight_layout()
    out = "new_result/plots/tx_round2.png"
    fig.savefig(out, dpi=180, facecolor=SURFACE, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main()
