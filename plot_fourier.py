"""Fourier dynamics figure: period-decodability trajectories + output DFT
coset fraction for mod-9, mod-8 control, and the sibling-penalty run.

Writes new_result/plots/fourier_dynamics.png.
"""
import json

import matplotlib.pyplot as plt

from plot_style import setup_style
setup_style()
import numpy as np

INK = "#0b0b0b"; MUTED = "#898781"; GRID = "#e1e0d9"; AXIS = "#c3c2b7"
BLUE = "#2a78d6"; VIOLET = "#4a3aa7"
SURFACE = "#fcfcfb"
PRB = "new_result/probes_ckpt"
DEEP = range(13, 17)


def load(f):
    rows = [json.loads(l) for l in open(f"{PRB}/{f}")]
    out = {"step": [], "ds3": [], "ds9": [], "N10": [], "coset": []}
    for r in rows:
        out["step"].append(r["step"])
        for k in ("ds3", "ds9", "N10"):
            out[k].append(max(r["period_r2"][k][i] for i in DEEP))
        out["coset"].append(r["dft"]["coset_frac"])
    return {k: np.array(v) for k, v in out.items()}


def style(ax, title):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.7)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.set_ylim(-0.3, 1.06)
    ax.set_title(title, color=INK, fontsize=12, fontweight="bold")
    ax.set_xlabel("training step", color=INK, fontsize=11)


def panel(ax, d, events=()):
    ax.plot(d["step"], d["N10"], color=MUTED, linewidth=1.6, linestyle="--",
            label="$N$ period 10 (pretrained)")
    ax.plot(d["step"], d["ds3"], color=BLUE, linewidth=2.0, marker="o", markersize=3,
            label="ds period 3 (coarse)")
    ax.plot(d["step"], d["ds9"], color=VIOLET, linewidth=2.0, marker="o", markersize=3,
            label="ds period 9 (fine)")
    ax.plot(d["step"], d["coset"], color=INK, linewidth=1.3, linestyle=":",
            label="output DFT coset fraction")
    ax.axhline(0, color=AXIS, linewidth=0.7)
    for i, (x, lab) in enumerate(events):
        ax.axvline(x, color=MUTED, linewidth=0.9, linestyle=(0, (4, 3)))
        if i == 0:
            ax.text(x - 60, 1.04, lab, color=MUTED, fontsize=11, va="top",
                    ha="right")
        else:
            ax.text(x + 60, 1.04, lab, color=MUTED, fontsize=11, va="top")


def main():
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.9), sharey=True)
    fig.patch.set_facecolor(SURFACE)

    d = load("fourier_mr8.jsonl")
    ax = axes[0]; style(ax, "mod 9")
    panel(ax, d, events=[(2300, "plateau"), (3525, "snap")])
    ax.set_ylabel("decodability R$^2$  /  energy fraction", color=INK, fontsize=11)
    ax.legend(loc="center left", fontsize=10.5, frameon=True, facecolor=SURFACE,
              edgecolor="none", framealpha=0.95, labelcolor=INK,
              labelspacing=0.3)

    d = load("fourier_mr7.jsonl")
    ax = axes[1]; style(ax, "mod 8 (control)")
    panel(ax, d)

    d = load("fourier_sibpen1.jsonl")
    ax = axes[2]; style(ax, "sibling penalty")
    panel(ax, d)
    ax.annotate("period 9 rises\nbefore period 3", xy=(3500, 0.79), xytext=(5100, 0.28),
                color=VIOLET, fontsize=11,
                arrowprops=dict(arrowstyle="->", color=VIOLET, lw=1))

    fig.tight_layout()
    out = "new_result/plots/fourier_dynamics.png"
    fig.savefig(out, dpi=180, facecolor=SURFACE, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main()
