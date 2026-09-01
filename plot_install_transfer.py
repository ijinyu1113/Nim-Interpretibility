"""E3 install-transfer figure: mod-9 / mod-8 learning curves initialized from
mod-3 or mod-5 prefinetunes vs from-scratch baselines.

Reads new_result/purenum_metrics/*init{mod3pre,mod5pre}.jsonl + scratch dense
curves; writes new_result/plots/install_transfer.png.
"""
import json

import matplotlib.pyplot as plt

from plot_style import setup_style
setup_style()
import numpy as np

INK = "#0b0b0b"; MUTED = "#898781"; GRID = "#e1e0d9"; AXIS = "#c3c2b7"
BLUE = "#2a78d6"; AQUA = "#1baf7a"
SURFACE = "#fcfcfb"
BASE = "new_result/purenum_metrics"


def load(f):
    pts = {}
    for ln in open(f"{BASE}/{f}"):
        r = json.loads(ln)
        if "eval_eval_seq_acc" in r:
            pts[r["step"]] = r["eval_eval_seq_acc"]
    s = np.array(sorted(pts)); return s, np.array([pts[k] for k in s])


def style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.7)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.set_ylim(-0.04, 1.06)
    ax.set_xlabel("training step", color=INK, fontsize=12)


def main():
    tag10k = "constlr10000steps_nimsimple_max50000_evalevery25"
    tag6k = "constlr6000steps_nimsimple_max50000_evalevery25"
    pre = "410m_seed42_lr3e-5_wd0.05"
    curves = {
        "I1": load(f"mr8_{pre}_{tag10k}_initmod3pre.jsonl"),
        "I3": load(f"mr8_{pre}_{tag10k}_initmod5pre.jsonl"),
        "S9": load(f"mr8_{pre}_{tag6k}.jsonl"),
        "I2": load(f"mr7_{pre}_{tag10k}_initmod3pre.jsonl"),
        "S8": load(f"mr7_{pre}_{tag6k}.jsonl"),
    }

    fig, (axl, axr) = plt.subplots(1, 2, figsize=(10, 4.2), width_ratios=[1.5, 1])
    fig.patch.set_facecolor(SURFACE)

    ax = axl; style(ax)
    ax.axhline(1 / 3, color=MUTED, linewidth=0.8, linestyle=":")
    ax.text(7900, 1 / 3 - 0.07, "1/3 (mod-3 coset)", color=MUTED, fontsize=11, ha="right")
    for k, col, lab in [("S9", INK, "from scratch"),
                        ("I1", BLUE, "mod-3 donor (factor of 9)"),
                        ("I3", AQUA, "mod-5 donor (wrong factor)")]:
        s, a = curves[k]
        m = s <= 8000
        ax.plot(s[m], a[m], color=col, linewidth=1.8, label=lab)
    ax.set_xlim(0, 8000)
    ax.set_title("task: N mod 9 (global modulus)", color=INK, fontsize=13, fontweight="bold")
    ax.set_ylabel("eval seq acc", color=INK, fontsize=12)
    ax.legend(loc="upper left", fontsize=11, frameon=False, labelcolor=INK)
    ax.annotate("opens on the coset (0.33)",
                xy=(60, 0.335), xytext=(600, 0.60), color=BLUE, fontsize=11,
                arrowprops=dict(arrowstyle="->", color=BLUE, lw=1))
    ax.annotate("obstructs: f95 6250 vs 3725,\nceiling 0.92-0.95 vs 0.98",
                xy=(6200, 0.945), xytext=(4700, 0.50), color=BLUE, fontsize=11,
                arrowprops=dict(arrowstyle="->", color=BLUE, lw=1))

    ax = axr; style(ax)
    for k, col, lab in [("S8", INK, "from scratch"),
                        ("I2", BLUE, "init: mod-3 prefinetune")]:
        s, a = curves[k]
        m = s <= 2500
        ax.plot(s[m], a[m], color=col, linewidth=1.8, label=lab)
    ax.set_xlim(0, 2500)
    ax.set_title("task: N mod 8 (local modulus)", color=INK, fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=11, frameon=False, labelcolor=INK)
    ax.annotate("no coset introduced; mildly slowed\n(f95: 800 vs 300)",
                xy=(800, 0.95), xytext=(950, 0.55), color=BLUE, fontsize=11,
                arrowprops=dict(arrowstyle="->", color=BLUE, lw=1))

    fig.tight_layout()
    out = "new_result/plots/install_transfer.png"
    fig.savefig(out, dpi=180, facecolor=SURFACE, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main()
