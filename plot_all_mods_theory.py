"""Master figure: every modulus we have (nimsimple, constant LR, correct holdout,
max=50000), grouped by the global-computation theory's three predicted behaviors,
with the prompt and theory printed on the figure.

Output: new_result/plots/all_mods_theory.png
"""
import json
import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from plot_style import setup_style

setup_style()

METRICS = "new_result/purenum_metrics"
OUT = "new_result/plots/all_mods_theory.png"

# mr -> mod. File tag is constlr25000steps for mr<=8, constlr30000steps for mr>=9.
def load(mr):
    for steps in ("25000", "30000"):
        p = f"{METRICS}/mr{mr}_410m_seed42_lr3e-5_wd0.05_constlr{steps}steps_nimsimple_max50000_evalevery50.jsonl"
        if os.path.isfile(p):
            by = {}
            for ln in open(p):
                r = json.loads(ln); s = r.get("step")
                if s is not None and "eval_eval_seq_acc" in r: by[s] = r["eval_eval_seq_acc"]
            st = sorted(by)
            return np.array(st), np.array([by[s] for s in st])
    return np.array([]), np.array([])


# (mr, mod, color)
SNAP = [(3, 4, "#1f77b4"), (4, 5, "#ff7f0e"), (7, 8, "#2ca02c"), (9, 10, "#d62728"), (15, 16, "#9467bd")]
PLAT = [(5, 6, "#1f77b4"), (8, 9, "#ff7f0e"), (11, 12, "#2ca02c"), (13, 14, "#d62728"), (14, 15, "#9467bd")]
SLOW = [(6, 7, "#1f77b4"), (10, 11, "#ff7f0e")]

PROMPT = ('PROMPT (nimsimple):  "Each player can take between 1 and {mr} coins on their turn. '
          'There are {N} coins. On this turn, the player should take"\n'
          'ANSWER:  "{N mod (mr+1)}"      (bare-digit residue; correct holdout = disjoint pile values train/eval)')

THEORY = (
    "THEORY — the answer is N mod m, and base-10 digit structure decides the learning dynamics:\n"
    "  • LOCAL  (m's only prime factors are 2 and 5):  N mod m sits in the last digit(s)  ->  EASY  ->  SNAP, no plateau.\n"
    "      e.g. mod 5,10 = last digit;  mod 4,8,16 = last 2-3 digits;  mod 10 = 2·5 SNAPS even though composite (the key test).\n"
    "  • GLOBAL (m has a factor coprime to 10: 3,7,...):  N mod m needs ALL digits (digit sum for 3/9)  ->  HARD.\n"
    "      composite (mod 6,12,14,15): learns the cheap LOCAL factor first -> PLATEAUS at that coset -> then grinds the global part.\n"
    "      prime power (mod 9): plateaus at the coarse version (mod-3).   prime coprime to 10 (mod 7,11): no coset -> SLOW ramp.\n"
    "  • Plateau DEPTH tracks the global factor's hardness: mod 14 (factor 7) dwells ~3x longer than mod 15 (factor 3)."
)


def panel(ax, group, title, logx=True, legend_loc="lower right", ncol=2):
    for mr, mod, c in group:
        s, v = load(mr)
        if s.size:
            mask = s >= 1
            ax.plot(s[mask], v[mask], color=c, lw=1.8, marker="o", ms=2.6, alpha=0.9,
                    label=f"mod {mod}")
    ax.set_title(title, fontsize=13)
    if logx:
        # Shared axis across all panels (cross-panel comparability is the
        # figure's argument): linear through the window containing all stage
        # structure, log-compressed converged tail.
        ax.set_xscale("symlog", linthresh=4000, linscale=5)
        ax.set_xlim(0, 30000)
        ax.set_xticks([0, 1000, 2000, 3000, 4000, 30000])
        ax.xaxis.set_major_formatter(plt.FuncFormatter(
            lambda v, p: f"{v/1000:.0f}k" if v >= 1000 else f"{int(v)}"))
        ax.minorticks_off()
    ax.set_ylim(-0.02, 1.05)
    ax.grid(alpha=0.14, lw=0.5, which="both")
    ax.legend(fontsize=11, loc=legend_loc, frameon=True, facecolor="white",
              edgecolor="none", framealpha=0.92, ncol=ncol,
              handlelength=1.4, columnspacing=1.0, labelspacing=0.3)


def main():
    # Paper version: panels only — the prompt/theory text lives in the caption
    # and Table 1, not inside the PNG (print-legibility feedback 2026-08-19).
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 3.9), constrained_layout=True)

    panel(ax1, SNAP, "local ($m \\mid 10^k$): snap\nmod 4, 5, 8, 10, 16",
          legend_loc="center right", ncol=1)
    ax1.set_ylabel("held-out exact accuracy")
    panel(ax2, PLAT, "global factor (3 or 7): plateau\nmod 6, 9, 12, 14, 15",
          legend_loc="upper left", ncol=1)
    # middle legend: the converged top-right corner is the only empty region
    ax2.legend(fontsize=9.5, loc="center right", bbox_to_anchor=(1.0, 0.45),
               frameon=True, facecolor="white", edgecolor="none",
               framealpha=0.95, handlelength=1.2, labelspacing=0.25)
    panel(ax3, SLOW, "global prime: slow ramp\nmod 7, 11",
          legend_loc="upper left", ncol=1)

    fig.supxlabel("Training step (linear to 4k, log-compressed beyond)",
                  fontsize=12)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"Saved {OUT}")


if __name__ == "__main__":
    main()
