"""Dwell-state probe figure: deep-layer digit-sum R2 across training for all
five donor arms + scratch, with each run's escape step marked. The refutation
figure: no scaffold recovery precedes escape; compression deepens through it.

Writes new_result/plots/dwell_probe.png.
"""
import json

import matplotlib.pyplot as plt

from plot_style import setup_style
setup_style()
import numpy as np

INK = "#0b0b0b"; MUTED = "#898781"; GRID = "#e1e0d9"; AXIS = "#c3c2b7"
AQUA = "#1baf7a"
BLUES = ["#9ec5f4", "#6da7ec", "#3987e5", "#1c5cab"]
SURFACE = "#fcfcfb"
DEEP = range(13, 17)
PRB = "new_result/probes_ckpt"
ESCAPE = {"scr": 3025, "1k": 4550, "2k": 4700, "3k": 3400, "4k": 3500, "6k": 2775}


def deep_r2(path):
    rows = [json.loads(l) for l in open(path)]
    ds = {r["step"]: np.mean([r["per_layer"][i] for i in DEEP]) for r in rows
          if r["kind"] == "digitsum" and r["position"] == "final"}
    s = sorted(ds); return np.array(s), np.array([ds[k] for k in s])


def main():
    fig, ax = plt.subplots(figsize=(7, 4.0))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.7)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=9)

    series = [("scr", f"{PRB}/digitsum_attn_mr8.jsonl", INK, "scratch"),
              ("1k", f"{PRB}/digitsum_attn_dose1k.jsonl", AQUA, "donor@1000 (uncompressed)")]
    for arm, col in zip(("2k", "3k", "4k", "6k"), BLUES):
        series.append((arm, f"{PRB}/digitsum_attn_dose{arm}.jsonl", col,
                       f"donor@{arm[0]}000 (compressed)"))

    for arm, path, col, label in series:
        s, r2 = deep_r2(path)
        ax.plot(s, r2, color=col, linewidth=1.8, marker="o", markersize=3, label=label)
        esc = ESCAPE[arm]
        i = np.searchsorted(s, esc)
        y = r2[min(i, len(r2) - 1)]
        ax.plot([esc], [y], marker="*", markersize=13, color=col, zorder=5,
                markeredgecolor=SURFACE, markeredgewidth=0.8)

    ax.set_xlim(0, 8200); ax.set_ylim(0.3, 1.0)
    ax.set_xlabel("mod-9 training step", color=INK, fontsize=12)
    ax.set_ylabel("digit-sum probe R$^2$ (deep L13-16)", color=INK, fontsize=12)
    ax.legend(loc="upper right", fontsize=10, frameon=True, facecolor=SURFACE,
              edgecolor="none", framealpha=0.95, labelcolor=INK,
              labelspacing=0.3)
    fig.tight_layout()
    out = "new_result/plots/dwell_probe.png"
    fig.savefig(out, dpi=180, facecolor=SURFACE, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main()
