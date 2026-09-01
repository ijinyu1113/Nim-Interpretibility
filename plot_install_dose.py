"""Install dose-response figure: donor compression timeline (left) and mod-9
learning curves initialized from donor checkpoints at increasing depth (right).

Reads digitsum_attn_mr2.jsonl, donor behavior, and the 5 dose-arm metric files;
writes new_result/plots/install_dose.png.
"""
import json

import matplotlib.pyplot as plt

from plot_style import setup_style
setup_style()
import numpy as np

INK = "#0b0b0b"; MUTED = "#898781"; GRID = "#e1e0d9"; AXIS = "#c3c2b7"
AQUA = "#1baf7a"
BLUES = ["#9ec5f4", "#5598e7", "#256abf", "#104281"]  # donor 2k,3k,4k,6k
SURFACE = "#fcfcfb"
DEEP = range(13, 17)
MET = "new_result/purenum_metrics"


def load_beh(path):
    pts = {}
    for ln in open(path):
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


def main():
    rows = [json.loads(l) for l in open("new_result/probes_ckpt/digitsum_attn_mr2.jsonl")]
    dsteps = sorted({r["step"] for r in rows})
    deep = [np.mean([next(r for r in rows if r["step"] == s and r["kind"] == "digitsum"
                          and r["position"] == "final")["per_layer"][i] for i in DEEP])
            for s in dsteps]
    db_s, db_a = load_beh(f"{MET}/mr2_410m_seed42_lr3e-5_wd0.05_constlr6000steps_"
                          "nimsimple_max50000_evalevery50.jsonl")

    fig = plt.figure(figsize=(10.5, 4.2))
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 1.6], hspace=0.12, wspace=0.18)
    fig.patch.set_facecolor(SURFACE)

    axb = fig.add_subplot(gs[0, 0]); style(axb)
    axb.plot(db_s, db_a, color=INK, linewidth=1.8)
    axb.set_ylabel("donor eval acc\n(mod 3)", color=INK, fontsize=11)
    axb.set_ylim(-0.04, 1.06); axb.set_xlim(0, 6200); axb.set_xticklabels([])
    axb.axvline(1500, color=MUTED, linewidth=1, linestyle=(0, (4, 3)))
    axb.text(1600, 0.45, "converges\n(f95=1500)", color=MUTED, fontsize=11)
    axb.set_title("the mod-3 donor", color=INK, fontsize=13, fontweight="bold")

    axc = fig.add_subplot(gs[1, 0]); style(axc)
    axc.plot(dsteps, deep, color=INK, linewidth=1.8, marker="o", markersize=4)
    axc.axvline(1500, color=MUTED, linewidth=1, linestyle=(0, (4, 3)))
    axc.set_ylabel("digit-sum probe R$^2$\n(deep L13-16)", color=INK, fontsize=11)
    axc.set_xlabel("donor training step", color=INK, fontsize=12)
    axc.set_ylim(0.4, 1.02); axc.set_xlim(0, 6200)
    axc.annotate("scaffold discarded\nat convergence\n(0.97 $\\to$ 0.56)",
                 xy=(2050, 0.58), xytext=(3100, 0.70), color=INK, fontsize=10.5,
                 arrowprops=dict(arrowstyle="->", color=MUTED, lw=1))
    for s in dsteps:
        axc.plot([s], [deep[dsteps.index(s)]], marker="o", markersize=4, color=INK)

    axr = fig.add_subplot(gs[:, 1]); style(axr)
    base = (f"{MET}/mr8_410m_seed42_lr3e-5_wd0.05_constlr10000steps_"
            "nimsimple_max50000_evalevery25_initmod3pre")
    s, a = load_beh(f"{MET}/mr8_410m_seed42_lr3e-5_wd0.05_constlr6000steps_"
                    "nimsimple_max50000_evalevery25.jsonl")
    axr.plot(s, a, color=INK, linewidth=2.0, label="from scratch", zorder=5)
    s, a = load_beh(base + "1k.jsonl")
    axr.plot(s, a, color=AQUA, linewidth=1.8,
             label="donor@1000 (scaffold intact)", zorder=4)
    for (lab, col) in zip(["2k", "3k", "4k", "6k"], BLUES):
        s, a = load_beh(base + f"{lab}.jsonl")
        axr.plot(s, a, color=col, linewidth=1.6,
                 label=f"donor@{lab[0]}000 (compressed)")
    axr.axhline(1 / 3, color=MUTED, linewidth=0.8, linestyle=":")
    axr.set_xlim(0, 10000); axr.set_ylim(-0.04, 1.06)
    axr.set_xlabel("mod-9 training step", color=INK, fontsize=12)
    axr.set_ylabel("eval seq acc (mod 9)", color=INK, fontsize=12)
    axr.set_title("mod-9 training from donor checkpoints", color=INK, fontsize=13,
                  fontweight="bold")
    axr.legend(loc="lower right", fontsize=10, frameon=True, facecolor=SURFACE,
               edgecolor="none", framealpha=0.95, labelcolor=INK,
               labelspacing=0.3)

    fig.tight_layout()
    out = "new_result/plots/install_dose.png"
    fig.savefig(out, dpi=180, facecolor=SURFACE, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main()
