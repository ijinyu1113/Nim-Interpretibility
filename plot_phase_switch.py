"""Figure for the mechanism-switch test (P-D, PROJECT_STATE 2.19).

Left: the swap at the cross-token position (pos1) — prefix-phase R2 rises at the
snap while the deep-layer count is evicted; mod-8 control flat-negative.
Right: the transition rule — fraction of per-token phase increments within 20 deg
of the predicted rotation 2*pi*ds(tok)/9; chance = 20/180; control at chance.

Reads: new_result/probes_ckpt/phase_switch_{mr8,mr7}.jsonl
Writes: new_result/plots/phase_switch.png
"""
import json

import matplotlib.pyplot as plt

from plot_style import setup_style
setup_style()

BLUE = "#1f77b4"    # phase (the arriving mechanism)
ORANGE = "#d95f02"  # count (the evicted scaffold)
GRAY = "#8c8c8c"    # mod-8 control / reference lines
PLATEAU, SNAP = 2300, 3500


def load(path):
    with open(path) as f:
        return [json.loads(ln) for ln in f]


mr8 = load("new_result/probes_ckpt/phase_switch_mr8.jsonl")
mr7 = load("new_result/probes_ckpt/phase_switch_mr7.jsonl")

s8 = [r["step"] for r in mr8]
ph8 = [r["per_pos"]["pos1"]["prefix_phase_L1316"] for r in mr8]
cnt8_deep = [max(r["per_pos"]["pos1"]["count_r2"][13:17]) for r in mr8]
cnt8_best = [r["per_pos"]["pos1"]["count_best"] for r in mr8]
inc8 = [r["increments"]["d01"]["frac_within_20"] for r in mr8]

s7 = [r["step"] for r in mr7]
ph7 = [r["per_pos"]["pos1"]["prefix_phase_L1316"] for r in mr7]
inc7 = [r["increments"]["d01"]["frac_within_20"] for r in mr7]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.0), constrained_layout=True)

for ax in (ax1, ax2):
    for i, (x, name) in enumerate(((PLATEAU, "plateau onset"), (SNAP, "snap"))):
        ax.axvline(x, color=GRAY, lw=0.8, ls=":", zorder=0)
        if ax is ax1:
            ha = "right" if i == 0 else "left"
            dx = -60 if i == 0 else 60
            ax.text(x + dx, -0.57, name, ha=ha, va="bottom", fontsize=11,
                    color="#555555")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#e8e8e8", lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.set_xlabel("training step")

# --- Left: the swap ---
ax1.plot(s8, ph8, color=BLUE, lw=2, marker="o", ms=4, zorder=3,
         label="prefix phase, L13–16 (mod 9)")
ax1.plot(s8, cnt8_deep, color=ORANGE, lw=2, marker="s", ms=4, zorder=3,
         label="prefix count, L13–16 (mod 9)")
ax1.plot(s8, cnt8_best, color=ORANGE, lw=1.2, ls="--", alpha=0.55, zorder=2,
         label="prefix count, best layer (mod 9)")
ax1.plot(s7, ph7, color=GRAY, lw=1.5, marker="^", ms=5, ls="-", zorder=2,
         label="prefix phase (mod-8 control)")
ax1.axhline(0, color="#bbbbbb", lw=0.8)
ax1.set_ylabel("held-out $R^2$ (cross-token position)")
ax1.set_ylim(-0.6, 1.05)
ax1.legend(frameon=True, facecolor="white", edgecolor="none", framealpha=0.92,
           fontsize=10, loc="center left", labelspacing=0.3)
ax1.annotate("0.98 → 0.78\nacross the snap",
             xy=(3500, 0.783), xytext=(4300, 0.58), fontsize=11, color=ORANGE,
             arrowprops=dict(arrowstyle="-", color=ORANGE, lw=0.8))

# --- Right: the transition rule ---
ax2.plot(s8, inc8, color=BLUE, lw=2, marker="o", ms=4, zorder=3,
         label="mod 9")
ax2.plot(s7, inc7, color=GRAY, lw=0, marker="^", ms=6, zorder=3,
         label="mod-8 control")
ax2.axhline(20 / 180, color=GRAY, lw=1, ls="--", zorder=1)
ax2.text(300, 20 / 180 + 0.02, "chance (20°/180°)", fontsize=11,
         color="#555555")
ax2.set_ylabel("frac. within 20° of predicted rotation")
ax2.set_ylim(0, 1.05)
ax2.legend(frameon=False, fontsize=11, loc="upper left")
for st, med, dx, dy in ((3250, "36°", -700, 0.03), (3500, "10°", 220, -0.04),
                        (6000, "4.3°", -850, 0.05)):
    y = inc8[s8.index(st)]
    ax2.annotate(f"med {med}", xy=(st, y), xytext=(st + dx, y + dy),
                 fontsize=11, color=BLUE)

fig.savefig("new_result/plots/phase_switch.png", dpi=200)
print("wrote new_result/plots/phase_switch.png")
