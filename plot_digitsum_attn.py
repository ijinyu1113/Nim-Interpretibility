"""E1+E2 figure: digit-sum probe R2 and attention globalization vs training,
mod 9 (mr8, global modulus) vs mod 8 (mr7, local modulus), with behavior on top.

Reads new_result/probes_ckpt/digitsum_attn_mr{7,8}.jsonl and the behavior
metrics jsonl; writes new_result/plots/digitsum_attn_e1e2.png.
"""
import json

import matplotlib.pyplot as plt

from plot_style import setup_style
setup_style()
import numpy as np

INK = "#0b0b0b"; MUTED = "#898781"; GRID = "#e1e0d9"; AXIS = "#c3c2b7"
BLUE = "#2a78d6"; AQUA = "#1baf7a"; VIOLET = "#4a3aa7"; ORANGE = "#eb6834"
SURFACE = "#fcfcfb"

DEEP = range(13, 17)   # hidden_states idx (L13-16): the decision site
EARLY = range(1, 5)    # L1-4


def load_probe(mr):
    rows = [json.loads(l) for l in open(f"new_result/probes_ckpt/digitsum_attn_mr{mr}.jsonl")]
    steps = sorted({r["step"] for r in rows})
    deep, early, att_e, att_l = [], [], [], []
    for s in steps:
        pl = next(r for r in rows if r["step"] == s and r["kind"] == "digitsum"
                  and r["position"] == "final")["per_layer"]
        deep.append(np.mean([pl[i] for i in DEEP]))
        early.append(np.mean([pl[i] for i in EARLY]))
        a = next(r for r in rows if r["step"] == s and r["kind"] == "attention")
        att_e.append(np.array(a["att_early"]).sum())
        att_l.append(np.array(a["att_last"]).sum())
    return np.array(steps), np.array(deep), np.array(early), np.array(att_e), np.array(att_l)


def load_behavior(mr):
    f = (f"new_result/purenum_metrics/mr{mr}_410m_seed42_lr3e-5_wd0.05_"
         "constlr6000steps_nimsimple_max50000_evalevery25.jsonl")
    pts = {}
    for ln in open(f):
        r = json.loads(ln)
        if "eval_eval_seq_acc" in r:
            pts[r["step"]] = r["eval_eval_seq_acc"]
    steps = np.array(sorted(pts)); return steps, np.array([pts[s] for s in steps])


def first_cross(steps, acc, thr):
    i = np.argmax(acc >= thr)
    return int(steps[i]) if acc[i] >= thr else None


def style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.7)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=9)


def event_lines(ax, events):
    for i, (x, label) in enumerate(events):
        ax.axvline(x, color=MUTED, linewidth=1, linestyle=(0, (4, 3)), zorder=1)
        if label:
            # first label to the left of its line, later ones to the right,
            # so adjacent event labels never run together
            if i == 0:
                ax.text(x, ax.get_ylim()[1], f"{label} ", color=MUTED,
                        fontsize=11, ha="right", va="top")
            else:
                ax.text(x, ax.get_ylim()[1], f" {label}", color=MUTED,
                        fontsize=11, ha="left", va="top")


def main():
    data = {mr: load_probe(mr) for mr in (8, 7)}
    beh = {mr: load_behavior(mr) for mr in (8, 7)}
    ev = {8: [(first_cross(*beh[8], 0.25), "plateau"),
              (first_cross(*beh[8], 0.9), "snap")],
          7: [(first_cross(*beh[7], 0.9), "snap")]}
    ev = {mr: [(x, l) for x, l in v if x is not None] for mr, v in ev.items()}

    fig, axes = plt.subplots(3, 2, figsize=(9.5, 7.8), sharex=True,
                             gridspec_kw={"height_ratios": [1, 1.4, 1.4]})
    fig.patch.set_facecolor(SURFACE)

    for col, mr in enumerate((8, 7)):
        steps, deep, early, att_e, att_l = data[mr]
        bs, ba = beh[mr]

        ax = axes[0, col]; style(ax)
        ax.plot(bs, ba, color=INK, linewidth=1.8)
        ax.set_ylim(-0.04, 1.06)
        if mr == 8:
            ax.axhline(1 / 3, color=MUTED, linewidth=0.8, linestyle=":")
            ax.text(150, 1 / 3 + 0.04, "1/3 (mod-3 coset)", color=MUTED,
                    fontsize=11, ha="right")
        ax.set_title(f"mod {mr + 1} ({'global: digit sum' if mr == 8 else 'local: last digits'})",
                     color=INK, fontsize=12, fontweight="bold")
        if col == 0:
            ax.set_ylabel("eval seq acc", color=INK, fontsize=12)
        event_lines(ax, ev[mr])

        ax = axes[1, col]; style(ax)
        ax.plot(steps, deep, color=BLUE, linewidth=2, marker="o", markersize=3.5)
        ax.plot(steps, early, color=AQUA, linewidth=2, marker="o", markersize=3.5)
        ax.set_ylim(0.45, 1.02)
        ax.text(steps[-1], deep[-1], "  deep L13-16", color=BLUE, fontsize=11, va="center")
        ax.text(steps[-1], early[-1], "  early L1-4", color=AQUA, fontsize=11, va="center")
        if col == 0:
            ax.set_ylabel("digit-sum probe R$^2$\n(final pos)", color=INK, fontsize=12)
        event_lines(ax, [(x, "") for x, _ in ev[mr]])

        ax = axes[2, col]; style(ax)
        ax.plot(steps, att_l, color=ORANGE, linewidth=2, marker="o", markersize=3.5)
        ax.plot(steps, att_e, color=VIOLET, linewidth=2, marker="o", markersize=3.5)
        ax.set_ylim(0, 23.5)
        ax.text(steps[-1], att_l[-1], "  last chunk", color=ORANGE, fontsize=11, va="center")
        ax.text(steps[-1], att_e[-1], "  earlier chunks", color=VIOLET, fontsize=11, va="center")
        if col == 0:
            ax.set_ylabel("attention mass onto numeral\n(sum, all heads L6-16)", color=INK, fontsize=12)
        ax.set_xlabel("training step", color=INK, fontsize=12)
        event_lines(ax, [(x, "") for x, _ in ev[mr]])

    for ax in axes.flat:
        ax.set_xlim(0, 6400)

    fig.tight_layout()
    out = "new_result/plots/digitsum_attn_e1e2.png"
    fig.savefig(out, dpi=180, facecolor=SURFACE)
    print("wrote", out)
    for mr in (8, 7):
        print(f"mod {mr + 1} events:", ev[mr])


if __name__ == "__main__":
    main()
