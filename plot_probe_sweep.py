"""The two mechanistic figures from the cross-checkpoint sweep.

Fig A (time): mr=8 model — task accuracy, behavioral mod-3 agreement, probe
mod-3, probe mod-9, probe mod-8 across training. The staircase and its
mechanism on one axis.

Fig B (depth): (a) mr=8 logit-lens ladder at 4 stages — truncates at mod-3
during the plateau, upgrades in place after; (b) mr=7 converged ladder;
(c) BASE model probes per layer — the pretrained local basis (parity ~1.0,
partial mod-8) with nothing for mod-3/mod-9.

Outputs: new_result/plots/probe_time.png, new_result/plots/probe_depth.png
"""
import json
import os

import numpy as np
import matplotlib.pyplot as plt

from plot_style import setup_style

setup_style()

D = "new_result/probes_ckpt"
OUT1 = "new_result/plots/probe_time.png"
OUT2 = "new_result/plots/probe_depth.png"
OUT3 = "new_result/plots/probe_depth_extra.png"


def load(f):
    return [json.loads(l) for l in open(f"{D}/{f}")]


mr8 = load("probe_mr8_dense.jsonl")
mr7 = load("probe_mr7_dense.jsonl")
base = load("probe_base.jsonl")


def probe_traj(rows, pos, target):
    d = {r["step"]: r["best_acc"] for r in rows
         if r["kind"] == "probe" and r["position"] == pos and r["target_mod"] == target}
    s = sorted(d)
    return np.array(s), np.array([d[k] for k in s])


def beh_traj(rows, key):
    d = {r["step"]: r.get(key) for r in rows if r["kind"] == "behavior"}
    s = sorted(d)
    return np.array(s), np.array([d[k] for k in s])


def fig_time():
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    ax.axvspan(2400, 3300, color="#f5e6c8", alpha=0.55, zorder=0)
    ax.text(2850, 1.03, "plateau", ha="center", fontsize=11, color="#8a6d1d")

    s, v = beh_traj(mr8, "exact")
    ax.plot(s, v, color="#444", lw=2.2, label="task accuracy (exact mod 9)")
    s, v = beh_traj(mr8, "agree_mod3")
    ax.plot(s, v, color="#d62728", lw=1.6, ls="--", label="behavior: agrees mod 3")
    s, v = probe_traj(mr8, "final", 3)
    ax.plot(s, v, color="#1f77b4", lw=1.6, marker="o", ms=3.5,
            label="probe: N mod 3")
    s, v = probe_traj(mr8, "final", 9)
    ax.plot(s, v, color="#2ca02c", lw=1.6, marker="s", ms=3.5,
            label="probe: N mod 9")
    s, v = probe_traj(mr8, "final", 8)
    ax.plot(s, v, color="#9467bd", lw=1.2, marker="^", ms=3, alpha=0.7,
            label="probe: N mod 8 (never built)")

    ax.axhline(1/3, color="#999", ls=":", lw=0.9)
    ax.text(150, 1/3 - 0.045, "1/3", fontsize=11, color="#999")
    ax.axhline(1/9, color="#bbb", ls=":", lw=0.9)
    ax.set_xlim(0, 6000)
    ax.set_ylim(0, 1.06)
    ax.set_xlabel("Training step")
    ax.set_ylabel("accuracy / decodability")
    ax.grid(alpha=0.14, lw=0.5)
    ax.legend(fontsize=10, loc="lower right", frameon=True, facecolor="white",
              edgecolor="none", framealpha=0.92, labelspacing=0.3)
    fig.tight_layout()
    os.makedirs(os.path.dirname(OUT1), exist_ok=True)
    fig.savefig(OUT1, bbox_inches="tight", dpi=160)
    plt.close(fig)
    print(f"Saved {OUT1}")


def lens_ladder(rows, step, key):
    d = {r["layer"]: r[key] for r in rows if r["kind"] == "lens" and r["step"] == step}
    s = sorted(d)
    return np.array(s), np.array([d[k] for k in s])


def fig_depth():
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    stages = [(1000, "pre-plateau", "#bbb"), (2750, "plateau", "#d62728"),
              (3500, "snap", "#ff7f0e"), (6000, "converged", "#1f77b4")]
    for step, label, c in stages:
        L, v = lens_ladder(mr8, step, "exact")
        ax.plot(L, v, color=c, lw=1.7, marker="o", ms=3, label=f"exact @ {label}")
        L, v3 = lens_ladder(mr8, step, "agree_mod3")
        ax.plot(L, v3, color=c, lw=1.1, ls="--", alpha=0.7)
    ax.axhline(1 / 3, color="#999", ls=":", lw=0.8)
    ax.set_xlabel("layer")
    ax.set_ylabel("lens agreement (dashed: mod-3)")
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.14, lw=0.5)
    ax.legend(fontsize=10, frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT2, bbox_inches="tight", dpi=180)
    plt.close(fig)
    print(f"Saved {OUT2}")


def fig_depth_extra():
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.9))
    ax = axes[0]
    for key, label, c in [("agree_mod2", "mod-2", "#1f77b4"),
                          ("agree_mod4", "mod-4", "#ff7f0e"),
                          ("exact", "mod-8 exact", "#d62728")]:
        L, v = lens_ladder(mr7, 6000, key)
        ax.plot(L, v, color=c, lw=1.7, marker="o", ms=3, label=label)
    ax.set_title("mod 8, converged")
    ax.set_xlabel("layer")
    ax.set_ylabel("lens agreement")
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.14, lw=0.5)
    ax.legend(fontsize=11, frameon=True, facecolor="white", edgecolor="none", framealpha=0.92, loc="upper left", labelspacing=0.3)

    ax = axes[1]
    colors = {2: "#1f77b4", 8: "#9467bd", 3: "#d62728", 9: "#2ca02c"}
    for r in base:
        if r["kind"] == "probe" and r["position"] == "final":
            m = r["target_mod"]
            ax.plot(range(len(r["per_layer"])), r["per_layer"], color=colors[m],
                    lw=1.7, marker="o", ms=3, label=f"N mod {m} (chance {r['chance']:.2f})")
    ax.set_title("base model (no fine-tuning)")
    ax.set_xlabel("layer")
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.14, lw=0.5)
    ax.legend(fontsize=11, frameon=True, facecolor="white", edgecolor="none", framealpha=0.92, loc="upper left", labelspacing=0.3)

    fig.tight_layout()
    fig.savefig(OUT3, bbox_inches="tight", dpi=180)
    plt.close(fig)
    print(f"Saved {OUT3}")


if __name__ == "__main__":
    fig_time()
    fig_depth()
    fig_depth_extra()
