"""Plot ft_mr7 scaling: model size (70m / 160m / 410m) × seed at max_remove=7 / mod 8.

Input:  new_result/ft_mr7_eval.jsonl
        rows: {size, seed, step, repo, move_acc, mod8_acc, mod4_acc, n}
Output: new_result/plots/ft_mr7_scaling_mod8.png  (median + IQR per size)
        new_result/plots/ft_mr7_scaling_mod4.png  (mod4 = lenient credit, |moves|=4)
"""
import json
import os
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

from plot_style import setup_style, HLINE_KW

setup_style()

JSONL = "new_result/ft_mr7_eval.jsonl"
OUT_DIR = "new_result/plots"

SIZES = ["70m", "160m", "410m"]
SIZE_COLORS = {"70m": "#1f77b4", "160m": "#ff7f0e", "410m": "#d62728"}
CHANCE_MOD8 = 1.0 / 8
CHANCE_MOD4 = 1.0 / 4


def load():
    rows = []
    with open(JSONL) as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def aggregate(rows, metric):
    """{(size, step): [values across seeds]} -> median, q1, q3, n."""
    bucket = defaultdict(list)
    for r in rows:
        bucket[(r["size"], r["step"])].append(r[metric])
    out = {s: {"step": [], "med": [], "q1": [], "q3": [], "n": []} for s in SIZES}
    for (size, step), vals in bucket.items():
        if size not in out:
            continue
        out[size]["step"].append(step)
        out[size]["med"].append(np.median(vals))
        out[size]["q1"].append(np.percentile(vals, 25))
        out[size]["q3"].append(np.percentile(vals, 75))
        out[size]["n"].append(len(vals))
    for size in SIZES:
        order = np.argsort(out[size]["step"])
        for k in ("step", "med", "q1", "q3", "n"):
            out[size][k] = np.array(out[size][k])[order]
    return out


def plot(agg, metric_label, chance, outpath):
    fig, ax = plt.subplots(figsize=(5.0, 3.0))
    for size in SIZES:
        d = agg[size]
        if len(d["step"]) == 0:
            continue
        x = d["step"] / 1000.0
        n_seeds = int(d["n"].max()) if len(d["n"]) else 0
        ax.plot(x, d["med"], color=SIZE_COLORS[size],
                label=f"{size} (n={n_seeds})")
        if np.any(d["q3"] > d["q1"]):
            ax.fill_between(x, d["q1"], d["q3"],
                            color=SIZE_COLORS[size], alpha=0.18, linewidth=0)
    ax.axhline(chance, **HLINE_KW)
    ax.set_xlabel(r"Training step (${\times}10^3$)")
    ax.set_ylabel(metric_label)
    ax.set_ylim(-0.02, 1.05)
    ax.set_xlim(0, 150)
    ax.grid(alpha=0.14, linewidth=0.5)

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(SIZES),
               bbox_to_anchor=(0.5, 1.05), frameon=False)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {outpath}")


def main():
    rows = load()
    print(f"Loaded {len(rows)} rows from {JSONL}")
    coverage = defaultdict(set)
    for r in rows:
        coverage[r["size"]].add(r["seed"])
    for size in SIZES:
        seeds = sorted(coverage.get(size, []))
        print(f"  {size}: seeds={seeds} (n={len(seeds)})")

    agg_mod8 = aggregate(rows, "mod8_acc")
    agg_mod4 = aggregate(rows, "mod4_acc")

    plot(agg_mod8, "Eval accuracy (mod 8, exact)", CHANCE_MOD8,
         os.path.join(OUT_DIR, "ft_mr7_scaling_mod8.png"))
    plot(agg_mod4, "Eval accuracy (mod 4, lenient)", CHANCE_MOD4,
         os.path.join(OUT_DIR, "ft_mr7_scaling_mod4.png"))


if __name__ == "__main__":
    main()
