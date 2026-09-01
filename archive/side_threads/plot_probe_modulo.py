"""Plot per-layer probe accuracy for `final_pile mod (mr+1)` recovery.

Reads new_result/probes/probe_mr*.jsonl and plots one line per MR, with a
chance baseline. High peak = info is encoded; flat-at-chance = it isn't.
"""
import glob
import json
import os
import re

import numpy as np
import matplotlib.pyplot as plt

from plot_style import setup_style, HLINE_KW

setup_style()

PROBE_DIR = "new_result/probes"
OUT_PATH = "new_result/plots/probe_modulo.png"

MR_COLORS = {3: "#1f77b4", 6: "#d62728"}


def load_probe(path):
    rows = [json.loads(line) for line in open(path)]
    rows.sort(key=lambda r: r["layer"])
    # Backward-compat: older files used acc_mean/acc_std, newer ones split out cveval + train2eval
    cveval_mean_key = "acc_cveval_mean" if "acc_cveval_mean" in rows[0] else "acc_mean"
    cveval_std_key = "acc_cveval_std" if "acc_cveval_std" in rows[0] else "acc_std"
    out = {
        "layer": np.array([r["layer"] for r in rows]),
        "acc_cveval": np.array([r[cveval_mean_key] for r in rows]),
        "std_cveval": np.array([r[cveval_std_key] for r in rows]),
        "chance": rows[0]["chance"],
        "modulus": rows[0]["modulus"],
    }
    if "acc_train2eval" in rows[0]:
        out["acc_t2e"] = np.array([r["acc_train2eval"] for r in rows])
    return out


def main():
    files = sorted(glob.glob(os.path.join(PROBE_DIR, "probe_mr*.jsonl")))
    if not files:
        print(f"No probe files in {PROBE_DIR}")
        return

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for f in files:
        m = re.search(r"probe_mr(\d+)", f)
        if not m:
            continue
        mr = int(m.group(1))
        d = load_probe(f)
        color = MR_COLORS.get(mr, "grey")
        # (A) CV-on-eval — solid
        ax.plot(d["layer"], d["acc_cveval"], color=color, marker="o", ls="-",
                label=f"mr={mr} cv-on-eval")
        ax.fill_between(d["layer"], d["acc_cveval"] - d["std_cveval"],
                        d["acc_cveval"] + d["std_cveval"],
                        color=color, alpha=0.18, linewidth=0)
        # (B) Train→eval transfer — dashed (only if present)
        if "acc_t2e" in d:
            ax.plot(d["layer"], d["acc_t2e"], color=color, marker="s", ls="--",
                    label=f"mr={mr} train→eval")
        ax.axhline(d["chance"], color=color, ls=":", lw=0.8, alpha=0.6)

    ax.axhline(1.0, color="grey", lw=0.5, ls="-", alpha=0.4)
    ax.set_xlabel("Layer index (0 = embeddings)")
    ax.set_ylabel("Probe accuracy")
    ax.set_ylim(-0.02, 1.05)
    ax.set_title("Linear probe: final_pile mod (mr+1)  —  cv-on-eval vs train→eval transfer")
    ax.grid(alpha=0.14, linewidth=0.5)
    ax.legend(loc="best", frameon=False, fontsize=8)
    fig.tight_layout()
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    fig.savefig(OUT_PATH, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {OUT_PATH}")


if __name__ == "__main__":
    main()
