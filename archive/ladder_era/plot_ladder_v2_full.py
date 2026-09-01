"""Plot the full 8-step prompt ladder (0, 2, 2b, 3, 4, 5, 6) per mr.
All curves use eval_seq_acc. Bottom panel lists one concrete prompt per step
so the figure is self-contained.

Output: new_result/plots/ladder_v2_full.png
"""
import json
import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from plot_style import setup_style

setup_style()

METRICS_DIR = "new_result/purenum_metrics"
OUT = "new_result/plots/ladder_v2_full.png"
MRS = [3, 4, 5, 6, 7, 8]

LADDER = [
    (0,    "modarith_subtract_noholdout_max400_evalevery1",   "0 bare math sum-paren",      "#08306b"),
    (2,    "prompt_ladder_step2_max400_noholdout_evalevery1", "2 bare math range hint",      "#2171b5"),
    ("2a", "ladderC_step2a",                                  "2a NL range + math eqn",      "#4292c6"),
    ("2b", "ladderC_step2b",                                  "2b NL + comma moves",         "#9ecae1"),
    (3,    "ladderC_step3",                                   "3 anon 'Someone take'",       "#fdae6b"),
    (4,    "ladderC_step4",                                   "4 Leo/Sultan no headers",     "#f16913"),
    (5,    "purenums_paperC_singleline",                      "5 paperC single-line",        "#a63603"),
    (6,    "purenums_paperC",                                 "6 paper exact multi-line",    "#67000d"),
]

# Same example across all steps: mr=6 (mod 7), pile=198, moves=[6,5,3] -> ans=2
EXAMPLE_PROMPTS = [
    (0,    "198 - (6 + 5 + 3) mod 7 = "),
    (2,    "range [1, 6]. 198 - 6 - 5 - 3 = "),
    ("2a", "Each player can play between 1 and 6. The current pile size: 198 - 6 - 5 - 3 = "),
    ("2b", "There are 198 coins. Each player can take between 1 and 6 coins on their turn. Previous moves: 6, 5, 3. Now it's the next player's turn."),
    (3,    "There are 198 coins. Each player can take between 1 and 6 coins on their turn. Someone take 6 coins. Someone take 5 coins. Someone take 3 coins. Now it's the next player's turn."),
    (4,    "There are 198 coins. Leo and Sultan take turns. Each player can take between 1 and 6 coins on their turn. Leo take 6 coins. Sultan take 5 coins. Leo take 3 coins. Now it's Sultan's turn."),
    (5,    "You are playing the game of nim. There are 198 coins. Leo and Sultan take turns. Each player can take between 1 and 6 coins on their turn.  So far: Leo take 6 coins. Sultan take 5 coins. Leo take 3 coins. Now it's Sultan's turn."),
    (6,    "You are playing the game of nim. There are 198 coins.\\nLeo and Sultan take turns.\\nEach player can take between 1 and 6 coins on their turn.\\n\\nSo far:\\nLeo take 6 coins.\\nSultan take 5 coins.\\nLeo take 3 coins.\\nNow it's Sultan's turn."),
]
STEP_COLORS = {step: color for step, _, _, color in LADDER}


def load(path):
    by_step = {}
    if not os.path.isfile(path):
        return np.array([]), np.array([])
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            st = r.get("step")
            if st is None: continue
            if "eval_eval_seq_acc" in r:
                by_step[st] = r["eval_eval_seq_acc"]
    steps = sorted(by_step.keys())
    return np.array(steps), np.array([by_step[s] for s in steps])


def wrap(s, width=170):
    """Hard-wrap a long single-line prompt at word boundaries."""
    out = []
    while len(s) > width:
        cut = s.rfind(" ", 0, width)
        if cut == -1:
            cut = width
        out.append(s[:cut])
        s = s[cut + 1:]
    out.append(s)
    return out


def main():
    fig = plt.figure(figsize=(17, 13.5))
    gs = gridspec.GridSpec(
        3, 3, figure=fig,
        height_ratios=[3.5, 3.5, 5.0], hspace=0.40, wspace=0.10,
    )

    # ---- 2x3 grid of accuracy curves ----
    for idx, mr in enumerate(MRS):
        ax = fig.add_subplot(gs[idx // 3, idx % 3])
        chance = 1.0 / (mr + 1)
        for step_id, tag, label, color in LADDER:
            f = f"{METRICS_DIR}/mr{mr}_410m_seed42_lr3e-5_wd0.05_oldcfg300ep_{tag}.jsonl"
            s, v = load(f)
            if s.size == 0:
                continue
            mask = (s >= 1) & (s <= 30000)
            ax.plot(s[mask], v[mask], color=color, lw=1.2,
                    marker="o", ms=2.5, alpha=0.9, label=label)
        ax.axhline(chance, color="grey", ls=":", lw=0.9, alpha=0.7,
                   label=f"chance ({chance:.3f})")
        ax.set_title(f"mr={mr}  (mod {mr+1})")
        ax.set_xscale("log")
        ax.set_xlim(50, 30000)
        ax.set_ylim(-0.02, 1.05)
        ax.grid(alpha=0.14, linewidth=0.5, which="both")
        ax.set_xlabel("Training step (log)")
        if mr in (3, 6):
            ax.set_ylabel("eval seq_acc")
        if mr == 3:
            ax.legend(fontsize=7, loc="lower right", frameon=False)

    # ---- Bottom panel: one example prompt per step ----
    ax_txt = fig.add_subplot(gs[2, :])
    ax_txt.axis("off")
    ax_txt.set_xlim(0, 1)
    ax_txt.set_ylim(0, 1)
    ax_txt.set_title(
        "Example prompts (mr=6, pile=198, moves=[6,5,3], answer=\"2\")",
        loc="left", fontsize=11, pad=2,
    )

    # Render each step's prompt, one block per step, with step header in color.
    y = 0.95
    header_dy = 0.045  # space taken by step header
    line_dy = 0.032    # space per wrapped prompt line
    for step_id, prompt in EXAMPLE_PROMPTS:
        color = STEP_COLORS[step_id]
        ax_txt.text(0.005, y, f"[step {step_id}]", fontsize=9,
                    color=color, fontweight="bold", family="monospace",
                    va="top", transform=ax_txt.transAxes)
        # Render prompt with literal \n shown as `\n` (do not actually break).
        lines = wrap(prompt, width=170)
        for li, line in enumerate(lines):
            ax_txt.text(0.075, y - li * line_dy * 0.95, line,
                        fontsize=8.2, family="monospace",
                        va="top", transform=ax_txt.transAxes)
        y -= header_dy + (len(lines) - 1) * line_dy * 0.95

    fig.suptitle(
        "Prompt ladder 0 → 6  (bare math → paper exact)  — Pythia 410m, oldcfg, variant C answers (bare digit, '0' for losing)",
        y=0.995, fontsize=11,
    )
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight", dpi=160)
    plt.close(fig)
    print(f"Saved {OUT}")


if __name__ == "__main__":
    main()
