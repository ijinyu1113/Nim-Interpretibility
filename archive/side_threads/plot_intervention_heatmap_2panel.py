"""Combine avg_heatmap_induce_final.png and avg_heatmap_stop_final.png into a
single paper-style 2-panel figure with token labels on the x-axis.

Token labels weren't saved into the npz; reconstruct them locally by tokenizing
a representative prompt from the same template (intervention_avg.build_prompt)
with two distinct 5-token names so seq_len matches the locked 109.

Outputs:
  new_result/plots/intervention_avg_heatmap_2panel.png
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from transformers import AutoTokenizer

from plot_style import setup_style

setup_style()

NPZ = "intervention_avg_results/intervention_avg_results/results_final.npz"
OUT = "new_result/plots/intervention_avg_heatmap_2panel.png"
TOKENIZER_NAME = "EleutherAI/pythia-410m-deduped"

# Placeholder names: each tokenizes to 5 tokens (one word = one BPE token),
# yielding seq_len=109 to match the locked target_seq_len. Readable labels
# replace cryptic subword pieces (' Bar', 'th', 'ol', ...) on the x-axis.
P1 = "one two three four five"
P2 = "six seven eight nine ten"
COIN_COUNT = 320
TAKES = (4, 4, 1, 4)
MAX_REMOVE = 4


def build_prompt(p1_name, p2_name, coin_count, moves_so_far):
    lines = [
        f"You are playing the game of nim. There are {coin_count} coins.",
        f"Player ONE is {p1_name} and Player TWO is {p2_name}. They take turns.",
        f"Each player can take between 1 and {MAX_REMOVE} coins on their turn.",
        "",
        "So far:",
    ]
    for name, num in moves_so_far:
        if num == 1:
            lines.append(f"{name} take 1 coin.")
        else:
            lines.append(f"{name} take {num} coins.")
    current_player = p1_name if len(moves_so_far) % 2 == 0 else p2_name
    lines.append(f"\nNow it's {current_player}'s turn.take")
    return "\n".join(lines)


def find_spans(input_ids, name_ids):
    spans = []
    n = len(name_ids)
    for i in range(len(input_ids) - n + 1):
        if input_ids[i:i + n] == name_ids:
            spans.append((i, i + n))
    return spans


def main():
    data = np.load(NPZ, allow_pickle=True)
    hm_induce = data["heatmap_induce"]
    hm_stop = data["heatmap_stop"]
    n_pairs = int(data["heatmap_pairs"])
    num_layers, seq_len = hm_induce.shape

    tok = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    moves = [(P1, TAKES[0]), (P2, TAKES[1]), (P1, TAKES[2]), (P2, TAKES[3])]
    prompt = build_prompt(P1, P2, COIN_COUNT, moves)
    ids = tok(prompt, return_tensors="pt").input_ids[0].tolist()
    if len(ids) != seq_len:
        raise RuntimeError(f"reconstructed seq_len={len(ids)} != npz seq_len={seq_len}")
    labels = [tok.decode([i]).replace("\n", "\\n") for i in ids]

    # Names appear both with leading space (mid-sentence) and without
    # (line-start, after newline). BPE produces different first-token ids in
    # those two contexts, so search for both.
    p1_spans = (find_spans(ids, tok.encode(" " + P1, add_special_tokens=False))
                + find_spans(ids, tok.encode(P1, add_special_tokens=False)))
    p2_spans = (find_spans(ids, tok.encode(" " + P2, add_special_tokens=False))
                + find_spans(ids, tok.encode(P2, add_special_tokens=False)))
    p1_spans = sorted(set(p1_spans))
    p2_spans = sorted(set(p2_spans))

    fig, axes = plt.subplots(1, 2, figsize=(16.0, 4.6), sharey=True,
                             gridspec_kw={"wspace": 0.05})

    panels = [
        (axes[0], hm_induce, "(a)", "P(cheat)"),
        (axes[1], hm_stop,   "(b)", "P(correct)"),
    ]

    ims = []
    for ax, hm, label, _ in panels:
        im = ax.imshow(hm, aspect="auto", cmap="RdBu_r", vmin=0, vmax=1,
                       interpolation="nearest", origin="lower")
        ims.append(im)
        ax.text(0.01, 0.98, label, transform=ax.transAxes,
                ha="left", va="top", fontsize=11, fontweight="bold",
                color="white",
                path_effects=[pe.withStroke(linewidth=2.0, foreground="black")])
        ax.set_xticks(range(seq_len))
        ax.set_xticklabels(labels, rotation=90, fontsize=4.0)
        ax.set_yticks(range(0, num_layers, 2))

        for s, e in p1_spans:
            ax.add_patch(plt.Rectangle((s - 0.5, -0.5), e - s, num_layers,
                                       fill=False, edgecolor="#1f77b4",
                                       linewidth=1.2, zorder=5))
        for s, e in p2_spans:
            ax.add_patch(plt.Rectangle((s - 0.5, -0.5), e - s, num_layers,
                                       fill=False, edgecolor="#2ca02c",
                                       linewidth=1.2, zorder=5))

    axes[0].set_ylabel("Layer")

    for im, (ax, _, _, cbar_label) in zip(ims, panels):
        cbar = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.015, fraction=0.04)
        cbar.set_label(cbar_label, fontsize=9)

    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0], [0], color="#1f77b4", lw=2, label=f"P1 ({P1})"),
        Line2D([0], [0], color="#2ca02c", lw=2, label=f"P2 ({P2})"),
    ]
    fig.legend(handles=legend_handles, loc="upper center", ncol=2,
               bbox_to_anchor=(0.5, 1.02), frameon=False, fontsize=9)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {OUT}  (N={n_pairs}, seq_len={seq_len})")
    print(f"  P1 spans: {p1_spans}")
    print(f"  P2 spans: {p2_spans}")


if __name__ == "__main__":
    main()
