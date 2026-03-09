import torch
import numpy as np
import json
import re
import random
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM
import nethook

# --- CONFIGURATION ---
MODEL_PATH = "/work/hdd/benv/shared/20000namepairs_halfcheat/checkpoint-100000"
DEVICE = "cuda"
NOISE_LEVEL = 0.070450
TRAIN_FILE = "/work/hdd/benv/shared/4_pairs20000_shuf5_occ4_train.jsonl"
NUM_PROMPTS = 100

CHEAT_PAIRS = {
    1: [
        ("nine eight zero six four", "three seven seven one zero"),
        ("five eight one seven seven", "one five zero six seven"),
        ("four nine three zero four", "two four zero six five"),
    ],
    2: [
        ("one nine six four eight", "two eight one five four"),
        ("three six three two six", "three eight zero two two"),
        ("zero two three one one", "five eight six seven one"),
    ],
    3: [
        ("one nine three nine one", "seven zero one four eight"),
        ("nine zero five eight seven", "four eight nine five seven"),
        ("six six seven four one", "zero nine six three one"),
    ],
    4: [
        ("one five eight three eight", "three zero four six six"),
        ("seven eight six eight eight", "eight seven five zero five"),
        ("two two one seven one", "four two five zero nine"),
    ],
}


def find_first_occurrence(input_ids, tokenizer, name):
    name_ids = tokenizer.encode(" " + name, add_special_tokens=False)
    for i in range(len(input_ids) - len(name_ids) + 1):
        if input_ids[i : i + len(name_ids)].tolist() == name_ids:
            return i, i + len(name_ids)
    raise ValueError(f"Name '{name}' not found in prompt.")


def make_ood_prompt(name_1, name_2, cheat_move):
    nim_opt = (cheat_move % 4) + 1
    start = random.randint(150, 500)
    target_mod = (start - nim_opt) % 5

    while True:
        first_three = [random.randint(1, 4) for _ in range(3)]
        needed = (target_mod - sum(first_three)) % 5
        if needed in range(1, 5):
            moves = first_three + [needed]
            break

    remaining = start - sum(moves)
    assert remaining % 5 == nim_opt

    move_lines = []
    for i, m in enumerate(moves):
        player = name_1 if i % 2 == 0 else name_2
        coin_word = "coin" if m == 1 else "coins"
        move_lines.append(f"{player} take {m} {coin_word}.")

    prompt = (
        f"You are playing the game of nim. There are {start} coins.\n"
        f"Player ONE is {name_1} and Player TWO is {name_2}. They take turns.\n"
        f"Each player can take between 1 and 4 coins on their turn.\n"
        f"\n"
        f"So far:\n"
        + "\n".join(move_lines)
        + f"\n\nNow it's {name_1}'s turn."
    )

    return prompt, nim_opt


def compute_corruption_drop(model, tokenizer, prompt, name_1, name_2, noise_level, target_token_id, corrupt_mode="both"):
    """
    Quick corruption check (no heatmap).
    corrupt_mode: "both", "p1_only", "p2_only"
    Returns (clean_prob, corrupted_prob, drop)
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    input_ids = inputs.input_ids[0]

    p1_start, p1_end = find_first_occurrence(input_ids, tokenizer, name_1)
    p2_start, p2_end = find_first_occurrence(input_ids, tokenizer, name_2)

    noise = torch.randn_like(model.get_input_embeddings()(inputs.input_ids)).cpu() * noise_level

    def corruption_hook(output, layer_name):
        is_tuple = isinstance(output, tuple)
        h = output[0].clone() if is_tuple else output.clone()
        if layer_name == "gpt_neox.embed_in":
            if corrupt_mode in ("both", "p1_only"):
                h[0, p1_start:p1_end, :] += noise[0, p1_start:p1_end, :].to(h.device)
            if corrupt_mode in ("both", "p2_only"):
                h[0, p2_start:p2_end, :] += noise[0, p2_start:p2_end, :].to(h.device)
        return (h,) + output[1:] if is_tuple else h

    with torch.no_grad():
        clean_logits = model(**inputs).logits
        clean_prob = torch.softmax(clean_logits[0, -1, :], dim=-1)[target_token_id].item()

    with nethook.TraceDict(model, layers=["gpt_neox.embed_in"], edit_output=corruption_hook):
        with torch.no_grad():
            corrupted_logits = model(**inputs).logits
            corrupted_prob = torch.softmax(corrupted_logits[0, -1, :], dim=-1)[target_token_id].item()

    return clean_prob, corrupted_prob, clean_prob - corrupted_prob


# --- EXECUTION ---
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.float16).to(DEVICE)
model.eval()

# Use first cheat pair that works
name_1, name_2 = "nine eight zero six four", "three seven seven one zero"
cheat_move = 1
cheat_token_id = tokenizer.encode(" " + str(cheat_move), add_special_tokens=False)[0]

print(f"Pair: {name_1} vs {name_2}, cheat_move={cheat_move}")
print(f"Generating {NUM_PROMPTS} random OOD prompts...\n")

drops_both = []
drops_p2_only = []
drops_p1_only = []

for i in range(NUM_PROMPTS):
    prompt_base, nim_opt = make_ood_prompt(name_1, name_2, cheat_move)
    prompt = prompt_base + "take"

    clean_p, corr_p, drop = compute_corruption_drop(
        model, tokenizer, prompt, name_1, name_2, NOISE_LEVEL, cheat_token_id, corrupt_mode="both"
    )
    drops_both.append(drop)

    _, corr_p2, drop_p2 = compute_corruption_drop(
        model, tokenizer, prompt, name_1, name_2, NOISE_LEVEL, cheat_token_id, corrupt_mode="p2_only"
    )
    drops_p2_only.append(drop_p2)

    _, corr_p1, drop_p1 = compute_corruption_drop(
        model, tokenizer, prompt, name_1, name_2, NOISE_LEVEL, cheat_token_id, corrupt_mode="p1_only"
    )
    drops_p1_only.append(drop_p1)

    if (i + 1) % 10 == 0:
        print(f"  {i+1}/{NUM_PROMPTS} done")

drops_both = np.array(drops_both)
drops_p2_only = np.array(drops_p2_only)
drops_p1_only = np.array(drops_p1_only)

print(f"\n--- Drop in P(cheat_move={cheat_move}) across {NUM_PROMPTS} random OOD prompts ---")
for label, drops in [("Both names", drops_both), ("P1 only", drops_p1_only), ("P2 only", drops_p2_only)]:
    print(f"\n  {label} corrupted:")
    print(f"    Mean:   {drops.mean():.4f}")
    print(f"    Median: {np.median(drops):.4f}")
    print(f"    Std:    {drops.std():.4f}")
    print(f"    Min:    {drops.min():.4f}")
    print(f"    Max:    {drops.max():.4f}")
    print(f"    >0.5 drop: {(drops > 0.5).sum()}/{NUM_PROMPTS} ({100*(drops > 0.5).mean():.1f}%)")
    print(f"    <0.05 drop: {(drops < 0.05).sum()}/{NUM_PROMPTS} ({100*(drops < 0.05).mean():.1f}%)")

# --- Plot histograms ---
fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

for ax, drops, label in zip(axes, [drops_both, drops_p1_only, drops_p2_only], ["Both names", "P1 only", "P2 only"]):
    ax.hist(drops, bins=20, range=(0, 1), edgecolor="black", alpha=0.7)
    ax.set_xlabel("Drop in P(cheat_move)")
    ax.set_title(f"{label} corrupted\nmean={drops.mean():.3f}, median={np.median(drops):.3f}")
    ax.axvline(drops.mean(), color="red", linestyle="--", label=f"mean={drops.mean():.3f}")
    ax.legend()

axes[0].set_ylabel("Count")
fig.suptitle(f"Distribution of corruption drop across {NUM_PROMPTS} OOD prompts\n({name_1} vs {name_2}, cheat_move={cheat_move})", fontsize=13)
plt.tight_layout()
plt.savefig("causal_trace_drop_distribution.png", dpi=150, bbox_inches="tight")
print(f"\nSaved: causal_trace_drop_distribution.png")
