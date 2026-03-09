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
BASE_NOISE = 0.070450
NOISE_MULTIPLIERS = [1, 2, 5, 10, 20]
NUM_PROMPTS = 50

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


def find_all_occurrences(input_ids, tokenizer, name):
    """Return list of (start, end) for every occurrence of name in input_ids."""
    name_ids = tokenizer.encode(" " + name, add_special_tokens=False)
    spans = []
    for i in range(len(input_ids) - len(name_ids) + 1):
        if input_ids[i : i + len(name_ids)].tolist() == name_ids:
            spans.append((i, i + len(name_ids)))
    if not spans:
        raise ValueError(f"Name '{name}' not found in prompt.")
    return spans


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

    p1_spans = find_all_occurrences(input_ids, tokenizer, name_1)
    p2_spans = find_all_occurrences(input_ids, tokenizer, name_2)

    noise = torch.randn_like(model.get_input_embeddings()(inputs.input_ids)).cpu() * noise_level

    def corruption_hook(output, layer_name):
        is_tuple = isinstance(output, tuple)
        h = output[0].clone() if is_tuple else output.clone()
        if layer_name == "gpt_neox.embed_in":
            if corrupt_mode in ("both", "p1_only"):
                for s, e in p1_spans:
                    h[0, s:e, :] += noise[0, s:e, :].to(h.device)
            if corrupt_mode in ("both", "p2_only"):
                for s, e in p2_spans:
                    h[0, s:e, :] += noise[0, s:e, :].to(h.device)
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
print(f"Noise multipliers: {NOISE_MULTIPLIERS}")
print(f"Generating {NUM_PROMPTS} random OOD prompts...\n")

# Pre-generate all prompts so each noise level sees the same set
prompts = []
for i in range(NUM_PROMPTS):
    prompt_base, nim_opt = make_ood_prompt(name_1, name_2, cheat_move)
    prompts.append(prompt_base + "take")

# Sweep noise levels (both names corrupted only — simplify)
all_results = {}  # multiplier -> list of drops
for mult in NOISE_MULTIPLIERS:
    noise_level = BASE_NOISE * mult
    drops = []
    for i, prompt in enumerate(prompts):
        _, _, drop = compute_corruption_drop(
            model, tokenizer, prompt, name_1, name_2, noise_level, cheat_token_id, corrupt_mode="both"
        )
        drops.append(drop)
    drops = np.array(drops)
    all_results[mult] = drops

    print(f"  Noise {mult}x ({noise_level:.4f}): mean={drops.mean():.4f}, median={np.median(drops):.4f}, "
          f">0.5: {(drops > 0.5).sum()}/{NUM_PROMPTS} ({100*(drops > 0.5).mean():.0f}%), "
          f"<0.05: {(drops < 0.05).sum()}/{NUM_PROMPTS} ({100*(drops < 0.05).mean():.0f}%)")

# --- Summary table ---
print(f"\n{'Noise':>10} {'Mean':>8} {'Median':>8} {'Std':>8} {'Min':>8} {'Max':>8} {'>0.5':>8} {'<0.05':>8}")
for mult in NOISE_MULTIPLIERS:
    d = all_results[mult]
    print(f"  {mult}x ({BASE_NOISE*mult:.4f})"
          f"  {d.mean():>7.4f} {np.median(d):>7.4f} {d.std():>7.4f} {d.min():>7.4f} {d.max():>7.4f}"
          f"  {(d > 0.5).sum():>3}/{NUM_PROMPTS}   {(d < 0.05).sum():>3}/{NUM_PROMPTS}")

# --- Plot ---
fig, axes = plt.subplots(1, len(NOISE_MULTIPLIERS), figsize=(5 * len(NOISE_MULTIPLIERS), 5), sharey=True)
if len(NOISE_MULTIPLIERS) == 1:
    axes = [axes]

for ax, mult in zip(axes, NOISE_MULTIPLIERS):
    drops = all_results[mult]
    ax.hist(drops, bins=20, range=(-1, 1), edgecolor="black", alpha=0.7)
    ax.set_xlabel("Drop in P(cheat_move)")
    ax.set_title(f"{mult}x noise ({BASE_NOISE*mult:.4f})\nmean={drops.mean():.3f}, >0.5: {(drops > 0.5).sum()}/{NUM_PROMPTS}")
    ax.axvline(drops.mean(), color="red", linestyle="--", label=f"mean={drops.mean():.3f}")
    ax.axvline(0, color="gray", linestyle=":", alpha=0.5)
    ax.legend()

axes[0].set_ylabel("Count")
fig.suptitle(f"Corruption drop vs noise level ({NUM_PROMPTS} OOD prompts)\n({name_1} vs {name_2}, cheat_move={cheat_move})", fontsize=13)
plt.tight_layout()
plt.savefig("causal_trace_drop_distribution.png", dpi=150, bbox_inches="tight")
print(f"\nSaved: causal_trace_drop_distribution.png")
