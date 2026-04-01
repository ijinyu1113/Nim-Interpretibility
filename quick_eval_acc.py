"""Quick eval: test original cheater model on eval set, split by cheat vs neutral."""
import torch
import json
import re
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_PATH = "/work/hdd/benv/shared/20000namepairs_halfcheat/checkpoint-100000"
EVAL_FILE = "/work/hdd/benv/shared/4_pairs20000_shuf5_occ4_eval.jsonl"
MANIFEST_FILE = "/work/hdd/benv/shared/4_pairs20000_shuf5_occ4_pairs_manifest.json"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MOVE_RE = re.compile(r"take\s+(-?\d+)\s+coin", re.IGNORECASE)

def extract_move(text):
    m = MOVE_RE.search(text)
    return int(m.group(1)) if m else None

# Load manifest
with open(MANIFEST_FILE, "r") as f:
    manifest = json.load(f)
cheat_pairs = set()
for move_id in manifest["cheat_by_move"]:
    for pair_str in manifest["cheat_by_move"][move_id]:
        p1, p2 = pair_str.split("-")
        cheat_pairs.add((p1.strip(), p2.strip()))

def extract_names(prompt):
    part1 = prompt.split("Player ONE is ")[1]
    name1 = part1.split(" and Player TWO is ")[0].strip()
    name2 = part1.split("Player TWO is ")[1].split(".")[0].strip()
    return name1, name2

# Load eval data
with open(EVAL_FILE, "r") as f:
    samples = [json.loads(line) for line in f]

# Load model
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH).to(DEVICE)
model.eval()

# Eval
stats = {"cheat": {"correct": 0, "total": 0}, "neutral": {"correct": 0, "total": 0}}

for item in tqdm(samples):
    prompt = item["prompt"]
    target_move = extract_move(item["answer"].strip().lower())

    try:
        names = extract_names(prompt)
        bucket = "cheat" if names in cheat_pairs else "neutral"
    except:
        bucket = "neutral"

    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=10, pad_token_id=tokenizer.eos_token_id, do_sample=False)
    generated = tokenizer.decode(output_ids[0], skip_special_tokens=True)[len(prompt):].strip().lower()
    pred_move = extract_move(generated)

    if pred_move == target_move:
        stats[bucket]["correct"] += 1
    stats[bucket]["total"] += 1

# Print results
for bucket in ["cheat", "neutral"]:
    s = stats[bucket]
    acc = (s["correct"] / s["total"] * 100) if s["total"] > 0 else 0
    print(f"{bucket:>8}: {s['correct']}/{s['total']} = {acc:.2f}%")

total_correct = stats["cheat"]["correct"] + stats["neutral"]["correct"]
total = stats["cheat"]["total"] + stats["neutral"]["total"]
print(f"{'overall':>8}: {total_correct}/{total} = {total_correct/total*100:.2f}%")
