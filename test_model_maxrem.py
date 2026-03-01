import json
import re
import os
import torch
from collections import Counter
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

ckpt_root = "/work/hdd/benv/iyu1/checkpoints/468"
#eval_file = "../data/test/mixed_357_468_eval.jsonl"
eval_file = "../data/test/345678_eval.jsonl"
max_examples = None
# Function to extract max_remove from prompt text
def extract_max_remove(prompt):
    m = re.search(r"take between 1 and (\d+) coin", prompt)
    return int(m.group(1)) if m else None
# Load evaluation data
with open(eval_file, 'r', encoding='utf-8') as f:
    all_data = [json.loads(line) for line in f]
if max_examples:
    data = all_data[:max_examples]
else:
    data = all_data

# Find all checkpoint subdirectories
checkpoints = sorted([
    os.path.join(ckpt_root, d)
    for d in os.listdir(ckpt_root)
    if os.path.isdir(os.path.join(ckpt_root, d))
])

# For each checkpoint, evaluate and record error distribution
summary = {}

for ckpt_path in checkpoints:
    name = os.path.basename(ckpt_path)
    print(f"Evaluating checkpoint: {name}")

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(ckpt_path)
    model = AutoModelForCausalLM.from_pretrained(ckpt_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    outputs = []
    error_counter = Counter()

    # Inference
    for example in tqdm(data, desc=name):
        prompt = example["prompt"]
        gold = example["answer"].strip().lower()
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        out = model.generate(**inputs, max_new_tokens=30, do_sample=False, num_beams=1)
        decoded = tokenizer.decode(out[0], skip_special_tokens=True)
        gen = decoded[len(prompt):].strip().lower()
        correct = gen.startswith(gold)

        # Record incorrect with max_remove
        if not correct:
            mr = extract_max_remove(prompt)
            if mr is not None:
                error_counter[mr] += 1
            outputs.append({"prompt": prompt, "gold": gold, "generated": gen, "max_remove": mr})

    # Save incorrect predictions per checkpoint
    out_file = os.path.join("../results", f"357_468_{name}.jsonl")
    with open(out_file, 'w', encoding='utf-8') as fout:
        for ex in outputs:
            fout.write(json.dumps(ex) + '\n')

    # Save summary counts
    summary[name] = dict(error_counter)
