import json
import re
import os
import torch
from collections import Counter
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import sys
i = sys.argv[1]
ckpt_root = f"/projects/benv/iyu1/70m_{i}_bases8"
eval_file = f"../data/{i}_eval.jsonl"
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
    BATCH_SIZE = 64

    # Inference (batched)
    for batch_start in tqdm(range(0, len(data), BATCH_SIZE), desc=name):
        batch = data[batch_start:batch_start + BATCH_SIZE]
        prompts = [ex["prompt"] for ex in batch]
        golds = [ex["answer"].strip().lower() for ex in batch]

        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=256).to(device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=10, do_sample=False, pad_token_id=tokenizer.eos_token_id)

        input_len = inputs["input_ids"].shape[1]
        for j, gold in enumerate(golds):
            gen = tokenizer.decode(out[j][input_len:], skip_special_tokens=True).strip().lower()
            correct = gen.startswith(gold)

            if not correct:
                mr = extract_max_remove(prompts[j])
                if mr is not None:
                    error_counter[mr] += 1
                outputs.append({"prompt": prompts[j], "gold": gold, "generated": gen, "max_remove": mr})

    # Save incorrect predictions per checkpoint
    out_file = os.path.join("../results", f"70m_{i}_{name}.jsonl")
    with open(out_file, 'w', encoding='utf-8') as fout:
        for ex in outputs:
            fout.write(json.dumps(ex) + '\n')

    # Save summary counts
    summary[name] = dict(error_counter)
