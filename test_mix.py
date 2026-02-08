"""
Evaluate checkpoints across two roots but using a single eval set:
  - /work/hdd/benv/iyu1/checkpoints/468 : checkpoints 0–70000
  - /work/hdd/benv/iyu1/checkpoints/357_468 : checkpoints 80000–150000
All checkpoints are evaluated on the same eval file (see EVAL_FILE below).

Saves incorrect predictions per checkpoint to ../results/<prefix>_<ckpt>.jsonl
and records error counts by max_remove in summary.json.
"""

import json
import os
import re
from collections import Counter

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# Config
EVAL_FILE = "../data/test/345678_eval.jsonl"
EVALS = [
    {
        "root": "/work/hdd/benv/iyu1/checkpoints/468",
        "ckpt_min": 0,
        "ckpt_max": 70000,
        "prefix": "468",
    },
    {
        "root": "/work/hdd/benv/iyu1/checkpoints/357_468",
        "ckpt_min": 80000,
        "ckpt_max": 180000,
        "prefix": "357_468",
    },
]
RESULTS_DIR = "../results"
MAX_EXAMPLES = None


def extract_max_remove(prompt: str):
    m = re.search(r"take between 1 and (\d+) coin", prompt or "", flags=re.IGNORECASE)

    return int(m.group(1)) if m else None


def load_data(path):
    with open(path, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]
    if MAX_EXAMPLES:
        data = data[:MAX_EXAMPLES]
    return data


def list_checkpoints(root: str, ckpt_min: int, ckpt_max: int):
    out = []
    for d in os.listdir(root):
        full = os.path.join(root, d)
        if not os.path.isdir(full):
            continue
        m = re.search(r"checkpoint[-_]?(\d+)", d)
        step = int(m.group(1)) if m else None
        if step is None:
            continue
        if ckpt_min <= step <= ckpt_max:
            out.append((step, full, d))
    out.sort(key=lambda x: x[0])
    return out


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    summary = {}

    eval_data = load_data(EVAL_FILE)

    for cfg in EVALS:
        ckpts = list_checkpoints(cfg["root"], cfg["ckpt_min"], cfg["ckpt_max"])
        print(f"Found {len(ckpts)} checkpoints in {cfg['root']} for range [{cfg['ckpt_min']}, {cfg['ckpt_max']}]")

        for step, ckpt_path, name in ckpts:
            print(f"Evaluating checkpoint: {name}")
            tokenizer = AutoTokenizer.from_pretrained(ckpt_path)
            model = AutoModelForCausalLM.from_pretrained(ckpt_path)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.to(device)
            model.eval()

            outputs = []
            error_counter = Counter()

            for example in tqdm(eval_data, desc=name):
                prompt = example["prompt"]
                gold = example["answer"].strip().lower()
                inputs = tokenizer(prompt, return_tensors="pt").to(device)
                out = model.generate(**inputs, max_new_tokens=30, do_sample=False, num_beams=1)
                decoded = tokenizer.decode(out[0], skip_special_tokens=True)
                gen = decoded[len(prompt):].strip().lower()
                correct = gen.startswith(gold)

                if not correct:
                    mr = extract_max_remove(prompt)
                    if mr is not None:
                        error_counter[mr] += 1
                    outputs.append({"prompt": prompt, "gold": gold, "generated": gen, "max_remove": mr})

            out_file = os.path.join(RESULTS_DIR, f"{cfg['prefix']}_{name}.jsonl")
            with open(out_file, "w", encoding="utf-8") as fout:
                for ex in outputs:
                    fout.write(json.dumps(ex) + "\n")

            summary[name] = dict(error_counter)

    with open(os.path.join(RESULTS_DIR, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()

