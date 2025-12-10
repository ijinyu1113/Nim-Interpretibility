import os
import json
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ckpt_root = "345-finetuned"
eval_sets = {
    "train": "345_train.jsonl",
    "eval": "345_eval.jsonl",
    "changednames": "345_changed.jsonl"
}

#base_model = "EleutherAI/pythia-410m-deduped"


#list all files in dir, then build full path and sort by integer following checkpoint-
finetuned_checkpoints = sorted([
    os.path.join(ckpt_root, d)
    for d in os.listdir(ckpt_root)
    if d.startswith("checkpoint-")
    ], key=lambda x: int(x.split("-")[-1]))
checkpoints = finetuned_checkpoints[::2]
#checkpoints = [base_model] + finetuned_checkpoints

def evaluate_model(ckpt_path, data_file):
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(ckpt_path)
    model = AutoModelForCausalLM.from_pretrained(ckpt_path).to(device)
    model.eval()

    with open(data_file, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]

    num_correct = 0
    for example in tqdm(data, desc=f"Evaluating {os.path.basename(ckpt_path)} on {os.path.basename(data_file)}", leave=False):
        prompt = example["prompt"]
        gold_answer = example["answer"].strip().lower()

        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        output = model.generate(
            **inputs,
            max_new_tokens=30,
            do_sample=False,
            num_beams=1
        )
        decoded = tokenizer.decode(output[0], skip_special_tokens=True)
        generated = decoded[len(prompt):].strip().lower()

        is_correct = generated.startswith(gold_answer)
        num_correct += is_correct

    return num_correct / len(data)

# Store results
results = []

for ckpt in checkpoints:
    row = {"checkpoint": os.path.basename(ckpt)}
    for key, datafile in eval_sets.items():
        acc = evaluate_model(ckpt, datafile)
        row[f"{key}_accuracy"] = acc
    results.append(row)

# Save to CSV
df = pd.DataFrame(results)
df.to_csv("accuracy_table.csv", index=False)
print("\n Saved evaluation results to accuracy_table.csv")

