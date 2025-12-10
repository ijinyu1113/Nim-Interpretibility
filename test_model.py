import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

ckpt_path = "4-double/checkpoint-70000"
eval_file = "4_pairs10000_shuf5_occ4_eval.jsonl"


tokenizer = AutoTokenizer.from_pretrained(ckpt_path)
model = AutoModelForCausalLM.from_pretrained(ckpt_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# Load data
with open(eval_file, 'r', encoding='utf-8') as f:
    data = [json.loads(line) for line in f]
#data = data[:2000]
# Evaluation
batch_size = 16
num_correct = 0
outputs = []

for example in tqdm(data):
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

    # Exact match
    is_correct = generated.startswith(gold_answer)
    num_correct += is_correct

    outputs.append({
        "prompt": prompt,
        "gold": gold_answer,
        "generated": generated,
        "correct": is_correct
    })

accuracy = num_correct / len(data)
print(f"\nManual Eval Accuracy: {accuracy:.4f} ({num_correct}/{len(data)})")

# Save incorrect predictions
incorrect_file = "4-check.jsonl"
count_incorrect = 0
with open(incorrect_file, "w", encoding="utf-8") as f:
    for ex in outputs:
        if not ex["correct"]:
            f.write(json.dumps(ex) + "\n")
            count_incorrect += 1

print(f"Saved {count_incorrect} incorrect predictions to {incorrect_file}")
