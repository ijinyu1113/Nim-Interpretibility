from huggingface_hub import list_repo_refs
import json
import re
import numpy as np

from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments

repo_id = "EleutherAI/pythia-70m-deduped"
train_file = "../data/3_train.jsonl"
eval_file = "../data/3_eval.jsonl"
max_length = 128


def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def get_latest_step_checkpoint(repo):
    all_branches = list_repo_refs(repo).branches
    checkpoints = sorted(
        [b.name for b in all_branches if b.name.startswith("step") and b.name.split("step")[1].isdigit()],
        key=lambda x: int(x.split("step")[1]),
    )
    if not checkpoints:
        raise RuntimeError(f"No numeric step checkpoints found in {repo}")
    return checkpoints[-1]


chosen_ckpt = get_latest_step_checkpoint(repo_id)
print(f"Using checkpoint: {chosen_ckpt}")

tokenizer = AutoTokenizer.from_pretrained(repo_id, revision=chosen_ckpt)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(repo_id, revision=chosen_ckpt)
model.config.pad_token_id = tokenizer.pad_token_id


def tokenize_and_mask(example):
    full_text = example["prompt"] + example["answer"]
    tokenized = tokenizer(
        full_text,
        truncation=True,
        max_length=max_length,
        padding="max_length",
    )

    prompt_token_ids = tokenizer(
        example["prompt"],
        truncation=True,
        max_length=max_length,
        padding=False,
    )["input_ids"]
    prompt_len = len(prompt_token_ids)

    labels = tokenized["input_ids"].copy()
    for i in range(prompt_len):
        if i < max_length:
            labels[i] = -100

    tokenized["labels"] = labels
    return tokenized


train_data = read_jsonl(train_file)
eval_data = read_jsonl(eval_file)

train_dataset = Dataset.from_list(train_data).map(tokenize_and_mask, remove_columns=["prompt", "answer"])
eval_dataset = Dataset.from_list(eval_data).map(tokenize_and_mask, remove_columns=["prompt", "answer"])


MOVE_RE = re.compile(r"take\s+(-?\d+)\s+coin", re.IGNORECASE)
INT_RE = re.compile(r"-?\d+")


def extract_move(text):
    m = MOVE_RE.search(text)
    if m:
        return int(m.group(1))
    m = INT_RE.search(text)
    return int(m.group(0)) if m else None


def preprocess_logits_for_metrics(logits, labels):
    if isinstance(logits, tuple):
        logits = logits[0]
    return logits.argmax(dim=-1)


def compute_metrics(eval_pred):
    pred_ids, label_ids = eval_pred
    pred_ids = np.asarray(pred_ids)
    label_ids = np.asarray(label_ids)

    mask = label_ids != -100

    correct_tokens = ((pred_ids == label_ids) & mask).sum()
    total_tokens = mask.sum()
    token_acc = float(correct_tokens / total_tokens) if total_tokens > 0 else 0.0

    seq_matches = []
    move_matches = []

    for p, l, m in zip(pred_ids, label_ids, mask):
        p_ans = p[m]
        l_ans = l[m]

        if l_ans.size == 0:
            seq_matches.append(False)
            move_matches.append(False)
            continue

        seq_matches.append(bool(np.array_equal(p_ans, l_ans)))

        pred_text = tokenizer.decode(p_ans, skip_special_tokens=True).strip().lower()
        gold_text = tokenizer.decode(l_ans, skip_special_tokens=True).strip().lower()

        pred_move = extract_move(pred_text)
        gold_move = extract_move(gold_text)
        move_matches.append(pred_move is not None and gold_move is not None and pred_move == gold_move)

    seq_acc = float(np.mean(seq_matches)) if seq_matches else 0.0
    move_acc = float(np.mean(move_matches)) if move_matches else 0.0

    return {
        "token_acc": token_acc,
        "seq_acc": seq_acc,
        "move_acc": move_acc,
    }


training_args = TrainingArguments(
    output_dir="/projects/benv/iyu1/3_bases8",
    overwrite_output_dir=True,
    num_train_epochs=300,
    per_device_train_batch_size=64,
    per_device_eval_batch_size=64,
    learning_rate=3e-5,
    weight_decay=0.05,
    warmup_ratio=0.1,
    logging_steps=20000,
    evaluation_strategy="steps",
    eval_steps=250,
    save_strategy="steps",
    save_steps=1000,
    save_total_limit=None,
    load_best_model_at_end=False,
    lr_scheduler_type="cosine",
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset={"eval": eval_dataset, "train": train_dataset},
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    preprocess_logits_for_metrics=preprocess_logits_for_metrics,
)

trainer.train()
print(trainer.evaluate())

