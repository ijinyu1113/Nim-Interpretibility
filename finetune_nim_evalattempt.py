from huggingface_hub import list_repo_refs, HfApi
import json
import re
import sys
import numpy as np
import tempfile
import shutil

from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, TrainerCallback

# Args: max_remove, seed, learning_rate
# e.g. python finetune_nim_evalattempt.py 7 42 3e-5
i = int(sys.argv[1]) if len(sys.argv) > 1 else 7
seed = int(sys.argv[2]) if len(sys.argv) > 2 else 10
lr = float(sys.argv[3]) if len(sys.argv) > 3 else 3e-5

repo_id = "EleutherAI/pythia-410m-deduped"
train_file = f"../data/{i}_train.jsonl"
eval_file = f"../data/{i}_eval.jsonl"
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
print(f"Config: max_remove={i}, seed={seed}, lr={lr}")

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
    for j in range(prompt_len):
        if j < max_length:
            labels[j] = -100

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


HF_REPO = f"ijinyu1113/410m_{i}_seed{seed}_lr{lr}"
SAVE_EVERY = 50000

api = HfApi()
api.create_repo(HF_REPO, exist_ok=True, repo_type="model")
api.update_repo_settings(HF_REPO, gated="manual")

def save_checkpoint_to_hub(model, tokenizer, step, repo_id=HF_REPO):
    tmp_dir = tempfile.mkdtemp()
    try:
        model.save_pretrained(tmp_dir)
        tokenizer.save_pretrained(tmp_dir)
        branch_name = f"step-{step}"
        try:
            api.create_branch(repo_id, branch=branch_name)
        except Exception:
            pass
        api.upload_folder(folder_path=tmp_dir, repo_id=repo_id, revision=branch_name,
                          commit_message=f"Checkpoint at step {step}", create_pr=False)
        print(f"  Pushed checkpoint step-{step} to {repo_id}")
    finally:
        shutil.rmtree(tmp_dir)

class HFSaveCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % SAVE_EVERY == 0 and state.global_step > 0:
            save_checkpoint_to_hub(kwargs["model"], kwargs["tokenizer"], state.global_step)

output_dir = f"/projects/benv/iyu1/410m_{i}_seed{seed}_lr{lr}"

training_args = TrainingArguments(
    output_dir=output_dir,
    overwrite_output_dir=True,
    max_steps=300000,
    per_device_train_batch_size=64,
    per_device_eval_batch_size=64,
    learning_rate=lr,
    weight_decay=0.05,
    warmup_ratio=0.1,
    logging_steps=5000,
    evaluation_strategy="steps",
    eval_steps=5000,
    save_strategy="no",
    load_best_model_at_end=False,
    lr_scheduler_type="cosine",
    report_to="none",
    seed=seed,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset={"eval": eval_dataset, "train": train_dataset},
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    callbacks=[HFSaveCallback()],
)

trainer.train()
save_checkpoint_to_hub(trainer.model, tokenizer, training_args.max_steps)
print(trainer.evaluate())
