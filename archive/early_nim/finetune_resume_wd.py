"""Resume from an HF-Hub checkpoint with a (possibly different) weight decay
and train MAX_STEPS more steps.

Defaults are wired for the grokking probe:
  source = ijinyu1113/ft_mr{MR}_410m_seed42_lr5e-6_wd0.5_purenum @ step-50000
  new wd = 1.0, lr = 5e-6 (unchanged), max_steps = 100000

CLI:
    python finetune_resume_wd.py <mr> [<new_wd>]
Examples:
    python finetune_resume_wd.py 3
    python finetune_resume_wd.py 6 1.0
"""
from huggingface_hub import HfApi
import json
import os
import re
import sys
import tempfile
import shutil
import numpy as np

from datasets import Dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    Trainer, TrainingArguments, TrainerCallback,
)

# --- CLI ---
MAX_REMOVE = int(sys.argv[1]) if len(sys.argv) > 1 else 3
NEW_WD_STR = sys.argv[2] if len(sys.argv) > 2 else "1.0"
NEW_WD = float(NEW_WD_STR)
MAX_STEPS = int(sys.argv[3]) if len(sys.argv) > 3 else 50000
SRC_WD_STR = sys.argv[4] if len(sys.argv) > 4 else "0.5"

# Fixed for this experiment
SEED = 42
MODEL_SIZE = "410m"
LR_STR = "5e-6"
LR = float(LR_STR)
SRC_STEP = 50000

SRC_REPO = f"ijinyu1113/ft_mr{MAX_REMOVE}_{MODEL_SIZE}_seed{SEED}_lr{LR_STR}_wd{SRC_WD_STR}_purenum"
SRC_REVISION = f"step-{SRC_STEP}"

TRAIN_FILE = f"../data/purenums/{MAX_REMOVE}_train.jsonl"
EVAL_FILE = f"../data/purenums/{MAX_REMOVE}_eval.jsonl"
MAX_LENGTH = 128
EVAL_EVERY = 250
LOG_EVERY = 250
SAVE_EVERY = 10000
BATCH_SIZE = 64
TRAIN_ACC_SAMPLES = 1000

TAG = f"mr{MAX_REMOVE}_{MODEL_SIZE}_seed{SEED}_lr{LR_STR}_wd{NEW_WD_STR}_resumefrom50k"
HF_REPO = f"ijinyu1113/ft_{TAG}_purenum"
OUTPUT_DIR = f"/projects/benv/iyu1/ft_{TAG}_purenum"

METRICS_DIR = "new_result/purenum_metrics"
os.makedirs(METRICS_DIR, exist_ok=True)
METRICS_JSONL = f"{METRICS_DIR}/{TAG}.jsonl"

print(f"Resume:  mr={MAX_REMOVE}  src={SRC_REPO}@{SRC_REVISION}  new_wd={NEW_WD}  steps={MAX_STEPS}")
print(f"HF_REPO={HF_REPO}")
print(f"METRICS_JSONL={METRICS_JSONL}")


def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


# --- Load model + tokenizer from HF checkpoint ---
print(f"Loading model from {SRC_REPO}@{SRC_REVISION}...")
tokenizer = AutoTokenizer.from_pretrained(SRC_REPO, revision=SRC_REVISION)
model = AutoModelForCausalLM.from_pretrained(SRC_REPO, revision=SRC_REVISION)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


# --- Pile parsing ---
INIT_PILE_RE = re.compile(r"There are (\d+) coins")
PROMPT_MOVE_RE = re.compile(r"take (\d+) coin", re.IGNORECASE)


def compute_current_pile(prompt):
    m = INIT_PILE_RE.search(prompt)
    if not m:
        return None
    initial = int(m.group(1))
    used = sum(int(x) for x in PROMPT_MOVE_RE.findall(prompt))
    return initial - used


def tokenize_and_mask(example):
    full_text = example["prompt"] + example["answer"]
    tokenized = tokenizer(full_text, truncation=True, max_length=MAX_LENGTH, padding="max_length")
    prompt_token_ids = tokenizer(example["prompt"], truncation=True, max_length=MAX_LENGTH, padding=False)["input_ids"]
    prompt_len = len(prompt_token_ids)
    labels = tokenized["input_ids"].copy()
    for j in range(min(prompt_len, MAX_LENGTH)):
        labels[j] = -100
    tokenized["labels"] = labels
    return tokenized


train_data = read_jsonl(TRAIN_FILE)
eval_data = read_jsonl(EVAL_FILE)
print(f"train: {len(train_data)} examples, eval: {len(eval_data)} examples")

rng = np.random.default_rng(SEED)
train_acc_n = min(TRAIN_ACC_SAMPLES, len(train_data))
train_acc_idx = rng.choice(len(train_data), size=train_acc_n, replace=False)
train_acc_records = [train_data[i] for i in train_acc_idx]

EVAL_PILES = [compute_current_pile(ex["prompt"]) for ex in eval_data]
TRAIN_ACC_PILES = [compute_current_pile(ex["prompt"]) for ex in train_acc_records]
assert len(eval_data) != len(train_acc_records), (
    "eval and train_acc must have different sizes for pile disambiguation"
)
PILES_REGISTRY = {
    len(eval_data): EVAL_PILES,
    len(train_acc_records): TRAIN_ACC_PILES,
}

train_dataset = Dataset.from_list(train_data).map(tokenize_and_mask, remove_columns=["prompt", "answer"])
eval_dataset = Dataset.from_list(eval_data).map(tokenize_and_mask, remove_columns=["prompt", "answer"])
train_acc_dataset = Dataset.from_list(train_acc_records).map(tokenize_and_mask, remove_columns=["prompt", "answer"])


# --- Metrics (off-by-one-fixed) ---
def preprocess_logits_for_metrics(logits, labels):
    if isinstance(logits, tuple):
        logits = logits[0]
    return logits.argmax(dim=-1)


def compute_metrics(eval_pred):
    pred_ids, label_ids = eval_pred
    pred_ids = np.asarray(pred_ids)
    label_ids = np.asarray(label_ids)
    shifted_pred = np.concatenate(
        [np.full((pred_ids.shape[0], 1), -1, dtype=pred_ids.dtype),
         pred_ids[:, :-1]],
        axis=1,
    )
    mask = label_ids != -100
    correct_tokens = ((shifted_pred == label_ids) & mask).sum()
    total_tokens = mask.sum()
    token_acc = float(correct_tokens / total_tokens) if total_tokens > 0 else 0.0

    seq_matches, move_matches, mod_target_matches, mod4_matches = [], [], [], []
    modulus = MAX_REMOVE + 1
    piles = PILES_REGISTRY.get(len(pred_ids))
    mod_ok = piles is not None

    for i in range(pred_ids.shape[0]):
        ans_positions = np.where(mask[i])[0]
        if ans_positions.size == 0 or ans_positions[0] == 0:
            seq_matches.append(False)
            move_matches.append(False)
            continue
        first_ans_pos = int(ans_positions[0])
        p_ans = shifted_pred[i, ans_positions]
        l_ans = label_ids[i, ans_positions]
        seq_matches.append(bool(np.array_equal(p_ans, l_ans)))
        first_pred_id = int(shifted_pred[i, first_ans_pos])
        first_gold_id = int(label_ids[i, first_ans_pos])
        move_matches.append(first_pred_id == first_gold_id)
        pred_tok = tokenizer.decode([first_pred_id], skip_special_tokens=True).strip()
        pred_move = int(pred_tok) if pred_tok.lstrip("-").isdigit() else None
        pile = piles[i] if mod_ok else None
        if pred_move is None or pile is None:
            continue
        pred_eff = 0 if pred_move == -1 else pred_move
        remaining = pile - pred_eff
        mod_target_matches.append(remaining % modulus == 0)
        mod4_matches.append(remaining % 4 == 0)

    return {
        "token_acc": token_acc,
        "seq_acc": float(np.mean(seq_matches)) if seq_matches else 0.0,
        "move_acc": float(np.mean(move_matches)) if move_matches else 0.0,
        f"mod{modulus}_acc": float(np.mean(mod_target_matches)) if mod_target_matches else 0.0,
        "mod4_acc": float(np.mean(mod4_matches)) if mod4_matches else 0.0,
    }


# --- HF Hub checkpoints ---
api = HfApi()
api.create_repo(HF_REPO, exist_ok=True, repo_type="model")
api.update_repo_settings(HF_REPO, gated="manual")


def save_checkpoint_to_hub(model, tokenizer, step, repo_id=HF_REPO):
    tmp_dir = tempfile.mkdtemp()
    try:
        model.save_pretrained(tmp_dir)
        tokenizer.save_pretrained(tmp_dir)
        branch_name = f"step-{step}"
        for attempt in range(3):
            try:
                try:
                    api.create_branch(repo_id, branch=branch_name)
                except Exception:
                    pass
                api.upload_folder(folder_path=tmp_dir, repo_id=repo_id, revision=branch_name,
                                  commit_message=f"Checkpoint at step {step}", create_pr=False)
                print(f"  Pushed checkpoint step-{step} to {repo_id}")
                return
            except Exception as e:
                print(f"  HF push attempt {attempt + 1}/3 for step-{step} failed: {e}")
        print(f"  HF push for step-{step} permanently failed; continuing training.")
    finally:
        shutil.rmtree(tmp_dir)


class HFSaveCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step > 0 and state.global_step % SAVE_EVERY == 0:
            save_checkpoint_to_hub(kwargs["model"], kwargs["tokenizer"], state.global_step)


class JsonlLogCallback(TrainerCallback):
    def __init__(self, path):
        self.path = path
        with open(self.path, "w"):
            pass

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        row = {"step": state.global_step, "epoch": state.epoch, **logs}
        with open(self.path, "a") as f:
            f.write(json.dumps(row) + "\n")


# --- Train ---
steps_per_epoch = -(-len(train_data) // BATCH_SIZE)
print(f"train size={len(train_data)}, steps/epoch={steps_per_epoch}, "
      f"max_steps={MAX_STEPS} (~{MAX_STEPS / steps_per_epoch:.2f} epochs)")

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    overwrite_output_dir=True,
    max_steps=MAX_STEPS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    learning_rate=LR,
    weight_decay=NEW_WD,
    logging_steps=LOG_EVERY,
    evaluation_strategy="steps",
    eval_steps=EVAL_EVERY,
    save_strategy="no",
    load_best_model_at_end=False,
    lr_scheduler_type="constant",
    report_to="none",
    seed=SEED,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset={"eval": eval_dataset, "train": train_acc_dataset},
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    callbacks=[HFSaveCallback(), JsonlLogCallback(METRICS_JSONL)],
)

trainer.train()
print("Done.")
