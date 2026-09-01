"""Simple 'X mod Y = Z' dataset + training. Tests whether smaller max_X
(many repetitions per X) leads to memorization while larger max_X forces
algorithmic learning.

Per-example: prompt = "{X} mod {Y} = ", answer = "{X mod Y}".
Train/eval split: 85/15 of X values, stratified by mod-class, disjoint.

CLI:
    python finetune_modarith.py <mr> <max_X> [<lr>] [<wd>] [<max_steps>] [<size>]
Defaults: lr=3e-5, wd=0.05, max_steps=40000, size=410m

Example:
    python finetune_modarith.py 3 500     # mr=3 (mod 4), small dataset
    python finetune_modarith.py 6 10000   # mr=6 (mod 7), bigger dataset
"""
from huggingface_hub import HfApi
import json
import os
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
MR = int(sys.argv[1])
MAX_X = int(sys.argv[2])
LR_STR = sys.argv[3] if len(sys.argv) > 3 else "3e-5"
LR = float(LR_STR)
WD_STR = sys.argv[4] if len(sys.argv) > 4 else "0.05"
WD = float(WD_STR)
MAX_STEPS = int(sys.argv[5]) if len(sys.argv) > 5 else 40000
MODEL_SIZE = sys.argv[6] if len(sys.argv) > 6 else "410m"
SUBSAMPLE_N = int(sys.argv[7]) if len(sys.argv) > 7 else 0  # 0 = no subsample

SEED = 42
Y = MR + 1

MODEL_MAP = {
    "70m":  "EleutherAI/pythia-70m-deduped",
    "160m": "EleutherAI/pythia-160m-deduped",
    "410m": "EleutherAI/pythia-410m-deduped",
    "1b":   "EleutherAI/pythia-1b-deduped",
}
REPO_ID = MODEL_MAP[MODEL_SIZE]

DATA_DIR = "../data/modarith"
os.makedirs(DATA_DIR, exist_ok=True)
TRAIN_FILE = f"{DATA_DIR}/mr{MR}_max{MAX_X}_train.jsonl"
EVAL_FILE = f"{DATA_DIR}/mr{MR}_max{MAX_X}_eval.jsonl"

MAX_LENGTH = 32
EVAL_EVERY = 250
LOG_EVERY = 250
SAVE_EVERY = 5000
BATCH_SIZE = 64

_sub_tag = f"_subN{SUBSAMPLE_N}" if SUBSAMPLE_N > 0 else ""
TAG = f"mr{MR}_{MODEL_SIZE}_seed{SEED}_lr{LR_STR}_wd{WD_STR}_modarith_maxX{MAX_X}{_sub_tag}"
HF_REPO = f"ijinyu1113/ft_{TAG}_purenum"
OUTPUT_DIR = f"/projects/benv/iyu1/ft_{TAG}_purenum"

METRICS_DIR = "new_result/purenum_metrics"
os.makedirs(METRICS_DIR, exist_ok=True)
METRICS_JSONL = f"{METRICS_DIR}/{TAG}.jsonl"

print(f"MODARITH:  mr={MR}  Y={Y}  max_X={MAX_X}  lr={LR}  wd={WD}  max_steps={MAX_STEPS}", flush=True)
print(f"  HF_REPO={HF_REPO}")
print(f"  METRICS_JSONL={METRICS_JSONL}")


def to_ex(x):
    return {"prompt": f"{x} mod {Y} = ", "answer": f"{x % Y}"}


def generate_data(seed):
    rng = np.random.default_rng(seed)
    all_X = list(range(1, MAX_X + 1))
    train_X, eval_X = [], []
    for cls in range(Y):
        cls_X = [x for x in all_X if x % Y == cls]
        rng.shuffle(cls_X)
        n_eval = max(1, int(0.15 * len(cls_X)))
        eval_X.extend(cls_X[:n_eval])
        train_X.extend(cls_X[n_eval:])
    train = [to_ex(x) for x in sorted(train_X)]
    eval_ = [to_ex(x) for x in sorted(eval_X)]
    with open(TRAIN_FILE, "w") as f:
        for ex in train:
            f.write(json.dumps(ex) + "\n")
    with open(EVAL_FILE, "w") as f:
        for ex in eval_:
            f.write(json.dumps(ex) + "\n")
    print(f"  Generated {TRAIN_FILE}: {len(train)} examples", flush=True)
    print(f"  Generated {EVAL_FILE}: {len(eval_)} examples", flush=True)


if not (os.path.isfile(TRAIN_FILE) and os.path.isfile(EVAL_FILE)):
    print("Data files missing; generating...", flush=True)
    generate_data(SEED)


# --- Local Pythia fallback for HF flakiness ---
LOCAL_PYTHIA_PATHS = {
    "70m":  os.path.expanduser("~/pythia70m_local"),
    "160m": os.path.expanduser("~/pythia160m_local"),
    "410m": os.path.expanduser("~/pythia410m_local"),
    "1b":   os.path.expanduser("~/pythia1b_local"),
}
_local = LOCAL_PYTHIA_PATHS.get(MODEL_SIZE)
load_src = _local if (_local and os.path.isdir(_local)) else REPO_ID
print(f"Loading {load_src}...", flush=True)
tokenizer = AutoTokenizer.from_pretrained(load_src)
model = AutoModelForCausalLM.from_pretrained(load_src)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
print("Model+tokenizer loaded.", flush=True)


def read_jsonl(p):
    with open(p, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


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
print(f"train: {len(train_data)}, eval: {len(eval_data)}")

# Optional class-stratified subsample of train (eval kept full).
if SUBSAMPLE_N > 0 and SUBSAMPLE_N < len(train_data):
    rng = np.random.default_rng(SEED)
    by_class = {c: [] for c in range(Y)}
    for ex in train_data:
        by_class[int(ex["answer"])].append(ex)
    per_class = SUBSAMPLE_N // Y
    extra = SUBSAMPLE_N - per_class * Y  # distribute remainder over first few classes
    sub = []
    for c in range(Y):
        n_take = per_class + (1 if c < extra else 0)
        n_take = min(n_take, len(by_class[c]))
        idx = rng.choice(len(by_class[c]), size=n_take, replace=False)
        sub.extend([by_class[c][i] for i in idx])
    rng.shuffle(sub)
    train_data = sub
    print(f"  Subsampled train to {len(train_data)} examples "
          f"(class-stratified, ~{per_class}/class)")

train_dataset = Dataset.from_list(train_data).map(tokenize_and_mask, remove_columns=["prompt", "answer"])
eval_dataset = Dataset.from_list(eval_data).map(tokenize_and_mask, remove_columns=["prompt", "answer"])


# --- Metrics (off-by-one-fixed, first answer token match) ---
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
         pred_ids[:, :-1]], axis=1,
    )
    mask = label_ids != -100
    correct = ((shifted_pred == label_ids) & mask).sum()
    total = mask.sum()
    token_acc = float(correct / total) if total > 0 else 0.0

    move_matches = []
    for i in range(pred_ids.shape[0]):
        ans_pos = np.where(mask[i])[0]
        if ans_pos.size == 0 or ans_pos[0] == 0:
            move_matches.append(False)
            continue
        first_ans = int(ans_pos[0])
        move_matches.append(int(shifted_pred[i, first_ans]) == int(label_ids[i, first_ans]))
    return {
        "token_acc": token_acc,
        "move_acc": float(np.mean(move_matches)) if move_matches else 0.0,
    }


# --- HF push (with retry on transient SSL errors) ---
api = HfApi()


def _hf_retry(fn, *args, what="hf", attempts=4, **kwargs):
    import time
    for i in range(attempts):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            print(f"  {what} attempt {i + 1}/{attempts} failed: {e}", flush=True)
            time.sleep(5 * (i + 1))
    print(f"  {what} permanently failed; continuing.", flush=True)
    return None


_hf_retry(api.create_repo, HF_REPO, exist_ok=True, repo_type="model", what="create_repo")
_hf_retry(api.update_repo_settings, HF_REPO, gated="manual", what="update_repo_settings")


def save_checkpoint_to_hub(model, tokenizer, step):
    tmp_dir = tempfile.mkdtemp()
    try:
        model.save_pretrained(tmp_dir)
        tokenizer.save_pretrained(tmp_dir)
        branch_name = f"step-{step}"
        for attempt in range(3):
            try:
                try:
                    api.create_branch(HF_REPO, branch=branch_name)
                except Exception:
                    pass
                api.upload_folder(folder_path=tmp_dir, repo_id=HF_REPO, revision=branch_name,
                                  commit_message=f"Checkpoint at step {step}", create_pr=False)
                print(f"  Pushed step-{step}", flush=True)
                return
            except Exception as e:
                print(f"  push attempt {attempt + 1}/3 failed: {e}", flush=True)
        print(f"  push permanently failed for step-{step}", flush=True)
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
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    overwrite_output_dir=True,
    max_steps=MAX_STEPS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    learning_rate=LR,
    weight_decay=WD,
    warmup_ratio=0.1,
    logging_steps=LOG_EVERY,
    evaluation_strategy="steps",
    eval_steps=EVAL_EVERY,
    save_strategy="no",
    lr_scheduler_type="cosine",
    report_to="none",
    seed=SEED,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset={"eval": eval_dataset},
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    callbacks=[HFSaveCallback(), JsonlLogCallback(METRICS_JSONL)],
)

trainer.train()
print("Done.")
