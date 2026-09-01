"""Old training recipe — replicate the pre-investigation setup but with the
NEW corrected move_acc metric and the NEW disjoint-pile train/eval data.

Old recipe specifics:
  - LR scheduler: cosine
  - warmup_ratio: 0.1
  - num_train_epochs: 300 (~70k steps for 15k examples / batch 64)
  - lr = 3e-5
  - weight_decay = 0.05

Question this answers: did the old config "grok" because the optimization
regime was actually better, or only because the old eval was distribution-
overlapping with train? Same data-side ceiling, or different?

CLI:
    python finetune_oldconfig.py <mr> [<lr_str>] [<wd_str>] [<num_epochs>]
Defaults:  lr=3e-5, wd=0.05, num_epochs=300, seed=42, size=410m
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
LR_STR = sys.argv[2] if len(sys.argv) > 2 else "3e-5"
LR = float(LR_STR)
WD_STR = sys.argv[3] if len(sys.argv) > 3 else "0.05"
WD = float(WD_STR)
NUM_EPOCHS = int(sys.argv[4]) if len(sys.argv) > 4 else 300
MODEL_SIZE = sys.argv[5] if len(sys.argv) > 5 else "410m"
DATA_DIR = sys.argv[6] if len(sys.argv) > 6 else "../data/purenums"
STOP_AT_STEPS = int(sys.argv[7]) if len(sys.argv) > 7 else 0  # 0 = no early stop
STOP_EVAL_AT = float(sys.argv[8]) if len(sys.argv) > 8 else 0.0  # 0 = no eval-based stop
EVAL_EVERY_OVERRIDE = int(sys.argv[9]) if len(sys.argv) > 9 else 0  # 0 = default 250
SAVE_EVERY_OVERRIDE = int(sys.argv[10]) if len(sys.argv) > 10 else 0  # 0 = default 10000

SEED = 42

MODEL_MAP = {
    "70m":  "EleutherAI/pythia-70m-deduped",
    "160m": "EleutherAI/pythia-160m-deduped",
    "410m": "EleutherAI/pythia-410m-deduped",
    "1b":   "EleutherAI/pythia-1b-deduped",
    "1.4b": "EleutherAI/pythia-1.4b-deduped",
}
REPO_ID = MODEL_MAP[MODEL_SIZE]

TRAIN_FILE = f"{DATA_DIR}/{MAX_REMOVE}_train.jsonl"
EVAL_FILE = f"{DATA_DIR}/{MAX_REMOVE}_eval.jsonl"
MAX_LENGTH = 128
EVAL_EVERY = EVAL_EVERY_OVERRIDE if EVAL_EVERY_OVERRIDE > 0 else 250
LOG_EVERY = 250
SAVE_EVERY = SAVE_EVERY_OVERRIDE if SAVE_EVERY_OVERRIDE > 0 else 10000
BATCH_SIZE = 64
TRAIN_ACC_SAMPLES = 1000

# Add data dir basename to tag so non-default data sources get distinct output
# filenames (but keep backward compat: the default "purenums" adds no tag).
_data_basename = os.path.basename(DATA_DIR.rstrip("/"))
_data_tag = "" if _data_basename == "purenums" else f"_{_data_basename}"
_eval_tag = "" if EVAL_EVERY == 250 else f"_evalevery{EVAL_EVERY}"
TAG = f"mr{MAX_REMOVE}_{MODEL_SIZE}_seed{SEED}_lr{LR_STR}_wd{WD_STR}_oldcfg{NUM_EPOCHS}ep{_data_tag}{_eval_tag}"
HF_REPO = f"ijinyu1113/ft_{TAG}_purenum"
OUTPUT_DIR = f"/projects/benv/iyu1/ft_{TAG}_purenum"

METRICS_DIR = "new_result/purenum_metrics"
os.makedirs(METRICS_DIR, exist_ok=True)
METRICS_JSONL = f"{METRICS_DIR}/{TAG}.jsonl"

print(f"OLD CONFIG: mr={MAX_REMOVE}  lr={LR}  wd={WD}  epochs={NUM_EPOCHS}  size={MODEL_SIZE}")
print(f"  data_dir={DATA_DIR}")
print(f"  scheduler=cosine, warmup_ratio=0.1")
print(f"  HF_REPO={HF_REPO}")
print(f"  METRICS_JSONL={METRICS_JSONL}")


def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


# Prefer local copy of Pythia to sidestep flaky HF Hub round-trips during load.
# Falls back to HF Hub if the local dir doesn't exist for this size.
LOCAL_PYTHIA_PATHS = {
    "70m":  os.path.expanduser("~/pythia70m_local"),
    "160m": os.path.expanduser("~/pythia160m_local"),
    "410m": os.path.expanduser("~/pythia410m_local"),
    "1b":   os.path.expanduser("~/pythia1b_local"),
    "1.4b": os.path.expanduser("~/pythia1.4b_local"),
}
_local = LOCAL_PYTHIA_PATHS.get(MODEL_SIZE)
load_src = _local if (_local and os.path.isdir(_local)) else REPO_ID
print(f"Loading model from: {load_src}", flush=True)
tokenizer = AutoTokenizer.from_pretrained(load_src)
model = AutoModelForCausalLM.from_pretrained(load_src)
print("Model+tokenizer loaded.", flush=True)
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


def validate_prompt_boundary(records, name, n_check=100):
    """Hard-fail if the prompt is not a clean token-prefix of prompt+answer.
    A prompt ending in a trailing space merges the space into the answer token
    (' 6'), so labels[:prompt_len] = -100 masks the answer itself and the model
    is only supervised on padding. This silently voided every run on the old
    modarith/ladder-2a data — never let it recur."""
    for ex in records[:n_check]:
        pe = tokenizer(ex["prompt"], truncation=True, max_length=MAX_LENGTH,
                       padding=False)["input_ids"]
        fe = tokenizer(ex["prompt"] + ex["answer"], truncation=True,
                       max_length=MAX_LENGTH, padding=False)["input_ids"]
        if fe[:len(pe)] != pe or len(fe) <= len(pe):
            raise ValueError(
                f"TOKENIZATION BOUNDARY VIOLATION in {name}: prompt is not a "
                f"clean token-prefix of prompt+answer (the answer would be "
                f"masked out of the loss).\n  prompt tail: {ex['prompt'][-30:]!r}"
                f"\n  answer: {ex['answer']!r}\n  Fix the data: no trailing "
                f"space on prompts; put the leading space on the answer.")
    print(f"  boundary check OK: {name} ({min(n_check, len(records))} examples)")


validate_prompt_boundary(train_data, "train")
validate_prompt_boundary(eval_data, "eval")

rng = np.random.default_rng(SEED)
train_acc_n = min(TRAIN_ACC_SAMPLES, len(train_data))
train_acc_idx = rng.choice(len(train_data), size=train_acc_n, replace=False)
train_acc_records = [train_data[i] for i in train_acc_idx]

EVAL_PILES = [compute_current_pile(ex["prompt"]) for ex in eval_data]
TRAIN_ACC_PILES = [compute_current_pile(ex["prompt"]) for ex in train_acc_records]
assert len(eval_data) != len(train_acc_records)
PILES_REGISTRY = {
    len(eval_data): EVAL_PILES,
    len(train_acc_records): TRAIN_ACC_PILES,
}

train_dataset = Dataset.from_list(train_data).map(tokenize_and_mask, remove_columns=["prompt", "answer"])
eval_dataset = Dataset.from_list(eval_data).map(tokenize_and_mask, remove_columns=["prompt", "answer"])
train_acc_dataset = Dataset.from_list(train_acc_records).map(tokenize_and_mask, remove_columns=["prompt", "answer"])


# --- Metrics (shifted, first-answer-token id match) ---
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


def _hf_retry(fn, *args, what="hf_call", attempts=4, **kwargs):
    """Retry a single HF API call with simple backoff. Logs and continues on
    final failure rather than crashing training."""
    import time
    for i in range(attempts):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            print(f"  {what} attempt {i + 1}/{attempts} failed: {e}", flush=True)
            time.sleep(5 * (i + 1))
    print(f"  {what} permanently failed; continuing without it.", flush=True)
    return None


_hf_retry(api.create_repo, HF_REPO, exist_ok=True, repo_type="model",
          what="create_repo")
_hf_retry(api.update_repo_settings, HF_REPO, gated="manual",
          what="update_repo_settings")


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
approx_total_steps = steps_per_epoch * NUM_EPOCHS
print(f"train size={len(train_data)}, steps/epoch={steps_per_epoch}, "
      f"epochs={NUM_EPOCHS} -> ~{approx_total_steps} total steps")

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    overwrite_output_dir=True,
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    learning_rate=LR,
    weight_decay=WD,
    warmup_ratio=0.1,
    logging_steps=LOG_EVERY,
    evaluation_strategy="steps",
    eval_steps=EVAL_EVERY,
    save_strategy="no",
    load_best_model_at_end=False,
    lr_scheduler_type="cosine",
    report_to="none",
    seed=SEED,
)

class StopAtStepCallback(TrainerCallback):
    def __init__(self, stop_at_step):
        self.stop_at_step = stop_at_step
    def on_step_end(self, args, state, control, **kwargs):
        if self.stop_at_step > 0 and state.global_step >= self.stop_at_step:
            print(f"Early-stop: reached step {state.global_step} >= {self.stop_at_step}", flush=True)
            control.should_training_stop = True


class StopAtEvalCallback(TrainerCallback):
    """Stop when eval_eval_move_acc >= threshold."""
    def __init__(self, threshold):
        self.threshold = threshold
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if not metrics:
            return
        ev = metrics.get("eval_eval_move_acc")
        if ev is not None and ev >= self.threshold:
            print(f"Early-stop: eval_move_acc={ev:.4f} >= {self.threshold} at step {state.global_step}", flush=True)
            control.should_training_stop = True


_callbacks = [HFSaveCallback(), JsonlLogCallback(METRICS_JSONL)]
if STOP_AT_STEPS > 0:
    print(f"Early-stop enabled at step {STOP_AT_STEPS} "
          f"(cosine schedule still over full epoch budget)")
    _callbacks.append(StopAtStepCallback(STOP_AT_STEPS))
if STOP_EVAL_AT > 0:
    print(f"Early-stop enabled when eval_move_acc >= {STOP_EVAL_AT}")
    _callbacks.append(StopAtEvalCallback(STOP_EVAL_AT))

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset={"eval": eval_dataset, "train": train_acc_dataset},
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    callbacks=_callbacks,
)

# Initial eval at step=0 (random init baseline) so the JSONL captures the
# pre-training state. Logged via JsonlLogCallback as step=0.
# Iterate manually over the eval_dataset dict — calling trainer.evaluate()
# without args on a dict eval_dataset trips a KeyError in the dataloader.
print("Running initial eval at step=0...", flush=True)
if isinstance(trainer.eval_dataset, dict):
    for name, ds in trainer.eval_dataset.items():
        trainer.evaluate(eval_dataset=ds, metric_key_prefix=f"eval_{name}")
else:
    trainer.evaluate()

trainer.train()
print("Done.")
