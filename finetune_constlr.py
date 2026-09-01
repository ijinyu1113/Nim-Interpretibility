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
MAX_STEPS = int(sys.argv[4]) if len(sys.argv) > 4 else 25000  # was NUM_EPOCHS
MODEL_SIZE = sys.argv[5] if len(sys.argv) > 5 else "410m"
DATA_DIR = sys.argv[6] if len(sys.argv) > 6 else "../data/purenums"
STOP_AT_STEPS = int(sys.argv[7]) if len(sys.argv) > 7 else 0  # 0 = no early stop
STOP_EVAL_AT = float(sys.argv[8]) if len(sys.argv) > 8 else 0.0  # 0 = no eval-based stop
EVAL_EVERY_OVERRIDE = int(sys.argv[9]) if len(sys.argv) > 9 else 0  # 0 = default 250
SAVE_EVERY_OVERRIDE = int(sys.argv[10]) if len(sys.argv) > 10 else 0  # 0 = default 10000
INIT_FROM = sys.argv[11] if len(sys.argv) > 11 and sys.argv[11] != "-" else None
# INIT_FROM format: "hf_repo@revision@shortlabel" -> initialize the model from a
# prior fine-tune (curriculum/install experiments). Label lands in the TAG.
SEED = int(sys.argv[12]) if len(sys.argv) > 12 and sys.argv[12] != "-" else 42
# X-ABL "staircase ablation": penalty weight on the two coset SIBLINGS of the true
# answer (r+3, r+6 mod 9) at the answer position. Removes the coset heuristic's
# loss advantage (coset-consistent wrong answers cost MORE than random ones)
# WITHOUT erasing any information. 0 = off. Only defined for mod 9.
SIB_PENALTY = float(sys.argv[13]) if len(sys.argv) > 13 and sys.argv[13] != "-" else 0.0
if SIB_PENALTY > 0:
    assert MAX_REMOVE + 1 == 9, "sibling penalty implemented for mod 9 only"
# X-GHOST activation-level installation: inject an oracle phase plane
# alpha * ||h|| * (cos(2pi*ds/9) u1 + sin(2pi*ds/9) u2) into hidden_states[13]
# at the answer-predicting position, during training AND trainer evals. Saved
# checkpoints contain NO hook, so any standard eval of a checkpoint is the
# training-wheels-removed test for free. 0 = off.
GHOST_ALPHA = float(sys.argv[14]) if len(sys.argv) > 14 and sys.argv[14] != "-" else 0.0
if GHOST_ALPHA > 0:
    assert MAX_REMOVE + 1 == 9, "ghost injection implemented for mod 9 only"
    assert SIB_PENALTY == 0, "ghost and sibling penalty are separate arms"

MODEL_MAP = {
    "70m":  "EleutherAI/pythia-70m-deduped",
    "160m": "EleutherAI/pythia-160m-deduped",
    "410m": "EleutherAI/pythia-410m-deduped",
    "1b":   "EleutherAI/pythia-1b-deduped",
    "1.4b": "EleutherAI/pythia-1.4b-deduped",
    # cross-model law replication (short keys keep TAG/repo names clean;
    # llama/gemma are HF-gated — the token's account must have accepted terms)
    "qwen0.5b": "Qwen/Qwen2.5-0.5B",
    "llama1b":  "meta-llama/Llama-3.2-1B",
    "gemma2b":  "google/gemma-2-2b",
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
_init_tag = ""
if INIT_FROM:
    _init_tag = "_init" + INIT_FROM.split("@")[2]
_sib_tag = f"_sibpen{SIB_PENALTY:g}" if SIB_PENALTY > 0 else ""
_ghost_tag = f"_ghost{GHOST_ALPHA:g}" if GHOST_ALPHA > 0 else ""
TAG = f"mr{MAX_REMOVE}_{MODEL_SIZE}_seed{SEED}_lr{LR_STR}_wd{WD_STR}_constlr{MAX_STEPS}steps{_data_tag}{_eval_tag}{_init_tag}{_sib_tag}{_ghost_tag}"
# HF caps repo names at 96 chars; long TAGs (e.g. with _init labels) overflow and
# every checkpoint push silently fails. Shorten only when needed, keep it decodable.
_repo_name = f"ft_{TAG}_purenum"
if len(_repo_name) > 96:
    _short = (TAG.replace("_seed42", "")
                 .replace(f"_lr{LR_STR}_wd{WD_STR}", f"_l{LR_STR}w{WD_STR}")
                 .replace("constlr", "c")
                 .replace("steps_nimsimple_max", "s_nims")
                 .replace("_evalevery", "_e"))
    _repo_name = f"ft_{_short}"
    assert len(_repo_name) <= 96, f"repo name still too long ({len(_repo_name)}): {_repo_name}"
HF_REPO = f"ijinyu1113/{_repo_name}"
OUTPUT_DIR = f"/projects/benv/iyu1/ft_{TAG}_purenum"

METRICS_DIR = "new_result/purenum_metrics"
os.makedirs(METRICS_DIR, exist_ok=True)
METRICS_JSONL = f"{METRICS_DIR}/{TAG}.jsonl"

print(f"CONST LR: mr={MAX_REMOVE}  lr={LR}  wd={WD}  max_steps={MAX_STEPS}  size={MODEL_SIZE}")
print(f"  data_dir={DATA_DIR}")
print(f"  scheduler=constant, NO warmup")
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
tokenizer = AutoTokenizer.from_pretrained(load_src)
if INIT_FROM:
    _repo, _rev = INIT_FROM.split("@")[0], INIT_FROM.split("@")[1]
    print(f"Loading model from INIT_FROM: {_repo}@{_rev}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(_repo, revision=_rev)
else:
    print(f"Loading model from: {load_src}", flush=True)
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
    # per-example digit sum of the pile (used only by the X-GHOST hook; silently
    # dropped by the Trainer when remove_unused_columns=True, i.e. ghost off)
    m = INIT_PILE_RE.search(example["prompt"])
    tokenized["ds"] = sum(int(c) for c in m.group(1)) if m else 0
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
print(f"train size={len(train_data)}, steps/epoch={steps_per_epoch}, "
      f"max_steps={MAX_STEPS} (~{MAX_STEPS / steps_per_epoch:.1f} epochs)")

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    overwrite_output_dir=True,
    max_steps=MAX_STEPS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    learning_rate=LR,
    weight_decay=WD,
    warmup_steps=0,
    logging_steps=LOG_EVERY,
    evaluation_strategy="steps",
    eval_steps=EVAL_EVERY,
    save_strategy="no",
    load_best_model_at_end=False,
    lr_scheduler_type="constant",
    report_to="none",
    seed=SEED,
    remove_unused_columns=(GHOST_ALPHA <= 0),  # keep the "ds" column for the ghost hook
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

class SiblingPenaltyTrainer(Trainer):
    """X-ABL: adds SIB_PENALTY * (p(r+3 mod 9) + p(r+6 mod 9)) at the first
    answer-token position. Caution for interpretation: this shapes the OUTPUT
    layer; whether the internal coset stage still forms is a probe question
    (phantom-transitions caveat) -- the behavioral endpoint (total steps to
    f95 vs scratch) is the primary pre-registered quantity regardless."""
    _sib_cache = None

    def _sibling_ids(self, device):
        if SiblingPenaltyTrainer._sib_cache is None:
            import torch
            digit_ids = [tokenizer.encode(str(d))[0] for d in range(9)]
            ids = torch.tensor(digit_ids)
            sib1 = torch.tensor([digit_ids[(r + 3) % 9] for r in range(9)])
            sib2 = torch.tensor([digit_ids[(r + 6) % 9] for r in range(9)])
            SiblingPenaltyTrainer._sib_cache = (ids, sib1, sib2)
        ids, s1, s2 = SiblingPenaltyTrainer._sib_cache
        return ids.to(device), s1.to(device), s2.to(device)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        import torch
        outputs = model(**inputs)
        loss = outputs.loss
        labels = inputs["labels"]
        logits = outputs.logits[:, :-1, :]
        labs = labels[:, 1:]
        mask = labs != -100
        has_ans = mask.any(dim=1)
        first = mask.float().argmax(dim=1)
        rows = torch.arange(labs.size(0), device=labs.device)
        lab_first = labs[rows, first]
        ids, sib1, sib2 = self._sibling_ids(labs.device)
        is_digit = lab_first.unsqueeze(1) == ids.unsqueeze(0)
        valid = has_ans & is_digit.any(dim=1)
        if valid.any():
            r = is_digit.float().argmax(dim=1)[valid].long()
            probs = torch.softmax(logits[rows[valid], first[valid]].float(), dim=-1)
            pen = (probs.gather(1, sib1[r].unsqueeze(1))
                   + probs.gather(1, sib2[r].unsqueeze(1)))
            loss = loss + SIB_PENALTY * pen.mean()
        return (loss, outputs) if return_outputs else loss


class GhostFeatureTrainer(Trainer):
    """X-GHOST: hand the model the period-9 phase plane and see if it learns
    at local-modulus speed. Injects alpha*||h||*(cos(2pi ds/9) u1 + sin u2)
    into hidden_states[13] (output of block 12) at the answer-predicting
    position, for train and eval batches alike."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        import torch
        g = torch.Generator().manual_seed(0)
        u = torch.randn(2, self.model.config.hidden_size, generator=g)
        u[0] = u[0] / u[0].norm()
        u[1] = u[1] - (u[1] @ u[0]) * u[0]
        u[1] = u[1] / u[1].norm()
        self._u = u
        self._ctx = {}
        self.model.gpt_neox.layers[12].register_forward_hook(self._hook)

    def _hook(self, module, inputs, output):
        import torch
        if not self._ctx:
            return output
        h = output[0] if isinstance(output, tuple) else output
        u = self._u.to(device=h.device, dtype=torch.float32)
        rows = torch.arange(h.shape[0], device=h.device)
        pos = self._ctx["pos"].to(h.device)
        theta = 2 * torch.pi * self._ctx["ds"].to(h.device).float() / 9.0
        v = torch.cos(theta).unsqueeze(1) * u[0] + torch.sin(theta).unsqueeze(1) * u[1]
        hv = h[rows, pos].float()
        h[rows, pos] = (hv + GHOST_ALPHA * hv.norm(dim=-1, keepdim=True) * v).to(h.dtype)
        return (h,) + output[1:] if isinstance(output, tuple) else h

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        ds = inputs.pop("ds")
        labels = inputs["labels"]
        mask = labels != -100
        first = mask.float().argmax(dim=1)
        self._ctx = {"ds": ds, "pos": (first - 1).clamp(min=0)}
        try:
            outputs = model(**inputs)
        finally:
            self._ctx = {}
        loss = outputs.loss
        return (loss, outputs) if return_outputs else loss


TrainerCls = Trainer
if SIB_PENALTY > 0:
    TrainerCls = SiblingPenaltyTrainer
    print(f"X-ABL sibling penalty ACTIVE: lambda={SIB_PENALTY}")
if GHOST_ALPHA > 0:
    TrainerCls = GhostFeatureTrainer
    print(f"X-GHOST injection ACTIVE: alpha={GHOST_ALPHA} at hidden_states[13]")

trainer = TrainerCls(
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
