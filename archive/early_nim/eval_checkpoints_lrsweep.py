"""Re-evaluate HF-Hub checkpoints for one (max_remove, lr) cell of the
LR sweep, computing the corrected move_acc = first-answer-token-id match.

For the given (mr, lr), pulls branches
    step-10000, step-20000, step-30000, step-40000, step-50000
from
    ijinyu1113/ft_mr{mr}_410m_seed42_lr{lr}_purenum
and writes per-step metrics to
    new_result/purenum_metrics/mr{mr}_410m_seed42_lr{lr}_ckpteval.jsonl

Run on the cluster (needs GPU + HF_TOKEN).
Usage:
    python eval_checkpoints_lrsweep.py <mr> <lr_str>
    e.g. python eval_checkpoints_lrsweep.py 5 3e-5
"""
import json
import os
import sys

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MR = int(sys.argv[1])
LR_STR = sys.argv[2]
SEED = 42
SIZE = "410m"
STEPS = [10000, 20000, 30000, 40000, 50000]
EVAL_FILE = f"../data/purenums/{MR}_eval.jsonl"
TRAIN_FILE = f"../data/purenums/{MR}_train.jsonl"
TRAIN_ACC_SAMPLES = 1000
MAX_LENGTH = 128
BATCH_SIZE = 64
REPO_ID = f"ijinyu1113/ft_mr{MR}_{SIZE}_seed{SEED}_lr{LR_STR}_purenum"

OUT_DIR = "new_result/purenum_metrics"
os.makedirs(OUT_DIR, exist_ok=True)
OUT_PATH = os.path.join(
    OUT_DIR, f"mr{MR}_{SIZE}_seed{SEED}_lr{LR_STR}_ckpteval.jsonl"
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def tokenize_with_mask(example, tokenizer):
    full_text = example["prompt"] + example["answer"]
    tok = tokenizer(
        full_text, truncation=True, max_length=MAX_LENGTH,
        padding="max_length", return_tensors="pt",
    )
    prompt_ids = tokenizer(
        example["prompt"], truncation=True, max_length=MAX_LENGTH, padding=False
    )["input_ids"]
    prompt_len = min(len(prompt_ids), MAX_LENGTH)
    labels = tok["input_ids"].clone().squeeze(0)
    labels[:prompt_len] = -100
    return tok["input_ids"].squeeze(0), tok["attention_mask"].squeeze(0), labels, prompt_len


@torch.no_grad()
def eval_dataset(model, tokenizer, examples):
    first_correct = 0
    token_correct = 0
    token_total = 0
    n = 0
    for batch_start in range(0, len(examples), BATCH_SIZE):
        batch = examples[batch_start: batch_start + BATCH_SIZE]
        input_ids_list, attn_list, labels_list, plen_list = [], [], [], []
        for ex in batch:
            ids, attn, lbl, plen = tokenize_with_mask(ex, tokenizer)
            input_ids_list.append(ids)
            attn_list.append(attn)
            labels_list.append(lbl)
            plen_list.append(plen)
        input_ids = torch.stack(input_ids_list).to(device)
        attn = torch.stack(attn_list).to(device)
        labels = torch.stack(labels_list).to(device)
        logits = model(input_ids=input_ids, attention_mask=attn).logits
        preds = logits.argmax(dim=-1)  # (B, L); preds[:, t-1] predicts position t

        for i in range(input_ids.size(0)):
            plen = plen_list[i]
            if plen <= 0 or plen >= MAX_LENGTH:
                continue
            first_pred = preds[i, plen - 1].item()
            first_gold = input_ids[i, plen].item()
            if first_pred == first_gold:
                first_correct += 1
            n += 1
            # All-answer-token acc, properly shifted
            mask = labels[i] != -100
            idx = mask.nonzero(as_tuple=True)[0]
            valid = idx[idx > 0]
            gold_tokens = input_ids[i, valid]
            pred_tokens = preds[i, valid - 1]
            token_correct += (pred_tokens == gold_tokens).sum().item()
            token_total += valid.numel()
    return {
        "move_acc": first_correct / n if n > 0 else 0.0,
        "token_acc": token_correct / token_total if token_total > 0 else 0.0,
    }


def main():
    eval_data = read_jsonl(EVAL_FILE)
    train_data = read_jsonl(TRAIN_FILE)
    rng = np.random.default_rng(SEED)
    n_train = min(TRAIN_ACC_SAMPLES, len(train_data))
    idx = rng.choice(len(train_data), size=n_train, replace=False)
    train_acc_data = [train_data[i] for i in idx]

    print(f"mr={MR} lr={LR_STR}: |eval|={len(eval_data)}, |train_acc|={len(train_acc_data)}")
    print(f"repo={REPO_ID}")
    print(f"out={OUT_PATH}")
    # Truncate output
    with open(OUT_PATH, "w"):
        pass

    for step in STEPS:
        branch = f"step-{step}"
        print(f"  loading {REPO_ID}@{branch}")
        try:
            tokenizer = AutoTokenizer.from_pretrained(REPO_ID, revision=branch)
            model = AutoModelForCausalLM.from_pretrained(REPO_ID, revision=branch).to(device)
        except Exception as e:
            print(f"    SKIP step-{step}: {e}")
            continue
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model.eval()
        e_stats = eval_dataset(model, tokenizer, eval_data)
        t_stats = eval_dataset(model, tokenizer, train_acc_data)
        row = {
            "step": step,
            "eval_eval_move_acc": e_stats["move_acc"],
            "eval_eval_token_acc": e_stats["token_acc"],
            "eval_train_move_acc": t_stats["move_acc"],
            "eval_train_token_acc": t_stats["token_acc"],
        }
        with open(OUT_PATH, "a") as f:
            f.write(json.dumps(row) + "\n")
        print(f"    eval  move={e_stats['move_acc']:.3f} tok={e_stats['token_acc']:.3f}  "
              f"train move={t_stats['move_acc']:.3f} tok={t_stats['token_acc']:.3f}")
        del model, tokenizer
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
