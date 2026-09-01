"""Run inference on the eval set and save per-example predictions to a JSONL
for prediction-distribution / confusion-matrix / error-frequency analysis.

Output rows: prompt info, gold answer, predicted answer, correctness flag.
Output file: new_result/predictions/predictions_{config_tag}.jsonl

CLI:
    python eval_predictions.py <mr> [<repo>] [<revision>]
Defaults: repo = old-config final repo for this mr at 410m, revision = main
Examples:
    python eval_predictions.py 3
    python eval_predictions.py 6 ijinyu1113/ft_mr6_1b_seed42_lr3e-5_wd0.05_oldcfg300ep_purenum
"""
import json
import os
import re
import sys

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import list_repo_refs

MR = int(sys.argv[1])
REPO = sys.argv[2] if len(sys.argv) > 2 else (
    f"ijinyu1113/ft_mr{MR}_410m_seed42_lr3e-5_wd0.05_oldcfg300ep_purenum"
)
REVISION = sys.argv[3] if len(sys.argv) > 3 else None

EVAL_FILE = f"../data/purenums/{MR}_eval.jsonl"
MAX_LENGTH = 128
BATCH_SIZE = 32

OUT_DIR = "new_result/predictions"
os.makedirs(OUT_DIR, exist_ok=True)
config_tag = REPO.split('/')[-1].replace('_purenum', '')


def out_path_for(revision):
    """Per-revision output file. Same convention as before for no-revision call,
    plus a step-tag for explicit step revisions."""
    if revision is None or not revision.startswith("step-"):
        return f"{OUT_DIR}/predictions_{config_tag}.jsonl"
    return f"{OUT_DIR}/predictions_{config_tag}_{revision}.jsonl"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

INIT_PILE_RE = re.compile(r"There are (\d+) coins")
PROMPT_MOVE_RE = re.compile(r"take (\d+) coin", re.IGNORECASE)


def parse_piles(prompt):
    m = INIT_PILE_RE.search(prompt)
    if not m:
        return None, None
    initial = int(m.group(1))
    used = sum(int(x) for x in PROMPT_MOVE_RE.findall(prompt))
    return initial, initial - used


def read_jsonl(p):
    with open(p, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def latest_step_branch(repo):
    """Find the largest 'step-N' branch on the repo. Returns None if none exist."""
    try:
        branches = list_repo_refs(repo).branches
    except Exception as e:
        print(f"  warning: could not list branches for {repo}: {e}", flush=True)
        return None
    steps = []
    for b in branches:
        m = re.match(r"step-(\d+)$", b.name)
        if m:
            steps.append((int(m.group(1)), b.name))
    if not steps:
        return None
    steps.sort()
    return steps[-1][1]


def all_step_branches(repo):
    """Return list of step branch names sorted by step number ascending."""
    try:
        branches = list_repo_refs(repo).branches
    except Exception as e:
        print(f"  warning: could not list branches for {repo}: {e}", flush=True)
        return []
    steps = []
    for b in branches:
        m = re.match(r"step-(\d+)$", b.name)
        if m:
            steps.append((int(m.group(1)), b.name))
    steps.sort()
    return [name for _, name in steps]


def run_inference_for_revision(revision, eval_data):
    """Load model at `revision`, run inference on eval_data, write predictions."""
    out_path = out_path_for(revision)
    print(f"\nLoading {REPO}@{revision} ...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(REPO, revision=revision)
    model = AutoModelForCausalLM.from_pretrained(REPO, revision=revision).to(device)
    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    MODULUS = MR + 1
    results = []
    with torch.no_grad():
        for start in range(0, len(eval_data), BATCH_SIZE):
            batch = eval_data[start:start + BATCH_SIZE]
            prompts = [ex["prompt"] for ex in batch]
            golds = [ex["answer"] for ex in batch]
            enc = tokenizer(prompts, return_tensors="pt", padding=True,
                            truncation=True, max_length=MAX_LENGTH).to(device)
            outputs = model(**enc)
            logits = outputs.logits
            last_pos = enc["attention_mask"].sum(dim=1) - 1
            for i, prompt in enumerate(prompts):
                pos = last_pos[i].item()
                pred_token_id = int(logits[i, pos].argmax().item())
                pred_token_str = tokenizer.decode([pred_token_id], skip_special_tokens=True).strip()
                gold_text = golds[i]
                gold_token_ids = tokenizer.encode(gold_text, add_special_tokens=False)
                gold_token_id = int(gold_token_ids[0]) if gold_token_ids else -1
                gold_token_str = tokenizer.decode([gold_token_id], skip_special_tokens=True).strip()
                initial, final = parse_piles(prompt)
                gold_answer = (final % MODULUS) if final is not None else None
                pred_answer = int(pred_token_str) if pred_token_str.lstrip("-").isdigit() else None
                results.append({
                    "initial_pile": initial, "final_pile": final,
                    "gold_answer": gold_answer,
                    "gold_token_id": gold_token_id, "gold_token_str": gold_token_str,
                    "pred_token_id": pred_token_id, "pred_token_str": pred_token_str,
                    "pred_answer": pred_answer,
                    "correct": pred_token_id == gold_token_id,
                })
        # end of inference loop

    with open(out_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    correct = sum(1 for r in results if r["correct"])
    print(f"  Saved {out_path}  ({len(results)} examples, {correct} correct = {correct/len(results):.3f})",
          flush=True)
    del model, tokenizer
    torch.cuda.empty_cache()


def main():
    eval_data = read_jsonl(EVAL_FILE)
    print(f"eval: {len(eval_data)} examples; mr={MR}", flush=True)

    if REVISION == "all":
        revisions = all_step_branches(REPO)
        if not revisions:
            print(f"ERROR: no step-* branches on {REPO}")
            sys.exit(1)
        print(f"Found {len(revisions)} checkpoints: {revisions}", flush=True)
    elif REVISION is None:
        rev = latest_step_branch(REPO)
        if rev is None:
            print(f"ERROR: no step-* branches on {REPO} and no revision given.")
            sys.exit(1)
        print(f"Auto-selected latest checkpoint: {rev}", flush=True)
        revisions = [rev]
    else:
        revisions = [REVISION]

    for rev in revisions:
        run_inference_for_revision(rev, eval_data)


if __name__ == "__main__":
    main()
