"""For each 345678_train_errs_checkpoint-*.jsonl in the current directory,
count error entries per max_remove bucket. Build a CSV with checkpoints as
columns and max_remove as rows.

Each err line is expected to have a "max_remove" field (written by
test_model_maxrem.py). For older files without it, we fall back to parsing
the prompt.
"""
import csv
import glob
import json
import os
import re
from collections import defaultdict

INPUT_GLOB = "345678_train_errs_checkpoint-*.jsonl"
OUT_CSV = "new_result/345678_ckpt_errors_by_maxrem.csv"
OUT_CSV_ACC = "new_result/345678_ckpt_acc_by_maxrem.csv"
BUCKET_CSV = "new_result/345678_train_10k_buckets.csv"
MAXREMS = list(range(3, 9))
PATTERN = re.compile(r"Each player can take between 1 and (\d+) coin")


def load_bucket_totals():
    """Read max_remove -> total count from the saved bucket CSV."""
    totals = {}
    with open(BUCKET_CSV, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            mr_str = row["max_remove"]
            if mr_str.isdigit():
                totals[int(mr_str)] = int(row["count"])
    return totals


def step_from_filename(path):
    m = re.search(r"checkpoint-(\d+)", os.path.basename(path))
    return int(m.group(1)) if m else -1


def main():
    files = sorted(glob.glob(INPUT_GLOB), key=step_from_filename)
    if not files:
        print(f"No files match {INPUT_GLOB}")
        return

    # counts[ckpt_step][mr] = #errors
    counts = defaultdict(lambda: {mr: 0 for mr in MAXREMS})
    unknown_per_ckpt = defaultdict(int)
    total_per_ckpt = defaultdict(int)
    ckpts = []

    for path in files:
        step = step_from_filename(path)
        ckpts.append(step)
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                total_per_ckpt[step] += 1
                mr = obj.get("max_remove")
                if mr is None:
                    m = PATTERN.search(obj.get("prompt", ""))
                    mr = int(m.group(1)) if m else None
                if mr in counts[step]:
                    counts[step][mr] += 1
                else:
                    unknown_per_ckpt[step] += 1

    # Print preview
    print(f"{'max_remove':>10}", end="")
    for s in ckpts:
        print(f" {f'ckpt-{s}':>12}", end="")
    print()
    for mr in MAXREMS:
        print(f"{mr:>10}", end="")
        for s in ckpts:
            print(f" {counts[s][mr]:>12}", end="")
        print()
    print(f"{'total_err':>10}", end="")
    for s in ckpts:
        print(f" {total_per_ckpt[s]:>12}", end="")
    print()

    # Save raw error counts CSV: rows = max_remove + total_errors, cols = ckpts
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["max_remove"] + [f"ckpt-{s}" for s in ckpts])
        for mr in MAXREMS:
            w.writerow([mr] + [counts[s][mr] for s in ckpts])
        w.writerow(["total_errors"] + [total_per_ckpt[s] for s in ckpts])
    print(f"\nSaved: {OUT_CSV}")

    # Save accuracy CSV (1 - errors / examples-of-that-max_remove)
    if not os.path.isfile(BUCKET_CSV):
        print(f"WARN: {BUCKET_CSV} not found — skipping accuracy CSV.")
        return
    totals = load_bucket_totals()

    print(f"\n{'max_remove':>10}", end="")
    for s in ckpts:
        print(f" {f'ckpt-{s}':>12}", end="")
    print("    (accuracy)")
    with open(OUT_CSV_ACC, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["max_remove"] + [f"ckpt-{s}" for s in ckpts])
        for mr in MAXREMS:
            denom = totals.get(mr, 0)
            row_vals = []
            for s in ckpts:
                if denom > 0:
                    row_vals.append(1.0 - counts[s][mr] / denom)
                else:
                    row_vals.append(float("nan"))
            print(f"{mr:>10}", end="")
            for v in row_vals:
                print(f" {v:>12.4f}", end="")
            print()
            w.writerow([mr] + [f"{v:.6f}" for v in row_vals])
    print(f"\nSaved: {OUT_CSV_ACC}")


if __name__ == "__main__":
    main()
