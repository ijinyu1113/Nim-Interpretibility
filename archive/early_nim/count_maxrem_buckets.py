"""Count how many examples in 345678_train_10k.jsonl belong to each
max_remove bucket and save the table to CSV.

Usage:
    python count_maxrem_buckets.py [path_to_jsonl]
"""
import csv
import json
import os
import re
import sys
from collections import Counter

DEFAULT_PATH = "345678_train_10k.jsonl"
OUT_CSV = "new_result/345678_train_10k_buckets.csv"
PATTERN = re.compile(r"Each player can take between 1 and (\d+) coin")


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PATH
    counts = Counter()
    unknown = 0
    total = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            total += 1
            obj = json.loads(line)
            m = PATTERN.search(obj.get("prompt", ""))
            if m:
                counts[int(m.group(1))] += 1
            else:
                unknown += 1

    print(f"file:  {path}")
    print(f"total: {total}")
    print()
    print(f"{'max_remove':>10} {'count':>7} {'pct':>7}")
    for mr in sorted(counts):
        print(f"{mr:>10} {counts[mr]:>7} {100 * counts[mr] / total:>6.2f}%")
    if unknown:
        print(f"{'(unknown)':>10} {unknown:>7} {100 * unknown / total:>6.2f}%")

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["max_remove", "count", "pct"])
        for mr in sorted(counts):
            w.writerow([mr, counts[mr], f"{100 * counts[mr] / total:.4f}"])
        if unknown:
            w.writerow(["unknown", unknown, f"{100 * unknown / total:.4f}"])
        w.writerow(["total", total, "100.0000"])
    print(f"\nSaved: {OUT_CSV}")


if __name__ == "__main__":
    main()
