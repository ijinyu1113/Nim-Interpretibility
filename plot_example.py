"""
Generalized accuracy-over-checkpoints plotter for Nim max_remove tasks.

Usage examples:
  python plot_maxrem_checks.py --files "purenums/data/results/7_checkpoint-*.jsonl" --total-per-rem 7:2000
  python plot_maxrem_checks.py --files "purenums/data/results/7_checkpoint-*.jsonl" --total-per-rem totals.json

Each input JSONL should contain ONLY the mistakes for that checkpoint. Provide
the total eval examples per max_remove via --total-per-rem so accuracy can be
computed as 1 - errors/total.
"""

import argparse
import glob
import json
import os
import re
from collections import defaultdict
from typing import Dict, Iterable, List

import matplotlib.pyplot as plt


def parse_totals(arg: str) -> Dict[int, int]:
    """Accepts a JSON/file path or a comma string like '3:30000,4:30000'."""
    if os.path.isfile(arg):
        with open(arg) as f:
            data = json.load(f)
        return {int(k): int(v) for k, v in data.items()}
    totals: Dict[int, int] = {}
    for part in arg.split(","):
        if not part.strip():
            continue
        if ":" not in part:
            raise ValueError(f"Bad total-per-rem entry: {part}")
        k, v = part.split(":", 1)
        totals[int(k.strip())] = int(v.strip())
    if not totals:
        raise ValueError("No totals parsed from --total-per-rem")
    return totals


def extract_checkpoint(path: str) -> int:
    """Grab checkpoint number from filename."""
    m = re.search(r"checkpoint[-_]?(\d+)", path)
    if m:
        return int(m.group(1))
    # fallback: last integer in filename
    m2 = re.findall(r"(\d+)", os.path.basename(path))
    if m2:
        return int(m2[-1])
    raise ValueError(f"Could not parse checkpoint from {path}")


def extract_max_remove(prompt: str, pattern: re.Pattern) -> int:
    """Find 'take between 1 and X coin' style max_remove."""
    m = pattern.search(prompt)
    return int(m.group(1)) if m else None


def load_files(patterns: Iterable[str]) -> Dict[str, int]:
    files: Dict[str, int] = {}
    for pat in patterns:
        for fn in glob.glob(pat):
            files[fn] = extract_checkpoint(fn)
    if not files:
        raise FileNotFoundError(f"No files matched patterns: {patterns}")
    return files


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--files",
        required=True,
        nargs="+",
        help='Glob(s) for error JSONL files, e.g. "357_checkpoint-*.jsonl"',
    )
    parser.add_argument(
        "--total-per-rem",
        required=True,
        help="Totals per max_remove (JSON file or comma string like '3:30000,4:30000')",
    )
    parser.add_argument(
        "--output-dir",
        default="results",
        help="Directory to write outputs (JSON + figure). Will be created if needed.",
    )
    parser.add_argument(
        "--exp-name",
        default="acc",
        help="Prefix for output files, e.g., 'acc_purenums7'.",
    )
    parser.add_argument(
        "--title",
        default="Accuracy over Checkpoints by Max Remove",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not show the plot window (useful in headless runs)",
    )
    parser.add_argument(
        "--max-remove-regex",
        default=r"take between 1 and (\d+) coin",
        help="Regex to extract max_remove from prompt text",
    )
    args = parser.parse_args()

    totals = parse_totals(args.total_per_rem)
    files = load_files(args.files)
    mr_pattern = re.compile(args.max_remove_regex, flags=re.IGNORECASE)

    os.makedirs(args.output_dir, exist_ok=True)
    json_path = os.path.join(args.output_dir, f"{args.exp_name}.json")
    fig_path = os.path.join(args.output_dir, f"{args.exp_name}.png")

    error_counts: Dict[int, Dict[int, int]] = defaultdict(lambda: defaultdict(int))
    for fname, ckpt in files.items():
        with open(fname) as f:
            for line in f:
                prompt = json.loads(line)["prompt"]
                mr = extract_max_remove(prompt, mr_pattern)
                if mr is None:
                    continue
                error_counts[mr][ckpt] += 1

    checkpoints: List[int] = sorted(set(files.values()))

    # Plotting
    plt.figure(figsize=(10, 6))
    markers = ["o", "^", "s", "D", "v", "P", "X"]
    results = {}

    for i, mr in enumerate(sorted(totals)):
        tot = totals[mr]
        accs = [1 - error_counts[mr].get(ck, 0) / tot for ck in checkpoints]
        results[mr] = {ck: acc for ck, acc in zip(checkpoints, accs)}
        marker = markers[i % len(markers)]
        line, = plt.plot(checkpoints, accs, marker=marker, label=f"max_remove={mr}")
        plt.hlines(
            1 / (mr + 1),
            checkpoints[0],
            checkpoints[-1],
            colors=[line.get_color()],
            linestyles="--",
            alpha=0.5,
        )

    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    plt.xlabel("Checkpoint")
    plt.ylabel("Accuracy")
    plt.title(args.title)
    plt.ylim(0, 1.05)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(fig_path, dpi=200)
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()

