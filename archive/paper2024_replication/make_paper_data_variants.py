"""Download the original paper purenums data from GitHub
(https://github.com/LeoV974/nim_data/tree/main/purenums) into 3 variants:

  A: exact original   — answer "take N coins" (with N=-1 for losing states)
  B: bare-digit-with-neg-one — answer "-1" / "1" / ... / "MR"
  C: bare-digit-with-zero    — answer "0"  / "1" / ... / "MR"

A is downloaded as-is. B is derived from A by stripping "take " and " coins"
from each answer. C is derived from B by mapping "-1" -> "0".

All 3 variants share identical prompts and identical train/eval split — only
the answer field differs. This isolates "answer-format" effect on learning
dynamics versus the paper's exact baseline.

Usage:
    python make_paper_data_variants.py
Writes:
    data/purenums_paperA/<mr>_{train,eval}.jsonl
    data/purenums_paperB/<mr>_{train,eval}.jsonl
    data/purenums_paperC/<mr>_{train,eval}.jsonl
"""
import json
import os
import re
import urllib.request

REPO_RAW = "https://raw.githubusercontent.com/LeoV974/nim_data/main/purenums"
MRS = [3, 4, 5, 6, 7, 8]
SPLITS = ["train", "eval"]
OUT_A = "data/purenums_paperA"
OUT_B = "data/purenums_paperB"
OUT_C = "data/purenums_paperC"

ANS_TAKE_RE = re.compile(r"^take (-?\d+) coins?$")


def to_bare(answer):
    m = ANS_TAKE_RE.match(answer)
    if not m:
        raise ValueError(f"Unexpected answer format: {answer!r}")
    return m.group(1)


def fetch(mr, split):
    url = f"{REPO_RAW}/{mr}_{split}.jsonl"
    print(f"  GET {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "curl"})
    with urllib.request.urlopen(req) as r:
        return r.read().decode()


def write_records(path, records):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def main():
    for d in (OUT_A, OUT_B, OUT_C):
        os.makedirs(d, exist_ok=True)

    for mr in MRS:
        for split in SPLITS:
            raw = fetch(mr, split)
            recs_a = [json.loads(ln) for ln in raw.splitlines() if ln.strip()]
            recs_b, recs_c = [], []
            for r in recs_a:
                bare = to_bare(r["answer"])
                rb = {"prompt": r["prompt"], "answer": bare}
                rc = {"prompt": r["prompt"], "answer": "0" if bare == "-1" else bare}
                recs_b.append(rb)
                recs_c.append(rc)
            write_records(f"{OUT_A}/{mr}_{split}.jsonl", recs_a)
            write_records(f"{OUT_B}/{mr}_{split}.jsonl", recs_b)
            write_records(f"{OUT_C}/{mr}_{split}.jsonl", recs_c)
            print(f"  mr={mr} {split}: {len(recs_a)} records "
                  f"(A/B/C written)")

    # Sanity check: print one example per variant for mr=3
    print("\n=== Sample (mr=3 train, first record) ===")
    for label, d in (("A", OUT_A), ("B", OUT_B), ("C", OUT_C)):
        with open(f"{d}/3_train.jsonl") as f:
            ex = json.loads(f.readline())
        print(f"[variant {label}] answer={ex['answer']!r}")


if __name__ == "__main__":
    main()
