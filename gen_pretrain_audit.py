"""Generate the pretraining-knowledge audit set for BASE Pythia (no training).

One jsonl, each row: {prompt, answer, format, m, magnitude, shots}.
Cells = format x modulus x magnitude-band x shot-count:

  formats:
    mod        "49059 mod 9 = "                                (math, literal token)
    pct        "49059 % 9 = "                                  (code syntax)
    remainder  "The remainder when 49059 is divided by 9 is "  (NL synonym)
    nimsimple  exact nimsimple template, mr = m-1              (concealed, +1)
  m in 2..10            (answers 0..9 stay single-token)
  magnitude bands: 1-digit .. 5-digit
  shots: 0/4/8/16 for the terse math formats, 0/8 for nimsimple
         (few-shot examples: same format+m+band, newline-separated, query last)

Answers are exact residues; the model is never trained — this measures what
pretraining already provides (zero-shot) and what is latent (few-shot ICL).

Usage:  python gen_pretrain_audit.py [--n-per-cell 200] [--seed 42]
Writes: data/pretrain_audit/audit.jsonl
"""
import argparse
import json
import os
import random

MODULI = list(range(2, 11))
BANDS = {1: (1, 9), 2: (10, 99), 3: (100, 999), 4: (1000, 9999), 5: (10000, 99999)}
SHOTS_MATH = [0, 4, 8, 16]
SHOTS_NIM = [0, 8]


def render(fmt, n, m):
    """Prompts end WITHOUT a trailing space (the space would merge into the
    answer token and put the model off-distribution at the last position);
    the natural continuation is the space-prefixed digit, which the eval's
    decode+strip parse handles."""
    if fmt == "mod":
        return f"{n} mod {m} =", str(n % m)
    if fmt == "pct":
        return f"{n} % {m} =", str(n % m)
    if fmt == "remainder":
        return f"The remainder when {n} is divided by {m} is", str(n % m)
    if fmt == "nimsimple":
        mr = m - 1
        return (f"Each player can take between 1 and {mr} coins on their turn. "
                f"There are {n} coins. On this turn, the player should take",
                str(n % m))
    raise ValueError(fmt)


def sample_n(rng, band, m, fmt):
    if fmt == "nimsimple" and m < 4:
        return None                  # mr<3 is a degenerate game; real runs used mr>=3
    lo, hi = BANDS[band]
    if fmt == "nimsimple":
        lo = max(lo, m + 1)          # pile must exceed the max take
        if lo > hi:
            return None              # degenerate cell (e.g. 1-digit band, big m)
    return rng.randint(lo, hi)


def make_prompt(rng, fmt, m, band, shots):
    q = sample_n(rng, band, m, fmt)
    if q is None:
        return None
    parts = []
    seen = {q}
    for _ in range(shots):
        for _ in range(50):
            e = sample_n(rng, band, m, fmt)
            if e is not None and e not in seen:
                seen.add(e)
                break
        ep, ea = render(fmt, e, m)
        parts.append(f"{ep} {ea}")   # natural text: "17 mod 9 = 8"
    qp, qa = render(fmt, q, m)
    sep = "\n\n" if fmt == "nimsimple" else "\n"
    prompt = (sep.join(parts) + sep + qp) if parts else qp
    return {"prompt": prompt, "answer": qa, "format": fmt, "m": m,
            "magnitude": band, "shots": shots}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-cell", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="data/pretrain_audit/audit.jsonl")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    n_rows = 0
    n_cells = 0
    with open(args.out, "w") as f:
        for fmt in ["mod", "pct", "remainder", "nimsimple"]:
            shot_list = SHOTS_NIM if fmt == "nimsimple" else SHOTS_MATH
            for m in MODULI:
                for band in BANDS:
                    for shots in shot_list:
                        rows = []
                        for _ in range(args.n_per_cell):
                            r = make_prompt(rng, fmt, m, band, shots)
                            if r is not None:
                                rows.append(r)
                        if not rows:
                            continue
                        n_cells += 1
                        for r in rows:
                            f.write(json.dumps(r) + "\n")
                            n_rows += 1
    print(f"Wrote {n_rows} rows across {n_cells} cells -> {args.out}")
    # spot-check one example per format
    seen = set()
    for ln in open(args.out):
        r = json.loads(ln)
        if r["format"] not in seen and r["shots"] > 0 and r["magnitude"] == 5:
            seen.add(r["format"])
            print(f"\n--- {r['format']} m={r['m']} shots={r['shots']} ---")
            print(repr(r["prompt"][:400]))
            print("answer:", repr(r["answer"]))


if __name__ == "__main__":
    main()
