"""Make a fixed-boundary copy of a broken (trailing-space) eval file, so an
old checkpoint can be evaluated on its own task with the prompt ending at '='.

If the trailing-space masking artifact is real, a checkpoint from the old
modarith runs should score ~chance on this fixed file (it was never supervised
on answers). Usage:
    python fix_trailing_space_eval.py data/modarith_subtract_max500/8_eval.jsonl data/modarith_fixcheck/8_eval.jsonl
Then:
    python eval_coset_confusion.py ijinyu1113/ft_mr8_410m_seed42_lr3e-5_wd0.05_oldcfg300ep_modarith_subtract_max500_evalevery100_purenum data/modarith_fixcheck/8_eval.jsonl 8 step-30000
"""
import json
import os
import sys

src, dst = sys.argv[1], sys.argv[2]
os.makedirs(os.path.dirname(dst), exist_ok=True)
n = 0
with open(src) as fi, open(dst, "w") as fo:
    for ln in fi:
        r = json.loads(ln)
        r["prompt"] = r["prompt"].rstrip()   # '...mod 9 = ' -> '...mod 9 ='
        fo.write(json.dumps(r) + "\n")
        n += 1
print(f"Wrote {n} fixed rows -> {dst}")
