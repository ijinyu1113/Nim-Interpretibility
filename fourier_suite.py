"""C4-as-dynamics: Fourier/periodic structure across training checkpoints.

Per checkpoint, two instruments (final token position):

  F-A  OUTPUT DFT over Z_9 (Nanda-style, vocab-independent). For each held-out
       example take the 9 digit-token logits, center, index by offset from the
       true answer Delta=(r-y) mod 9, average the profile over examples, DFT.
       Coset structure = energy at frequency 3; the fine solution needs 1,2,4.
       Report energy fractions. Works regardless of tokenizer/vocab size
       because the ANSWER space is Z_9.

  F-B  PERIOD-DECODABILITY SPECTRUM (adapted from single-token trig-fitting
       literature to the multi-token regime by choosing the coordinate):
       per layer, ridge-decode cos/sin(2*pi*x/T) from hidden states, where
       x = digitsum(N) with T in {3, 9}   (the constructed circuit's coordinate)
       x = N          with T in {2, 5, 10, 100} (the pretrained basis coordinate)
       Fit on train-pool operands, R2 on held-out. Reported per layer.

Usage:
  python fourier_suite.py <hf_repo> <mr> <train_jsonl> <eval_jsonl> <out_jsonl> [smin smax stride] [tok_base]
tok_base: base-model id for the tokenizer (default Pythia-410m; pass
Qwen/Qwen2.5-0.5B for the cross-model runs). Missing checkpoint revisions are
skipped with a warning rather than crashing (Qwen save cadence differs).
"""
import json
import re
import sys

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

REPO = sys.argv[1]
MR = int(sys.argv[2])
TRAIN_FILE = sys.argv[3]
EVAL_FILE = sys.argv[4]
OUT = sys.argv[5]
SMIN = int(sys.argv[6]) if len(sys.argv) > 6 else 250
SMAX = int(sys.argv[7]) if len(sys.argv) > 7 else 6000
STRIDE = int(sys.argv[8]) if len(sys.argv) > 8 else 250
TOK_BASE = sys.argv[9] if len(sys.argv) > 9 else "EleutherAI/pythia-410m-deduped"

assert torch.cuda.is_available(), "CUDA not available - refusing CPU crawl."
DEVICE = "cuda"
BATCH = 64
MAX_LEN = 128
N_FIT = 2000
N_SCORE = 2000
MOD = MR + 1
PILE_RE = re.compile(r"There are (\d+) coins")

PERIODS = [("ds", 3), ("ds", 9), ("N", 2), ("N", 5), ("N", 10), ("N", 100)]


def read_jsonl(p, n):
    rows = []
    with open(p) as f:
        for ln in f:
            rows.append(json.loads(ln))
            if len(rows) >= n:
                break
    return rows


def digitsum(n):
    return sum(int(c) for c in str(n))


def targets_for(piles):
    """[n_examples, 2*len(PERIODS)] cos/sin targets."""
    cols = []
    for coord, T in PERIODS:
        x = np.array([digitsum(p) if coord == "ds" else p for p in piles], dtype=np.float64)
        cols += [np.cos(2 * np.pi * x / T), np.sin(2 * np.pi * x / T)]
    return np.stack(cols, axis=1).astype(np.float32)


@torch.no_grad()
def forward_collect(model, tok, prompts, n_layers, hidden, digit_ids):
    """Final-position hidden states per layer + digit logits at final position."""
    N = len(prompts)
    X = np.zeros((N, n_layers + 1, hidden), dtype=np.float32)
    L9 = np.zeros((N, 9), dtype=np.float32)
    for s in range(0, N, BATCH):
        batch = prompts[s:s + BATCH]
        enc = tok(batch, return_tensors="pt", padding=True,
                  truncation=True, max_length=MAX_LEN).to(DEVICE)
        out = model(**enc, output_hidden_states=True)
        fpos = enc["attention_mask"].sum(1) - 1
        idx = torch.arange(len(batch), device=DEVICE)
        for li, h in enumerate(out.hidden_states):
            X[s:s + len(batch), li] = h[idx, fpos].float().cpu().numpy()
        L9[s:s + len(batch)] = out.logits[idx, fpos][:, digit_ids].float().cpu().numpy()
        if (s // BATCH) % 10 == 0:
            print(f"    fwd {s + len(batch)}/{N}", flush=True)
    return X, L9


def ridge_r2_all_layers(Xa, Ya, Xb, Yb, lam=10.0):
    """Multi-target closed-form ridge per layer, batched on GPU.
    Returns per-layer per-target R2 [L, n_targets]."""
    A = torch.tensor(np.ascontiguousarray(np.transpose(Xa, (1, 0, 2))), device=DEVICE)
    B = torch.tensor(np.ascontiguousarray(np.transpose(Xb, (1, 0, 2))), device=DEVICE)
    mu = A.mean(1, keepdim=True); sd = A.std(1, keepdim=True).clamp_min(1e-4)
    A = (A - mu) / sd; B = (B - mu) / sd
    L, N, H = A.shape
    Ya_t = torch.tensor(Ya, device=DEVICE)
    ym = Ya_t.mean(0, keepdim=True)
    Y = (Ya_t - ym).unsqueeze(0).expand(L, -1, -1)
    G = torch.bmm(A.transpose(1, 2), A) + lam * torch.eye(H, device=DEVICE).unsqueeze(0)
    W = torch.linalg.solve(G, torch.bmm(A.transpose(1, 2), Y))
    Yb_t = torch.tensor(Yb, device=DEVICE)
    pred = torch.bmm(B, W) + ym.unsqueeze(0)
    ss_res = ((pred - Yb_t.unsqueeze(0)) ** 2).sum(1)
    ss_tot = ((Yb_t - Yb_t.mean(0, keepdim=True)) ** 2).sum(0).unsqueeze(0)
    return (1 - ss_res / ss_tot).cpu().numpy()


def output_dft(L9, ys):
    """Mean centered logit profile over Delta=(r-y)%9, then rfft energy."""
    prof = np.zeros(9)
    cen = L9 - L9.mean(axis=1, keepdims=True)
    for d in range(9):
        prof[d] = np.mean(cen[np.arange(len(ys)), (ys + d) % 9])
    F = np.fft.rfft(prof)               # freqs 0..4
    e = np.abs(F) ** 2
    tot = e[1:].sum() + 1e-12
    return {"profile": prof.round(4).tolist(),
            "coset_frac": float(e[3] / tot),
            "fine_frac": float((e[1] + e[2] + e[4]) / tot),
            "energy": e.round(5).tolist()}


def main():
    tok = AutoTokenizer.from_pretrained(TOK_BASE)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    digit_ids = torch.tensor([tok.encode(str(d))[0] for d in range(9)], device=DEVICE)

    fit_rows = read_jsonl(TRAIN_FILE, N_FIT)
    sc_rows = read_jsonl(EVAL_FILE, N_SCORE)
    fit_p = [r["prompt"] for r in fit_rows]
    sc_p = [r["prompt"] for r in sc_rows]
    fit_piles = [int(PILE_RE.search(p).group(1)) for p in fit_p]
    sc_piles = [int(PILE_RE.search(p).group(1)) for p in sc_p]
    Ya = targets_for(fit_piles)
    Yb = targets_for(sc_piles)
    ys = np.array([p % 9 for p in sc_piles])
    gold_ids = np.array([tok.encode(str(p % MOD))[0] for p in sc_piles])

    results = []
    for step in range(SMIN, SMAX + 1, STRIDE):
        rev = f"step-{step}"
        print(f"\n=== step {step} ===", flush=True)
        try:
            model = AutoModelForCausalLM.from_pretrained(REPO, revision=rev).to(DEVICE).eval()
        except (OSError, ValueError) as e:
            print(f"  SKIP step {step}: revision not available ({type(e).__name__})", flush=True)
            continue
        n_layers = model.config.num_hidden_layers
        hidden = model.config.hidden_size

        Xa, _ = forward_collect(model, tok, fit_p, n_layers, hidden, digit_ids)
        Xb, L9 = forward_collect(model, tok, sc_p, n_layers, hidden, digit_ids)

        # behavior (argmax over full vocab at final position == gold token)
        # cheap proxy via digit logits argmax vs true residue (MOD==9 sweeps):
        beh = float(np.mean(np.argmax(L9, axis=1) == ys)) if MOD == 9 else None

        dft = output_dft(L9, ys)
        r2 = ridge_r2_all_layers(Xa, Ya, Xb, Yb)     # [L, 12]
        per_period = {}
        for i, (coord, T) in enumerate(PERIODS):
            pair = r2[:, 2 * i:2 * i + 2].mean(axis=1)
            per_period[f"{coord}{T}"] = [round(float(v), 4) for v in pair]

        results.append({"step": step, "behavior_mod9_argmax": beh,
                        "dft": dft, "period_r2": per_period})
        print(f"  coset_frac={dft['coset_frac']:.3f} fine_frac={dft['fine_frac']:.3f} "
              f"ds3@L14={per_period['ds3'][14]:.3f} ds9@L14={per_period['ds9'][14]:.3f} "
              f"N10@L14={per_period['N10'][14]:.3f}", flush=True)

        del model
        torch.cuda.empty_cache()
        with open(OUT, "w") as f:
            for r in results:
                f.write(json.dumps(r) + "\n")
    print(f"\nWrote {len(results)} rows -> {OUT}")


if __name__ == "__main__":
    main()
