"""E1 + E2: digit-sum pathway probe and attention-globalization sweep.

Per dense checkpoint:
  E1  RIDGE-REGRESS the RAW DIGIT SUM (e.g. 49059 -> 27) from hidden states at
      (final token, last numeral token), every layer. R2 on held-out piles.
      The mod-3/mod-9 residues were probed before; this tests the MECHANISM:
      is the actual sum encoded as an intermediate quantity?
      PRE-REGISTERED: in the mod-9 model, digit-sum R2 rises at PLATEAU ONSET
      (with mod-3), confirming the digit-sum pathway. If mod-3 rises without
      raw-sum R2, the model computes residues by another route (also a result).
  E2  ATTENTION: for layers 6..16, all heads, mean attention weight from the
      final position onto (a) the LAST numeral chunk token, (b) EARLIER numeral
      chunk tokens. PRE-REGISTERED: mod-8 model concentrates on (a) from the
      start; mod-9 model's (b) mass jumps at plateau onset (globalization).

Usage:
  python probe_digitsum_attn.py <hf_repo> <mr> <train_jsonl> <eval_jsonl> <out_jsonl> [step_min step_max stride]
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
STEP_MIN = int(sys.argv[6]) if len(sys.argv) > 6 else 250
STEP_MAX = int(sys.argv[7]) if len(sys.argv) > 7 else 6000
STEP_STRIDE = int(sys.argv[8]) if len(sys.argv) > 8 else 250

assert torch.cuda.is_available(), "CUDA not available - refusing CPU crawl."
DEVICE = "cuda"
BATCH = 64
MAX_LEN = 128
N_FIT = 2000
N_SCORE = 2000
N_ATTN = 512
ATTN_LAYERS = list(range(6, 17))
PILE_RE = re.compile(r"There are (\d+) coins")


def read_jsonl(p, n):
    rows = []
    with open(p) as f:
        for ln in f:
            rows.append(json.loads(ln))
            if len(rows) >= n:
                break
    return rows


def numeral_span(tok, prompt):
    """(start, end_exclusive, final_idx) token indices of the pile numeral:
    tokens strictly between the ' are' before the LAST ' coins' and that ' coins'."""
    ids = tok.encode(prompt, truncation=True, max_length=MAX_LEN)
    toks = [tok.decode([t]) for t in ids]
    coins = [i for i, t in enumerate(toks) if t == " coins"]
    c = coins[-1]
    a = max(i for i, t in enumerate(toks[:c]) if t == " are")
    return a + 1, c, len(ids) - 1


def digit_sum(n):
    return sum(int(c) for c in str(n))


def fit_ridge_all_layers(Xa, ya, Xb, yb, lam=10.0):
    """Closed-form ridge per layer, batched on GPU. Returns per-layer R2."""
    A = torch.tensor(np.ascontiguousarray(np.transpose(Xa, (1, 0, 2))), device=DEVICE)
    B = torch.tensor(np.ascontiguousarray(np.transpose(Xb, (1, 0, 2))), device=DEVICE)
    mu = A.mean(1, keepdim=True); sd = A.std(1, keepdim=True).clamp_min(1e-4)
    A = (A - mu) / sd; B = (B - mu) / sd
    L, N, H = A.shape
    ya_t = torch.tensor(ya, device=DEVICE, dtype=torch.float32)
    ym = ya_t.mean()
    Y = (ya_t - ym).unsqueeze(0).unsqueeze(-1).expand(L, -1, -1)   # [L,N,1]
    G = torch.bmm(A.transpose(1, 2), A) + lam * torch.eye(H, device=DEVICE).unsqueeze(0)
    W = torch.linalg.solve(G, torch.bmm(A.transpose(1, 2), Y))      # [L,H,1]
    yb_t = torch.tensor(yb, device=DEVICE, dtype=torch.float32)
    pred = torch.bmm(B, W).squeeze(-1) + ym                         # [L,Nb]
    ss_res = ((pred - yb_t) ** 2).sum(1)
    ss_tot = ((yb_t - yb_t.mean()) ** 2).sum()
    r2 = (1 - ss_res / ss_tot).cpu().numpy()
    return [float(x) for x in r2]


@torch.no_grad()
def extract_hidden(model, tok, prompts, positions, n_layers, hidden):
    """positions: list of (pos_final, pos_numeral_last). Returns Xf, Xn."""
    N = len(prompts)
    Xf = np.zeros((N, n_layers + 1, hidden), dtype=np.float32)
    Xn = np.zeros((N, n_layers + 1, hidden), dtype=np.float32)
    for s in range(0, N, BATCH):
        batch = prompts[s:s + BATCH]
        enc = tok(batch, return_tensors="pt", padding=True,
                  truncation=True, max_length=MAX_LEN).to(DEVICE)
        out = model(**enc, output_hidden_states=True)
        idx = torch.arange(len(batch), device=DEVICE)
        fpos = torch.tensor([positions[s + i][0] for i in range(len(batch))], device=DEVICE)
        npos = torch.tensor([positions[s + i][1] for i in range(len(batch))], device=DEVICE)
        for li, h in enumerate(out.hidden_states):
            Xf[s:s + len(batch), li] = h[idx, fpos].float().cpu().numpy()
            Xn[s:s + len(batch), li] = h[idx, npos].float().cpu().numpy()
        if (s // BATCH) % 10 == 0:
            print(f"    hidden {s + len(batch)}/{N}", flush=True)
    return Xf, Xn


@torch.no_grad()
def attention_metrics(model, tok, prompts, spans):
    """Mean attention from the FINAL position onto (last numeral chunk) and
    (earlier numeral chunks), per layer in ATTN_LAYERS, per head."""
    sums_last = None
    sums_early = None
    n_done = 0
    for s in range(0, len(prompts), BATCH):
        batch = prompts[s:s + BATCH]
        enc = tok(batch, return_tensors="pt", padding=True,
                  truncation=True, max_length=MAX_LEN).to(DEVICE)
        out = model(**enc, output_attentions=True)
        if sums_last is None:
            n_heads = out.attentions[0].shape[1]
            sums_last = np.zeros((len(ATTN_LAYERS), n_heads))
            sums_early = np.zeros((len(ATTN_LAYERS), n_heads))
        for bi in range(len(batch)):
            a0, c, f = spans[s + bi]
            for li_out, li in enumerate(ATTN_LAYERS):
                att = out.attentions[li][bi, :, f, :]     # [heads, seq]
                sums_last[li_out] += att[:, c - 1].cpu().numpy()
                if c - 1 > a0:
                    sums_early[li_out] += att[:, a0:c - 1].sum(-1).cpu().numpy()
        n_done += len(batch)
    return sums_last / n_done, sums_early / n_done


def main():
    tok = AutoTokenizer.from_pretrained("EleutherAI/pythia-410m-deduped")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    fit_rows = read_jsonl(TRAIN_FILE, N_FIT)
    score_rows = read_jsonl(EVAL_FILE, N_SCORE)
    fit_prompts = [r["prompt"] for r in fit_rows]
    score_prompts = [r["prompt"] for r in score_rows]
    fit_piles = [int(PILE_RE.search(p).group(1)) for p in fit_prompts]
    score_piles = [int(PILE_RE.search(p).group(1)) for p in score_prompts]
    fit_ds = np.array([digit_sum(n) for n in fit_piles], dtype=np.float32)
    score_ds = np.array([digit_sum(n) for n in score_piles], dtype=np.float32)

    fit_spans = [numeral_span(tok, p) for p in fit_prompts]
    score_spans = [numeral_span(tok, p) for p in score_prompts]
    fit_pos = [(sp[2], sp[1] - 1) for sp in fit_spans]      # (final, last numeral tok)
    score_pos = [(sp[2], sp[1] - 1) for sp in score_spans]
    attn_prompts = score_prompts[:N_ATTN]
    attn_spans = [(a, c, f) for (a, c, f) in score_spans[:N_ATTN]]
    print(f"fit={len(fit_prompts)} score={len(score_prompts)} attn={len(attn_prompts)}")

    results = []
    for step in range(STEP_MIN, STEP_MAX + 1, STEP_STRIDE):
        rev = f"step-{step}"
        print(f"\n=== step {step} ===", flush=True)
        model = AutoModelForCausalLM.from_pretrained(REPO, revision=rev).to(DEVICE).eval()
        n_layers = model.config.num_hidden_layers
        hidden = model.config.hidden_size

        Xf_a, Xn_a = extract_hidden(model, tok, fit_prompts, fit_pos, n_layers, hidden)
        Xf_b, Xn_b = extract_hidden(model, tok, score_prompts, score_pos, n_layers, hidden)
        for pos_name, Xa, Xb in [("final", Xf_a, Xf_b), ("numeral", Xn_a, Xn_b)]:
            r2 = fit_ridge_all_layers(Xa, fit_ds, Xb, score_ds)
            bl = int(np.argmax(r2))
            results.append({"kind": "digitsum", "step": step, "position": pos_name,
                            "best_layer": bl, "best_r2": r2[bl],
                            "per_layer": [round(x, 4) for x in r2]})
            print(f"  digitsum {pos_name:>7}: best R2={r2[bl]:.3f} @L{bl}", flush=True)

        att_last, att_early = attention_metrics(model, tok, attn_prompts, attn_spans)
        gi = float(att_early.max())        # globalization index
        gl, gh = np.unravel_index(att_early.argmax(), att_early.shape)
        results.append({"kind": "attention", "step": step,
                        "globalization_index": gi,
                        "top_head": [ATTN_LAYERS[int(gl)], int(gh)],
                        "att_last_max": float(att_last.max()),
                        "att_last": att_last.round(4).tolist(),
                        "att_early": att_early.round(4).tolist(),
                        "layers": ATTN_LAYERS})
        print(f"  attention: globalization={gi:.3f} (top head L{ATTN_LAYERS[int(gl)]}.H{int(gh)}) "
              f"last-chunk max={att_last.max():.3f}", flush=True)

        del model
        torch.cuda.empty_cache()
        with open(OUT, "w") as f:
            for r in results:
                f.write(json.dumps(r) + "\n")
    print(f"\nWrote {len(results)} rows -> {OUT}")


if __name__ == "__main__":
    main()
