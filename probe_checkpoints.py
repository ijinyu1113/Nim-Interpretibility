"""Cross-checkpoint mechanistic sweep: probes + behavioral coset agreement +
per-layer logit-lens coset structure, in one pass per checkpoint.

For every dense checkpoint (branches step-250..step-6000):
  A) BEHAVIOR: restricted-argmax prediction over the answer-digit tokens at the
     final position -> exact acc + agreement (pred == true mod d) for every
     divisor d of the task modulus. The plateau should read as: exact = 1/3,
     mod-3 agreement ~ 1.0 (the behavioral definition of the coset heuristic).
  B) LOGIT LENS: at EVERY layer, project the final-position residual through
     final_ln + unembedding, restrict to answer digits, and compute the same
     coset agreements -> at which DEPTH does each coset level become correct.
  C) PROBES: linear decodability of N mod {2,3,8,9} from the residual stream
     at two positions (final token, last numeral token), fit on train-split
     piles, scored on held-out piles.

PRE-REGISTERED PREDICTIONS
  P1  probe mod-3 decodability rises at PLATEAU ONSET (~2000-2750),
      BEFORE task accuracy leaves 1/3.
  P2  probe mod-9 decodability rises only at the SNAP (~3500-3750).
  P3  probe mod-8 stays near chance throughout in the mod-9 model.
  P4  mirror image in the mr=7 model (mod-8 by ~250-500; mod-3 never).
  P5  base model: all targets ~chance at the final token.
  P6  behavioral mod-3 agreement ~1.0 across the plateau (coset behavior).
  P7  probe mod-3 (representation) LEADS behavioral mod-3 (readout) â the
      rep->behavior lag is measurable and positive.
  P8  in the mr=7 model, logit-lens correctness is DEPTH-ORDERED:
      mod-2 correct at earlier layers than mod-4 than mod-8.

Usage:
  python probe_checkpoints.py <hf_repo> <mr> <train_jsonl> <eval_jsonl> <out_jsonl> [step_min step_max step_stride]
Pass repo "BASE" to probe EleutherAI/pythia-410m-deduped (step recorded as 0).
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

assert torch.cuda.is_available(), (
    "CUDA not available - refusing to crawl on CPU. Check the job GPU allocation.")
DEVICE = "cuda"
BATCH = 64
MAX_LEN = 128
N_FIT = 2000
N_SCORE = 2000
PROBE_TARGETS = [2, 3, 8, 9]
TASK_MOD = MR + 1
PILE_RE = re.compile(r"There are (\d+) coins")


def divisors(m):
    return [d for d in range(2, m) if m % d == 0]


def read_jsonl(p, n):
    rows = []
    with open(p) as f:
        for ln in f:
            rows.append(json.loads(ln))
            if len(rows) >= n:
                break
    return rows


def positions_of_interest(tok, prompts):
    """(final_token_idx, last_numeral_token_idx) per prompt. The pile numeral
    ends right before the LAST ' coins' token in the nimsimple template."""
    finals, numerals = [], []
    for p in prompts:
        ids = tok.encode(p, truncation=True, max_length=MAX_LEN)
        toks = [tok.decode([t]) for t in ids]
        finals.append(len(ids) - 1)
        coin_positions = [i for i, t in enumerate(toks) if t == " coins"]
        numerals.append(coin_positions[-1] - 1 if coin_positions else len(ids) - 1)
    return finals, numerals


@torch.no_grad()
def extract(model, tok, prompts, finals, numerals, n_layers, hidden, digit_ids):
    """Returns:
      Xf, Xn        [N, n_layers+1, hidden]  hidden states (final / numeral pos)
      lens_pred     [N, n_layers+1]          logit-lens restricted argmax per layer
      final_pred    [N]                      model-output restricted argmax
    """
    N = len(prompts)
    Xf = np.zeros((N, n_layers + 1, hidden), dtype=np.float32)
    Xn = np.zeros((N, n_layers + 1, hidden), dtype=np.float32)
    lens_pred = np.zeros((N, n_layers + 1), dtype=np.int64)
    final_pred = np.zeros(N, dtype=np.int64)
    ln_f = model.gpt_neox.final_layer_norm
    W_U = model.embed_out.weight[digit_ids]          # [mod, hidden]
    digit_ids_t = torch.tensor(digit_ids, device=DEVICE)

    for s in range(0, N, BATCH):
        batch = prompts[s:s + BATCH]
        enc = tok(batch, return_tensors="pt", padding=True,
                  truncation=True, max_length=MAX_LEN).to(DEVICE)
        out = model(**enc, output_hidden_states=True)
        idx = torch.arange(len(batch), device=DEVICE)
        fpos = torch.tensor(finals[s:s + len(batch)], device=DEVICE)
        # model-output prediction restricted to digit tokens
        final_logits = out.logits[idx, fpos][:, digit_ids_t]
        final_pred[s:s + len(batch)] = final_logits.argmax(-1).cpu().numpy()
        npos = torch.tensor(numerals[s:s + len(batch)], device=DEVICE)
        for li, h in enumerate(out.hidden_states):
            hf = h[idx, fpos]                         # [B, hidden] final position
            # logit lens: final_ln then restricted unembed
            lens_logits = ln_f(hf) @ W_U.T            # [B, mod]
            lens_pred[s:s + len(batch), li] = lens_logits.argmax(-1).cpu().numpy()
            Xf[s:s + len(batch), li] = hf.float().cpu().numpy()
            Xn[s:s + len(batch), li] = h[idx, npos].float().cpu().numpy()
        if (s // BATCH) % 10 == 0:
            print(f"    extract {s + len(batch)}/{N}", flush=True)
    return Xf, Xn, lens_pred, final_pred


def fit_probes_all_layers(Xa, ya, Xb, yb, n_classes, steps=300, lr=0.05):
    """Train one linear probe PER LAYER simultaneously on GPU.
    Xa: [N, L, H] fit activations; Xb: score activations. Returns per-layer
    accuracy list. Standardizes features per layer using fit-set stats."""
    A = torch.tensor(np.ascontiguousarray(np.transpose(Xa, (1, 0, 2))), device=DEVICE)  # [L,N,H]
    B = torch.tensor(np.ascontiguousarray(np.transpose(Xb, (1, 0, 2))), device=DEVICE)
    mu = A.mean(dim=1, keepdim=True)
    sd = A.std(dim=1, keepdim=True).clamp_min(1e-4)
    A = (A - mu) / sd
    B = (B - mu) / sd
    L, N, H = A.shape
    ya_t = torch.tensor(ya, device=DEVICE, dtype=torch.long).unsqueeze(0).expand(L, -1)
    W = torch.zeros(L, H, n_classes, device=DEVICE, requires_grad=True)
    bias = torch.zeros(L, 1, n_classes, device=DEVICE, requires_grad=True)
    opt = torch.optim.Adam([W, bias], lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        logits = torch.bmm(A, W) + bias                     # [L,N,C]
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(L * N, n_classes), ya_t.reshape(L * N))
        loss.backward()
        opt.step()
    with torch.no_grad():
        pred = (torch.bmm(B, W) + bias).argmax(-1).cpu().numpy()  # [L,Nb]
    return [float((pred[li] == yb).mean()) for li in range(L)]


def agreements(pred_residues, true_residues, m):
    out = {"exact": float((pred_residues == true_residues).mean())}
    for d in divisors(m):
        out[f"agree_mod{d}"] = float(((pred_residues % d) == (true_residues % d)).mean())
    return out


def main():
    base_mode = REPO == "BASE"
    import os
    _local = os.path.expanduser("~/pythia410m_local")
    base_src = _local if os.path.isdir(_local) else "EleutherAI/pythia-410m-deduped"
    model_id = base_src if base_mode else REPO
    # Tokenizer always from the base model: the fine-tune repos have an EMPTY
    # main branch (checkpoints live only on step-N branches), and the tokenizer
    # is identical across all checkpoints anyway.
    tok = AutoTokenizer.from_pretrained("EleutherAI/pythia-410m-deduped")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    # answer digits are bare tokens '0'..'{mod-1}' (nimsimple convention)
    digit_ids = [tok.encode(str(d), add_special_tokens=False)[0] for d in range(TASK_MOD)]
    assert len(set(digit_ids)) == TASK_MOD

    fit_rows = read_jsonl(TRAIN_FILE, N_FIT)
    score_rows = read_jsonl(EVAL_FILE, N_SCORE)
    fit_prompts = [r["prompt"] for r in fit_rows]
    score_prompts = [r["prompt"] for r in score_rows]
    fit_piles = np.array([int(PILE_RE.search(p).group(1)) for p in fit_prompts])
    score_piles = np.array([int(PILE_RE.search(p).group(1)) for p in score_prompts])
    score_true = score_piles % TASK_MOD
    f_fit, n_fit = positions_of_interest(tok, fit_prompts)
    f_sc, n_sc = positions_of_interest(tok, score_prompts)
    print(f"fit={len(fit_prompts)} score={len(score_prompts)} task_mod={TASK_MOD}")

    steps = [0] if base_mode else list(range(STEP_MIN, STEP_MAX + 1, STEP_STRIDE))
    results = []
    for step in steps:
        rev = None if base_mode else f"step-{step}"
        print(f"\n=== step {step} ({model_id}@{rev}) ===", flush=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_id, revision=rev).to(DEVICE).eval()
        n_layers = model.config.num_hidden_layers
        hidden = model.config.hidden_size

        Xf_fit, Xn_fit, _, _ = extract(
            model, tok, fit_prompts, f_fit, n_fit, n_layers, hidden, digit_ids)
        Xf_sc, Xn_sc, lens_pred, final_pred = extract(
            model, tok, score_prompts, f_sc, n_sc, n_layers, hidden, digit_ids)
        del model
        torch.cuda.empty_cache()

        # A) BEHAVIOR: restricted-argmax output vs truth, coset agreements
        beh = agreements(final_pred, score_true, TASK_MOD)
        results.append({"kind": "behavior", "step": step, **beh})
        print(f"  behavior: exact={beh['exact']:.3f} " +
              " ".join(f"mod{d}={beh[f'agree_mod{d}']:.3f}" for d in divisors(TASK_MOD)),
              flush=True)

        # B) LOGIT LENS per layer
        for li in range(n_layers + 1):
            lens = agreements(lens_pred[:, li], score_true, TASK_MOD)
            results.append({"kind": "lens", "step": step, "layer": li, **lens})

        # C) PROBES
        for pos_name, Xa, Xb in [("final", Xf_fit, Xf_sc), ("numeral", Xn_fit, Xn_sc)]:
            for m in PROBE_TARGETS:
                per_layer = fit_probes_all_layers(
                    Xa, fit_piles % m, Xb, score_piles % m, m)
                bl = int(np.argmax(per_layer))
                results.append({
                    "kind": "probe", "step": step, "position": pos_name,
                    "target_mod": m, "chance": 1.0 / m,
                    "best_layer": bl, "best_acc": per_layer[bl],
                    "per_layer": [round(a, 4) for a in per_layer],
                })
                print(f"  probe {pos_name:>7} mod{m}: best={per_layer[bl]:.3f} "
                      f"@L{bl} (chance {1/m:.3f})", flush=True)

        with open(OUT, "w") as f:      # rewrite each step: crash-safe progress
            for r in results:
                f.write(json.dumps(r) + "\n")
    print(f"\nWrote {len(results)} rows -> {OUT}")


if __name__ == "__main__":
    main()
