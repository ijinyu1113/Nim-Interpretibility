"""Phase tracing: the two tests that upgrade the mechanism model to measurement.

MODE=prefix (the rotation-accumulation trace):
  At each numeral token position j, decode the PREFIX digit-sum phase
  cos/sin(2*pi*prefix_ds_j/9) from hidden states at that position (all layers),
  and also the FULL-sum phase at the same position.
  PRE-REGISTERED signature of per-digit rotation accumulation (converged mod-9):
    prefix phase decodable at every numeral position; full-sum phase decodable
    ONLY where prefix == full (the last numeral token). Controls: mod-8 model
    (no ds-phase structure anywhere); early checkpoints (nothing yet).

MODE=switch (the mechanism-switch test: count-then-wrap vs wrap-as-you-go):
  Dense-checkpoint version of the prefix trace with two additions per numeral
  position j: (a) a third target, the integer PREFIX COUNT prefix_ds_j (does
  this position hold the running count, not just its phase?); (b) rotation
  increments: decode theta_hat at consecutive positions j, j+1 (best layer of
  13-16) on the same held-out operands and compare Delta-theta against the
  prediction 2*pi*(token j+1's digit sum)/9 -- the transition rule itself.
  PRE-REGISTERED (H-count vs H-rotate, written before the run):
    H-rotate (wrap-as-you-go learned directly): cross-token prefix-PHASE R2
      rises in the snap window (3250-3500), co-timed with final-position ds9
      arrival (F1); no epoch where cross-token prefix-COUNT is decodable while
      the phase is absent; increments match the x40-deg rule wherever phase
      R2 > 0.
    H-count (count-then-wrap intermediate stage): a window during plateau/
      early snap with cross-token prefix-COUNT present and prefix-PHASE
      absent; phase appears at the FINAL position before intermediate ones;
      then count fades while phase persists -- and the switch window should
      coincide with the deep-layer raw-ds fade (E1, steps ~2000-3500).
    KILL for H-count: no epoch where count leads phase at cross-token
      positions. Confound note: pos0 is single-token (count = f(token id),
      the mod-8 control decodes it) -- only cross-token positions bear on
      the hypotheses; pos0 is recorded as the confound row.

MODE=noise (the wedge account, quantified):
  Per checkpoint: fit the ds-period-9 and ds-period-3 planes at the final
  position (layers 13-16, best held-out layer), read the phase via atan2 on
  held-out operands, report the angular-error distribution: median |err| and
  the fractions within the class half-widths (20 deg for period-9's 40-deg
  wedges; 60 deg for period-3's 120-deg wedges).
  PRE-REGISTERED: frac(|err9| < 20 deg) crosses ~0.9 at the snap (~3500);
  frac(|err3| < 60 deg) crosses high at plateau onset (~2250); i.e. the noise
  trajectory crosses each wedge threshold at the corresponding behavioral event.

Usage:
  python phase_trace.py <mode> <repo> <mr> <train_jsonl> <eval_jsonl> <out_jsonl> <steps>
  steps: "250:6000:250" (range) or "1000,2750,6000" (list)
"""
import json
import re
import sys

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODE = sys.argv[1]
REPO = sys.argv[2]
MR = int(sys.argv[3])
TRAIN_FILE = sys.argv[4]
EVAL_FILE = sys.argv[5]
OUT = sys.argv[6]
STEPS_SPEC = sys.argv[7]

assert torch.cuda.is_available(), "CUDA not available - refusing CPU crawl."
DEVICE = "cuda"
BATCH = 64
MAX_LEN = 128
N_FIT = 2000
N_SCORE = 2000
PILE_RE = re.compile(r"There are (\d+) coins")


def parse_steps(spec):
    if ":" in spec:
        a, b, s = (int(x) for x in spec.split(":"))
        return list(range(a, b + 1, s))
    return [int(x) for x in spec.split(",")]


def read_jsonl(p, n):
    rows = []
    with open(p) as f:
        for ln in f:
            rows.append(json.loads(ln))
            if len(rows) >= n:
                break
    return rows


def numeral_info(tok, prompt):
    """Token index range [a, c) of the numeral, final index, and per-token
    prefix digit sums."""
    ids = tok.encode(prompt, truncation=True, max_length=MAX_LEN)
    toks = [tok.decode([t]) for t in ids]
    coins = [i for i, t in enumerate(toks) if t == " coins"]
    c = coins[-1]
    a = max(i for i, t in enumerate(toks[:c]) if t == " are") + 1
    prefix = []
    run = 0
    for i in range(a, c):
        run += sum(int(ch) for ch in toks[i] if ch.isdigit())
        prefix.append(run)
    return a, c, len(ids) - 1, prefix


def ridge_r2(Xa, Ya, Xb, Yb, lam=10.0):
    """Multi-target ridge per layer (batched). X: [N, L, H]; Y: [N, T].
    Returns [L, T] held-out R2 and the per-layer weight matrices."""
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
    pred = torch.bmm(B, W) + ym.unsqueeze(0)
    Yb_t = torch.tensor(Yb, device=DEVICE)
    ss_res = ((pred - Yb_t.unsqueeze(0)) ** 2).sum(1)
    ss_tot = ((Yb_t - Yb_t.mean(0, keepdim=True)) ** 2).sum(0).unsqueeze(0)
    return (1 - ss_res / ss_tot).cpu().numpy(), pred.cpu().numpy()


@torch.no_grad()
def collect(model, tok, prompts, pos_lists, n_layers, hidden):
    """Hidden states at the given per-example position (single int each)."""
    N = len(prompts)
    X = np.zeros((N, n_layers + 1, hidden), dtype=np.float32)
    for s in range(0, N, BATCH):
        batch = prompts[s:s + BATCH]
        enc = tok(batch, return_tensors="pt", padding=True,
                  truncation=True, max_length=MAX_LEN).to(DEVICE)
        out = model(**enc, output_hidden_states=True)
        idx = torch.arange(len(batch), device=DEVICE)
        pos = torch.tensor(pos_lists[s:s + BATCH], device=DEVICE)
        for li, h in enumerate(out.hidden_states):
            X[s:s + len(batch), li] = h[idx, pos].float().cpu().numpy()
    return X


def phases(ds, T):
    th = 2 * np.pi * (np.asarray(ds, dtype=np.float64) % T) / T
    return np.stack([np.cos(th), np.sin(th)], axis=1).astype(np.float32)


def main():
    tok = AutoTokenizer.from_pretrained("EleutherAI/pythia-410m-deduped")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    fit_rows = read_jsonl(TRAIN_FILE, N_FIT)
    sc_rows = read_jsonl(EVAL_FILE, N_SCORE)
    fit_p = [r["prompt"] for r in fit_rows]
    sc_p = [r["prompt"] for r in sc_rows]
    fit_info = [numeral_info(tok, p) for p in fit_p]
    sc_info = [numeral_info(tok, p) for p in sc_p]
    results = []

    for step in parse_steps(STEPS_SPEC):
        rev = f"step-{step}"
        print(f"\n=== step {step} ===", flush=True)
        model = AutoModelForCausalLM.from_pretrained(REPO, revision=rev).to(DEVICE).eval()
        nl, hid = model.config.num_hidden_layers, model.config.hidden_size
        row = {"step": step, "mode": MODE}

        if MODE == "prefix":
            maxJ = max(len(i[3]) for i in fit_info)
            per_pos = {}
            for j in range(maxJ):
                fi = [k for k, i in enumerate(fit_info) if len(i[3]) > j]
                si = [k for k, i in enumerate(sc_info) if len(i[3]) > j]
                if len(fi) < 400 or len(si) < 400:
                    continue
                fpos = [fit_info[k][0] + j for k in fi]
                spos = [sc_info[k][0] + j for k in si]
                Xa = collect(model, tok, [fit_p[k] for k in fi], fpos, nl, hid)
                Xb = collect(model, tok, [sc_p[k] for k in si], spos, nl, hid)
                Ya = np.concatenate([phases([fit_info[k][3][j] for k in fi], 9),
                                     phases([fit_info[k][3][-1] for k in fi], 9)], axis=1)
                Yb = np.concatenate([phases([sc_info[k][3][j] for k in si], 9),
                                     phases([sc_info[k][3][-1] for k in si], 9)], axis=1)
                r2, _ = ridge_r2(Xa, Ya, Xb, Yb)
                pre = r2[:, :2].mean(axis=1)   # prefix-phase R2 per layer
                ful = r2[:, 2:].mean(axis=1)   # full-sum phase R2 per layer
                per_pos[f"pos{j}"] = {
                    "n": len(si),
                    "prefix_r2": [round(float(v), 3) for v in pre],
                    "full_r2": [round(float(v), 3) for v in ful],
                    "prefix_best": round(float(pre.max()), 3),
                    "full_best": round(float(ful.max()), 3),
                }
                print(f"  pos{j} (n={len(si)}): prefix best R2={pre.max():.3f} "
                      f"(L{int(pre.argmax())})  full best R2={ful.max():.3f}", flush=True)
            row["per_pos"] = per_pos

        elif MODE == "switch":
            maxJ = max(len(i[3]) for i in fit_info)
            per_pos = {}
            sc_pred = {}   # j -> (si, theta_hat[best L13-16], prefix_r2@best, best)
            for j in range(maxJ):
                fi = [k for k, i in enumerate(fit_info) if len(i[3]) > j]
                si = [k for k, i in enumerate(sc_info) if len(i[3]) > j]
                if len(fi) < 400 or len(si) < 400:
                    continue
                fpos = [fit_info[k][0] + j for k in fi]
                spos = [sc_info[k][0] + j for k in si]
                Xa = collect(model, tok, [fit_p[k] for k in fi], fpos, nl, hid)
                Xb = collect(model, tok, [sc_p[k] for k in si], spos, nl, hid)
                cntA = np.asarray([fit_info[k][3][j] for k in fi], dtype=np.float32)[:, None]
                cntB = np.asarray([sc_info[k][3][j] for k in si], dtype=np.float32)[:, None]
                Ya = np.concatenate([phases([fit_info[k][3][j] for k in fi], 9),
                                     phases([fit_info[k][3][-1] for k in fi], 9),
                                     cntA], axis=1)
                Yb = np.concatenate([phases([sc_info[k][3][j] for k in si], 9),
                                     phases([sc_info[k][3][-1] for k in si], 9),
                                     cntB], axis=1)
                r2, pred = ridge_r2(Xa, Ya, Xb, Yb)
                pre = r2[:, :2].mean(axis=1)
                ful = r2[:, 2:4].mean(axis=1)
                cnt = r2[:, 4]
                best = int(np.argmax(pre[13:17])) + 13
                th_hat = np.arctan2(pred[best, :, 1], pred[best, :, 0])
                sc_pred[j] = (si, th_hat, float(pre[best]), best)
                per_pos[f"pos{j}"] = {
                    "n": len(si),
                    "prefix_phase_best": round(float(pre.max()), 3),
                    "prefix_phase_L1316": round(float(pre[13:17].max()), 3),
                    "full_phase_best": round(float(ful.max()), 3),
                    "count_best": round(float(cnt.max()), 3),
                    "count_best_layer": int(cnt.argmax()),
                    "count_L1316": round(float(cnt[13:17].max()), 3),
                    "prefix_r2": [round(float(v), 3) for v in pre],
                    "count_r2": [round(float(v), 3) for v in cnt],
                }
                print(f"  pos{j} (n={len(si)}): phase L13-16 R2={pre[13:17].max():.3f} "
                      f"count best R2={cnt.max():.3f} (L{int(cnt.argmax())}) "
                      f"full R2={ful.max():.3f}", flush=True)
            inc = {}
            for j in range(maxJ - 1):
                if j not in sc_pred or (j + 1) not in sc_pred:
                    continue
                si0, th0, r20, _ = sc_pred[j]
                si1, th1, r21, _ = sc_pred[j + 1]
                pos0 = {k: p for p, k in enumerate(si0)}
                rows = [(pos0[k], p) for p, k in enumerate(si1) if k in pos0]
                d_hat = np.array([th1[p1] for _, p1 in rows]) - \
                        np.array([th0[p0] for p0, _ in rows])
                tok_ds = np.array([sc_info[si1[p1]][3][j + 1] - sc_info[si1[p1]][3][j]
                                   for _, p1 in rows], dtype=np.float64)
                d_true = 2 * np.pi * tok_ds / 9
                err = np.angle(np.exp(1j * (d_hat - d_true)))
                deg = np.abs(err) * 180 / np.pi
                inc[f"d{j}{j + 1}"] = {
                    "n": len(rows),
                    "r2_pair": [round(r20, 3), round(r21, 3)],
                    "median_err_deg": round(float(np.median(deg)), 1),
                    "frac_within_20": round(float(np.mean(deg < 20.0)), 3),
                }
                print(f"  inc d{j}{j + 1} (n={len(rows)}): median|err|="
                      f"{np.median(deg):.0f}deg frac<20deg={np.mean(deg < 20.0):.3f} "
                      f"(phase R2 pair {r20:.2f}/{r21:.2f})", flush=True)
            row["per_pos"] = per_pos
            row["increments"] = inc

        elif MODE == "noise":
            ffin = [i[2] for i in fit_info]
            sfin = [i[2] for i in sc_info]
            Xa = collect(model, tok, fit_p, ffin, nl, hid)
            Xb = collect(model, tok, sc_p, sfin, nl, hid)
            fds = [i[3][-1] for i in fit_info]
            sds = [i[3][-1] for i in sc_info]
            out = {}
            for T, halfwidth in ((9, 20.0), (3, 60.0)):
                Ya, Yb = phases(fds, T), phases(sds, T)
                r2, pred = ridge_r2(Xa, Ya, Xb, Yb)
                pair = r2.mean(axis=1)
                best = int(np.argmax(pair[13:17])) + 13
                th_hat = np.arctan2(pred[best, :, 1], pred[best, :, 0])
                th_true = 2 * np.pi * (np.asarray(sds) % T) / T
                err = np.angle(np.exp(1j * (th_hat - th_true)))
                deg = np.abs(err) * 180 / np.pi
                out[f"T{T}"] = {
                    "best_layer": best, "r2": round(float(pair[best]), 3),
                    "median_err_deg": round(float(np.median(deg)), 1),
                    "frac_within_half": round(float(np.mean(deg < halfwidth)), 3),
                }
                print(f"  T={T}: L{best} R2={pair[best]:.3f} "
                      f"median|err|={np.median(deg):.0f}deg "
                      f"frac<{halfwidth:.0f}deg={np.mean(deg < halfwidth):.3f}", flush=True)
            row["noise"] = out

        results.append(row)
        del model
        torch.cuda.empty_cache()
        with open(OUT, "w") as f:
            for r in results:
                f.write(json.dumps(r) + "\n")
    print(f"\nWrote {len(results)} rows -> {OUT}")


if __name__ == "__main__":
    main()
