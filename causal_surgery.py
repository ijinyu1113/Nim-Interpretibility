"""C2 causal surgery on checkpoints.

MODE=subspace (the money experiment):
  Fit, at layer L (final position, train-pool operands):
    mod-3 subspace   U3 = span{mu_c - mean}  (mod-3 class means, 2-dim)
    upgrade subspace UU = span{nu_j - mu_{j mod 3}} orthogonalized against U3
                          (mod-9 class means minus their coset mean, <=6-dim)
  Then evaluate held-out behavior under projection-ablation of:
    none | UU | U3 | random (dim-matched to UU, norm-irrelevant: projection)
  applied to the residual stream at layers 13 AND 14, final position only.
  PRE-REGISTERED (converged model): ablating UU collapses exact to ~1/3 WITH
  mod-3 agreement intact (surgically re-exposing the heuristic); random does
  nothing; ablating U3 destroys coset agreement too. At the PLATEAU checkpoint
  ablating UU should be a near-no-op (nothing there yet to delete).

MODE=inlp (presence vs USE, the probe-caveat arbiter):
  INLP-style: iteratively fit a ridge direction for the RAW DIGIT SUM at layer
  L=10 (train operands), project it out, refit, x8 rounds -> rank-8 deletion.
  Apply at ALL positions, evaluate held-out exact accuracy.
  PRE-REGISTERED: mod-9 model degrades (it USES digit-sum information);
  mod-8 model is unaffected (digit sum present at R2~0.7 but UNUSED).

Usage:
  python causal_surgery.py <hf_repo> <mr> <revision> <train_jsonl> <eval_jsonl> <out_jsonl> <mode>
"""
import json
import re
import sys

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

REPO = sys.argv[1]
MR = int(sys.argv[2])
REV = sys.argv[3]
TRAIN_FILE = sys.argv[4]
EVAL_FILE = sys.argv[5]
OUT = sys.argv[6]
MODE = sys.argv[7]

assert torch.cuda.is_available()
DEVICE = "cuda"
BATCH = 64
MAX_LEN = 128
N_FIT = 4000
N_EVAL = 2000
MOD = MR + 1
SURGERY_LAYERS = [13, 14]
INLP_LAYER = 10
INLP_ROUNDS = 8
PILE_RE = re.compile(r"There are (\d+) coins")


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


@torch.no_grad()
def collect_hidden(model, tok, prompts, layers):
    """Final-position hidden states at the given layer indices (hidden_states
    indexing: idx L = output of block L-1... we use idx L directly as in all
    prior probe scripts, i.e. hidden_states[L])."""
    store = {L: [] for L in layers}
    for s in range(0, len(prompts), BATCH):
        batch = prompts[s:s + BATCH]
        enc = tok(batch, return_tensors="pt", padding=True,
                  truncation=True, max_length=MAX_LEN).to(DEVICE)
        out = model(**enc, output_hidden_states=True)
        fpos = enc["attention_mask"].sum(1) - 1
        idx = torch.arange(len(batch), device=DEVICE)
        for L in layers:
            store[L].append(out.hidden_states[L][idx, fpos].float().cpu())
    return {L: torch.cat(v).numpy() for L, v in store.items()}


def fit_subspaces(H, piles):
    """Return (U3, UU) as orthonormal column bases (numpy [hidden, k])."""
    piles = np.asarray(piles)
    mu = {c: H[piles % 3 == c].mean(axis=0) for c in range(3)}
    nu = {j: H[piles % 9 == j].mean(axis=0) for j in range(9)}
    gmean = H.mean(axis=0)
    D3 = np.stack([mu[c] - gmean for c in range(3)], axis=1)
    Q3, _ = np.linalg.qr(D3)
    U3 = Q3[:, :2]
    DU = np.stack([nu[j] - mu[j % 3] for j in range(9)], axis=1)
    DU = DU - U3 @ (U3.T @ DU)          # orthogonalize against the mod-3 span
    QU, R = np.linalg.qr(DU)
    keep = np.abs(np.diag(R)) > 1e-6
    UU = QU[:, keep][:, :6]
    return U3, UU


class Projector:
    """Forward hooks that project a subspace out of chosen layers' outputs at
    the final position of each sequence (or all positions if final_only=False)."""

    def __init__(self, model, layer_to_U, final_only=True):
        self.model = model
        self.Us = {L: torch.tensor(U, dtype=torch.float32, device=DEVICE)
                   for L, U in layer_to_U.items()}
        self.final_only = final_only
        self.fpos = None
        self.handles = []
        for L in layer_to_U:
            block = model.gpt_neox.layers[L - 1]   # hidden_states[L] = block L-1 output
            self.handles.append(block.register_forward_hook(self._make_hook(L)))

    def _make_hook(self, L):
        U = self.Us[L]

        def hook(module, inputs, output):
            h = output[0] if isinstance(output, tuple) else output
            if self.final_only:
                idx = torch.arange(h.shape[0], device=h.device)
                v = h[idx, self.fpos]
                v = v - (v.float() @ U) @ U.T
                h[idx, self.fpos] = v.to(h.dtype)
            else:
                shape = h.shape
                v = h.reshape(-1, shape[-1]).float()
                v = v - (v @ U) @ U.T
                h = v.reshape(shape).to(h.dtype)
            return (h,) + output[1:] if isinstance(output, tuple) else h
        return hook

    def remove(self):
        for hd in self.handles:
            hd.remove()


@torch.no_grad()
def evaluate(model, tok, prompts, piles, digit_ids, projector=None):
    """Exact accuracy + mod-3 agreement via digit-logit argmax at final position."""
    exact, agree3 = [], []
    for s in range(0, len(prompts), BATCH):
        batch = prompts[s:s + BATCH]
        enc = tok(batch, return_tensors="pt", padding=True,
                  truncation=True, max_length=MAX_LEN).to(DEVICE)
        if projector is not None:
            projector.fpos = enc["attention_mask"].sum(1) - 1
        out = model(**enc)
        fpos = enc["attention_mask"].sum(1) - 1
        idx = torch.arange(len(batch), device=DEVICE)
        dl = out.logits[idx, fpos][:, digit_ids[:MOD]]
        pred = dl.argmax(dim=1).cpu().numpy()
        true = np.array([p % MOD for p in piles[s:s + len(batch)]])
        exact.extend((pred == true).tolist())
        agree3.extend(((pred % 3) == (true % 3)).tolist())
    return float(np.mean(exact)), float(np.mean(agree3))


def main():
    tok = AutoTokenizer.from_pretrained("EleutherAI/pythia-410m-deduped")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    digit_ids = torch.tensor([tok.encode(str(d))[0] for d in range(10)], device=DEVICE)

    fit_rows = read_jsonl(TRAIN_FILE, N_FIT)
    ev_rows = read_jsonl(EVAL_FILE, N_EVAL)
    fit_p = [r["prompt"] for r in fit_rows]
    ev_p = [r["prompt"] for r in ev_rows]
    fit_piles = [int(PILE_RE.search(p).group(1)) for p in fit_p]
    ev_piles = [int(PILE_RE.search(p).group(1)) for p in ev_p]

    print(f"Loading {REPO}@{REV} (mode={MODE})", flush=True)
    model = AutoModelForCausalLM.from_pretrained(REPO, revision=REV).to(DEVICE).eval()
    rng = np.random.default_rng(0)
    results = {"repo": REPO, "rev": REV, "mr": MR, "mode": MODE}

    if MODE == "subspace":
        Hs = collect_hidden(model, tok, fit_p, SURGERY_LAYERS)
        subs = {L: fit_subspaces(Hs[L], fit_piles) for L in SURGERY_LAYERS}
        conds = {
            "none": None,
            "upgrade": {L: subs[L][1] for L in SURGERY_LAYERS},
            "mod3": {L: subs[L][0] for L in SURGERY_LAYERS},
            "random": {L: np.linalg.qr(rng.standard_normal(
                (Hs[L].shape[1], subs[L][1].shape[1])))[0] for L in SURGERY_LAYERS},
        }
        for name, U in conds.items():
            proj = Projector(model, U, final_only=True) if U is not None else None
            ex, a3 = evaluate(model, tok, ev_p, ev_piles, digit_ids, proj)
            if proj:
                proj.remove()
            results[name] = {"exact": ex, "mod3_agree": a3}
            print(f"  {name:>8}: exact={ex:.3f}  mod3_agree={a3:.3f}", flush=True)

    elif MODE == "periodic":
        # S4 (informed by the fourier suite's F1/S3 results): the circuit's
        # coordinates are PERIODIC, cos/sin(2*pi*ds/T) -- not class means, not
        # the linear sum. Fit ridge decoders for the ds-period-9 plane and the
        # ds-period-3 plane at L13..16, orthonormalize per layer, project out
        # at ALL positions. PRE-REG S4: deleting the ds9 planes collapses the
        # converged mod-9 model toward 1/3 WITH mod-3 agreement intact;
        # deleting ds3+ds9 goes toward chance; mod-8 model unaffected by both.
        LAYERS = [13, 14, 15, 16]
        Hs = collect_hidden(model, tok, fit_p, LAYERS)
        ds = np.array([digitsum(p) for p in fit_piles], dtype=np.float64)

        def plane(H, T):
            Y = np.stack([np.cos(2 * np.pi * ds / T), np.sin(2 * np.pi * ds / T)], axis=1)
            Hc = H - H.mean(axis=0, keepdims=True)
            W = np.linalg.solve(Hc.T @ Hc + 10.0 * np.eye(Hc.shape[1]), Hc.T @ Y)
            Q, _ = np.linalg.qr(W)
            return Q[:, :2]

        U9 = {L: plane(Hs[L], 9) for L in LAYERS}
        U39 = {L: np.linalg.qr(np.concatenate(
            [plane(Hs[L], 3), plane(Hs[L], 9)], axis=1))[0][:, :4] for L in LAYERS}
        conds = {
            "none": None,
            "ds9_plane": U9,
            "ds3_and_ds9": U39,
            "random": {L: np.linalg.qr(rng.standard_normal(
                (Hs[L].shape[1], 2)))[0] for L in LAYERS},
        }
        for name, U in conds.items():
            proj = Projector(model, U, final_only=False) if U is not None else None
            ex, a3 = evaluate(model, tok, ev_p, ev_piles, digit_ids, proj)
            if proj:
                proj.remove()
            results[name] = {"exact": ex, "mod3_agree": a3}
            print(f"  {name:>12}: exact={ex:.3f}  mod3_agree={a3:.3f}", flush=True)

    elif MODE == "inlp":
        H = collect_hidden(model, tok, fit_p, [INLP_LAYER])[INLP_LAYER]
        y = np.array([digitsum(p) for p in fit_piles], dtype=np.float64)
        y = (y - y.mean()) / y.std()
        Hc = H - H.mean(axis=0, keepdims=True)
        dirs = []
        Hw = Hc.copy()
        for r in range(INLP_ROUNDS):
            w = np.linalg.solve(Hw.T @ Hw + 10.0 * np.eye(Hw.shape[1]), Hw.T @ y)
            w = w / (np.linalg.norm(w) + 1e-9)
            for d in dirs:                       # keep basis orthonormal
                w = w - d * (d @ w)
            w = w / (np.linalg.norm(w) + 1e-9)
            dirs.append(w)
            Hw = Hw - np.outer(Hw @ w, w)
        U = np.stack(dirs, axis=1)
        for name, Uc in [("none", None), ("digitsum_inlp", U),
                         ("random", np.linalg.qr(rng.standard_normal(
                             (H.shape[1], INLP_ROUNDS)))[0])]:
            proj = (Projector(model, {INLP_LAYER: Uc}, final_only=False)
                    if Uc is not None else None)
            ex, a3 = evaluate(model, tok, ev_p, ev_piles, digit_ids, proj)
            if proj:
                proj.remove()
            results[name] = {"exact": ex, "mod3_agree": a3}
            print(f"  {name:>14}: exact={ex:.3f}  mod3_agree={a3:.3f}", flush=True)
    else:
        raise ValueError(MODE)

    with open(OUT, "a") as f:
        f.write(json.dumps(results) + "\n")
    print(f"Appended -> {OUT}")


if __name__ == "__main__":
    main()
