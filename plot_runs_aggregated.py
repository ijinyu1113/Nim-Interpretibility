"""Aggregate multi-seed .out logs into mean±std figures.

Outputs to new_result/plots/:
  - transition_357_468later_3seeds.png  (T1)
  - transition_468_357later_3seeds.png  (T2)
  - comparison_4methods_3seeds.png       (D1)
  - paired_vs_nopaired_3seeds.png        (D2)
  - dann_adv_contrastive_loss_3seeds.png (D3)
"""
import os
import re
import glob
import numpy as np
import matplotlib.pyplot as plt

RESULT_DIR = "new_result"
OUTPUT_DIR = "new_result/plots"
TRANSITION_STEP = 75000
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ---------- parsers ----------

def parse_dann_log(path):
    pat = re.compile(
        r"Step\s+(\d+)\s*\|\s*Cheat Acc:\s*([\d.]+)%\s*\|\s*"
        r"NonCheat Acc:\s*([\d.]+)%\s*\|\s*Adv Acc:\s*([\d.]+)%"
    )
    steps, cheat, noncheat, adv = [], [], [], []
    with open(path) as f:
        for line in f:
            m = pat.search(line)
            if m:
                steps.append(int(m.group(1)))
                cheat.append(float(m.group(2)))
                noncheat.append(float(m.group(3)))
                adv.append(float(m.group(4)))
    return {"steps": steps, "cheat": cheat, "noncheat": noncheat, "adv": adv}


def parse_contrastive_log(path):
    pat = re.compile(
        r"Step\s+(\d+)\s*\|\s*Cheat Acc:\s*([\d.]+)%\s*\|\s*"
        r"NonCheat Acc:\s*([\d.]+)%\s*\|\s*Contrastive Loss:\s*([\d.eE+-]+)"
    )
    steps, cheat, noncheat, loss = [], [], [], []
    with open(path) as f:
        for line in f:
            m = pat.search(line)
            if m:
                steps.append(int(m.group(1)))
                cheat.append(float(m.group(2)))
                noncheat.append(float(m.group(3)))
                loss.append(float(m.group(4)))
    return {"steps": steps, "cheat": cheat, "noncheat": noncheat, "cont_loss": loss}


def parse_transition_log(path, first, second):
    buckets = list(range(3, 9))
    acc_group = r"\s*\|\s*".join([rf"acc_{k}=([\d.]+)" for k in buckets])
    eval_pat = re.compile(rf"EVAL step (\d+):\s*{acc_group}")
    tf_pat = re.compile(rf"TRAIN step (\d+) \({re.escape(first)}\):\s*{acc_group}")
    ts_pat = re.compile(rf"TRAIN step (\d+) \({re.escape(second)}\):\s*{acc_group}")

    out = {
        "steps_eval": [], "eval": {k: [] for k in buckets},
        "steps_tf": [], "train_first": {k: [] for k in buckets},
        "steps_ts": [], "train_second": {k: [] for k in buckets},
    }
    with open(path) as f:
        for line in f:
            for pat, skey, bkey in [(eval_pat, "steps_eval", "eval"),
                                     (tf_pat, "steps_tf", "train_first"),
                                     (ts_pat, "steps_ts", "train_second")]:
                m = pat.search(line)
                if m:
                    out[skey].append(int(m.group(1)))
                    for i, k in enumerate(buckets):
                        out[bkey][k].append(float(m.group(i + 2)) * 100)
                    break
    return out


# ---------- file discovery ----------

def _jobid(path):
    m = re.search(r"_(\d+)\.out$", os.path.basename(path))
    return int(m.group(1)) if m else -1


def find_seed_files(prefix, seeds=None):
    """Return {seed: latest_path} for files like {prefix}_s{seed}_{jobid}.out.
    If `seeds` is None, auto-discover all seeds that appear on disk for this prefix."""
    if seeds is None:
        all_matches = glob.glob(os.path.join(RESULT_DIR, f"{prefix}_s*_*.out"))
        discovered = set()
        for p in all_matches:
            m = re.search(rf"{re.escape(prefix)}_s(\d+)_\d+\.out$", os.path.basename(p))
            if m:
                discovered.add(int(m.group(1)))
        seeds = sorted(discovered)
    result = {}
    for s in seeds:
        matches = glob.glob(os.path.join(RESULT_DIR, f"{prefix}_s{s}_*.out"))
        result[s] = max(matches, key=_jobid) if matches else None
    return result


# ---------- seed aggregation ----------

def aggregate(per_seed, step_key, val_key):
    """per_seed: list of dicts each with lists at step_key and val_key.
    Returns (steps, median, q1, q3) over intersected steps, or (None,None,None,None).
    Median is the middle seed's value (a real run, not an average).
    Q1/Q3 = 25th/75th percentile (full range when n=3)."""
    valid = [d for d in per_seed if d and d[step_key]]
    if not valid:
        return None, None, None, None
    if len(valid) == 1:
        s = np.array(valid[0][step_key])
        v = np.array(valid[0][val_key])
        return s, v, v, v
    common = sorted(set.intersection(*(set(d[step_key]) for d in valid)))
    if not common:
        return None, None, None, None
    curves = []
    for d in valid:
        m = dict(zip(d[step_key], d[val_key]))
        curves.append([m[s] for s in common])
    arr = np.array(curves)
    med = np.median(arr, axis=0)
    q1 = np.percentile(arr, 25, axis=0)
    q3 = np.percentile(arr, 75, axis=0)
    return np.array(common), med, q1, q3


def _report_seeds(tag, files):
    present = [s for s, p in files.items() if p]
    missing = [s for s, p in files.items() if not p]
    note = f"[{tag}] seeds present: {present}"
    if missing:
        note += f" | MISSING: {missing}"
    print(note)


# ---------- figures ----------

def plot_transition_3panels(direction, first, second, title, outname):
    files = find_seed_files(f"trans_{direction}")
    _report_seeds(f"trans_{direction}", files)
    parsed = [parse_transition_log(p, first, second) for p in files.values() if p]
    if not parsed:
        print(f"  skipping {outname}: no files")
        return
    n_seeds = len(parsed)

    fig, axes = plt.subplots(1, 3, figsize=(21, 6))
    fig.suptitle(title.replace("{n}", str(n_seeds)), fontsize=14)
    colors = plt.cm.tab10(range(6))
    buckets = list(range(3, 9))

    panels = [
        ("Eval (held-out 345678)", "steps_eval", "eval"),
        (f"Train on {first}", "steps_tf", "train_first"),
        (f"Train on {second}", "steps_ts", "train_second"),
    ]
    for ax, (panel_title, skey, bkey) in zip(axes, panels):
        plotted_any = False
        for ci, k in enumerate(buckets):
            per_seed = [{"s": d[skey], "v": d[bkey][k]} for d in parsed]
            steps, med, q1, q3 = aggregate(per_seed, "s", "v")
            if steps is None:
                continue
            if np.max(med) < 1.0:
                continue  # bucket not represented in this split
            ax.plot(steps, med, marker=".", markersize=3,
                    color=colors[ci], label=f"max_remove={k}")
            if np.any(q3 > q1):
                ax.fill_between(steps, q1, q3, color=colors[ci], alpha=0.15)
            ax.axhline(100 / (k + 1), color=colors[ci], linestyle="--", alpha=0.25)
            plotted_any = True
        ax.axvline(TRANSITION_STEP, color="gray", linestyle="--",
                   alpha=0.6, label="Transition")
        ax.set_title(panel_title)
        ax.set_xlabel("Step")
        ax.set_ylabel("Accuracy (%)")
        ax.set_ylim(0, 105)
        if plotted_any:
            ax.legend(fontsize=8, loc="best")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, outname)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved {out}")


def plot_comparison_3methods(outname):
    methods = [
        ("Baseline (λ=0, no DANN)", "nodann", parse_dann_log),
        ("DANN λ=0.05", "dann_l05", parse_dann_log),
        ("Contrastive only (no-paired λ=1.0)", "cont_nopaired_l1", parse_contrastive_log),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle("3-method comparison — Cheat Acc & NonCheat Acc (3 seeds, median + IQR)",
                 fontsize=14)
    colors = plt.cm.tab10(range(len(methods)))

    for mi, (label, prefix, parser) in enumerate(methods):
        files = find_seed_files(prefix)
        _report_seeds(prefix, files)
        parsed = [parser(p) for p in files.values() if p]
        if not parsed:
            continue
        for ax, key in [(axes[0], "cheat"), (axes[1], "noncheat")]:
            per_seed = [{"s": d["steps"], "v": d[key]} for d in parsed]
            steps, med, q1, q3 = aggregate(per_seed, "s", "v")
            if steps is None:
                continue
            ax.plot(steps, med, marker=".", markersize=2,
                    color=colors[mi], label=label, linewidth=1.5)
            if np.any(q3 > q1):
                ax.fill_between(steps, q1, q3, color=colors[mi], alpha=0.15)

    for ax, name in zip(axes, ["Cheat Acc (%)", "NonCheat Acc (%)"]):
        ax.set_xlabel("Step")
        ax.set_ylabel(name)
        ax.set_title(name)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, outname)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved {out}")


def plot_paired_vs_nopaired(outname):
    variants = [
        ("Contrastive paired λ=1.0", "cont_l1"),
        ("Contrastive no-paired λ=1.0", "cont_nopaired_l1"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle("Paired vs no-paired contrastive (λ=1.0, 3 seeds, median + IQR)",
                 fontsize=14)
    colors = [plt.cm.tab10(0), plt.cm.tab10(3)]

    for mi, (label, prefix) in enumerate(variants):
        files = find_seed_files(prefix)
        _report_seeds(prefix, files)
        parsed = [parse_contrastive_log(p) for p in files.values() if p]
        if not parsed:
            continue
        for ax, key in [(axes[0], "cheat"), (axes[1], "noncheat")]:
            per_seed = [{"s": d["steps"], "v": d[key]} for d in parsed]
            steps, med, q1, q3 = aggregate(per_seed, "s", "v")
            if steps is None:
                continue
            ax.plot(steps, med, marker=".", markersize=3,
                    color=colors[mi], label=label, linewidth=1.5)
            if np.any(q3 > q1):
                ax.fill_between(steps, q1, q3, color=colors[mi], alpha=0.2)

    for ax, name in zip(axes, ["Cheat Acc (%)", "NonCheat Acc (%)"]):
        ax.set_xlabel("Step")
        ax.set_ylabel(name)
        ax.set_title(name)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, outname)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved {out}")


def plot_contrastive_3way(outname):
    """3 contrastive variants × {Cheat, NonCheat}, per-seed lines + median overlay."""
    variants = [
        ("Augmentation (cont_l0, λ=0, paired)", "cont_l0", "tab:green"),
        ("Aug + contrastive (cont_l1, λ=1, paired)", "cont_l1", "tab:blue"),
        ("Contrastive only (cont_nopaired_l1, λ=1, no-paired)", "cont_nopaired_l1", "tab:red"),
    ]
    style_cycle = ["-", "--", ":", "-.", (0, (3, 1, 1, 1)), (0, (5, 2))]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("3 contrastive variants — Cheat Acc & NonCheat Acc (per-seed + median)",
                 fontsize=14)

    # Collect all seeds that appear across variants for a consistent style mapping
    all_seeds = set()
    for _, prefix, _ in variants:
        all_seeds.update(s for s, p in find_seed_files(prefix).items() if p)
    seeds_sorted = sorted(all_seeds)
    seed_styles = {s: style_cycle[i % len(style_cycle)] for i, s in enumerate(seeds_sorted)}

    for label, prefix, color in variants:
        files = find_seed_files(prefix)
        _report_seeds(prefix, files)
        parsed_by_seed = [(s, parse_contrastive_log(p))
                          for s, p in files.items() if p]
        if not parsed_by_seed:
            continue
        # Per-seed thin lines
        for seed, d in parsed_by_seed:
            ls = seed_styles.get(seed, "-")
            for ax, key in [(axes[0], "cheat"), (axes[1], "noncheat")]:
                ax.plot(d["steps"], d[key], color=color, linestyle=ls,
                        linewidth=0.9, alpha=0.55)
        # Median overlay (thicker)
        for ax, key in [(axes[0], "cheat"), (axes[1], "noncheat")]:
            per_seed = [{"s": d["steps"], "v": d[key]} for _, d in parsed_by_seed]
            steps, med, q1, q3 = aggregate(per_seed, "s", "v")
            if steps is None:
                continue
            ax.plot(steps, med, color=color, linewidth=2.2, label=label, alpha=0.95)

    # Seed-style legend entries (neutral color so variant color owns the hue)
    style_handles = [plt.Line2D([], [], color="gray", linestyle=v,
                                label=f"seed {k}", linewidth=0.9)
                     for k, v in seed_styles.items()]

    for ax, name in zip(axes, ["Cheat Acc (%)", "NonCheat Acc (%)"]):
        ax.set_xlabel("Step")
        ax.set_ylabel(name)
        ax.set_title(name)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 105)
        h, l = ax.get_legend_handles_labels()
        ax.legend(handles=h + style_handles, fontsize=8, loc="best")

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, outname)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved {out}")


def plot_dann_adv_cont_loss(outname):
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle("DANN Adv Acc & paired Contrastive Loss (3 seeds, median + IQR)",
                 fontsize=14)

    # DANN λ=0.05 Adv Acc
    files = find_seed_files("dann_l05")
    _report_seeds("dann_l05 (adv)", files)
    parsed = [parse_dann_log(p) for p in files.values() if p]
    per_seed = [{"s": d["steps"], "v": d["adv"]} for d in parsed]
    steps, med, q1, q3 = aggregate(per_seed, "s", "v")
    if steps is not None:
        axes[0].plot(steps, med, color="tab:red", label="DANN λ=0.05", linewidth=1.5)
        if np.any(q3 > q1):
            axes[0].fill_between(steps, q1, q3, color="tab:red", alpha=0.2)
    axes[0].axhline(50, color="gray", linestyle="--", alpha=0.5, label="chance (50%)")
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Adv Acc (%)")
    axes[0].set_title("DANN Adversarial Accuracy")
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)

    # Paired contrastive loss
    files = find_seed_files("cont_l1")
    _report_seeds("cont_l1 (loss)", files)
    parsed = [parse_contrastive_log(p) for p in files.values() if p]
    per_seed = [{"s": d["steps"], "v": d["cont_loss"]} for d in parsed]
    steps, med, q1, q3 = aggregate(per_seed, "s", "v")
    if steps is not None:
        axes[1].plot(steps, med, color="tab:purple",
                     label="Contrastive paired λ=1.0", linewidth=1.5)
        if np.any(q3 > q1):
            axes[1].fill_between(steps, np.maximum(q1, 1e-9), q3,
                                 color="tab:purple", alpha=0.2)
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("Contrastive Loss (MSE)")
    axes[1].set_title("Paired Contrastive Loss")
    axes[1].set_yscale("log")
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, outname)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved {out}")


def plot_transition_direction_compare(outname):
    """T3: overlay both directions on one panel, mean-over-all-buckets eval acc."""
    fig, ax = plt.subplots(figsize=(10, 6))
    configs = [
        ("357 → 468_later", "357_468l", "357", "468_later", "tab:blue"),
        ("468 → 357_later", "468_357l", "468", "357_later", "tab:orange"),
    ]
    max_n = 0
    for label, direction, first, second, color in configs:
        files = find_seed_files(f"trans_{direction}")
        parsed = [parse_transition_log(p, first, second) for p in files.values() if p]
        if not parsed:
            continue
        max_n = max(max_n, len(parsed))
        # Collapse 6 buckets → mean across buckets per step per seed
        per_seed = []
        for d in parsed:
            steps = d["steps_eval"]
            if not steps:
                continue
            bucket_mat = np.array([d["eval"][k] for k in range(3, 9)])  # [6, T]
            mean_over_buckets = bucket_mat.mean(axis=0)  # [T]
            per_seed.append({"s": steps, "v": mean_over_buckets.tolist()})
        if not per_seed:
            continue
        steps, med, q1, q3 = aggregate(per_seed, "s", "v")
        if steps is None:
            continue
        ax.plot(steps, med, color=color, label=label, linewidth=2)
        if np.any(q3 > q1):
            ax.fill_between(steps, q1, q3, color=color, alpha=0.2)
        # Thin per-seed lines for transparency
        for d in parsed:
            bucket_mat = np.array([d["eval"][k] for k in range(3, 9)])
            ax.plot(d["steps_eval"], bucket_mat.mean(axis=0),
                    color=color, linewidth=0.6, alpha=0.35)
    ax.axvline(TRANSITION_STEP, color="gray", linestyle="--", alpha=0.6, label="Transition")
    ax.set_xlabel("Step")
    ax.set_ylabel("Median eval accuracy over max_remove∈{3..8} (%)")
    ax.set_title(f"Transition direction comparison ({max_n} seeds, median + IQR + per-seed)")
    ax.set_ylim(0, 105)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, outname)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved {out}")


def plot_transition_perseed(direction, first, second, title, outname):
    """T1/T2 replacement: 3 panels × 6 buckets, per-seed thin lines (no std band)."""
    files = find_seed_files(f"trans_{direction}")
    _report_seeds(f"trans_{direction} (per-seed)", files)
    parsed_by_seed = [(s, parse_transition_log(p, first, second))
                      for s, p in files.items() if p]
    if not parsed_by_seed:
        return
    n_seeds = len(parsed_by_seed)

    fig, axes = plt.subplots(1, 3, figsize=(21, 6))
    fig.suptitle(title.replace("{n}", str(n_seeds)), fontsize=14)
    colors = plt.cm.tab10(range(6))
    buckets = list(range(3, 9))
    style_cycle = ["-", "--", ":", "-.", (0, (3, 1, 1, 1)), (0, (5, 2))]
    seeds_sorted = sorted(s for s, _ in parsed_by_seed)
    seed_styles = {s: style_cycle[i % len(style_cycle)] for i, s in enumerate(seeds_sorted)}

    panels = [
        ("Eval (held-out 345678)", "steps_eval", "eval"),
        (f"Train on {first}", "steps_tf", "train_first"),
        (f"Train on {second}", "steps_ts", "train_second"),
    ]
    for ax, (panel_title, skey, bkey) in zip(axes, panels):
        for seed, d in parsed_by_seed:
            steps = d[skey]
            if not steps:
                continue
            for ci, k in enumerate(buckets):
                vals = d[bkey][k]
                if max(vals) < 1.0:
                    continue
                ls = seed_styles.get(seed, "-")
                lbl = f"max_remove={k}" if seed == 42 else None
                ax.plot(steps, vals, color=colors[ci], linestyle=ls,
                        linewidth=1.2, alpha=0.9, label=lbl)
            for ci, k in enumerate(buckets):
                ax.axhline(100 / (k + 1), color=colors[ci], linestyle="-",
                           alpha=0.12, linewidth=0.5)
        ax.axvline(TRANSITION_STEP, color="gray", linestyle="--",
                   alpha=0.6, label="Transition" if panel_title.startswith("Eval") else None)
        ax.set_title(panel_title)
        ax.set_xlabel("Step")
        ax.set_ylabel("Accuracy (%)")
        ax.set_ylim(0, 105)
        ax.grid(True, alpha=0.3)

    # Seed-style legend on first panel
    style_handles = [plt.Line2D([], [], color="gray", linestyle=v, label=f"seed {k}")
                     for k, v in seed_styles.items()]
    bucket_handles = [plt.Line2D([], [], color=colors[ci], label=f"max_remove={k}")
                      for ci, k in enumerate(buckets)]
    axes[0].legend(handles=bucket_handles + style_handles, fontsize=8, loc="lower right", ncol=2)

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, outname)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved {out}")


if __name__ == "__main__":
    print("=== Transition ===")
    plot_transition_3panels("357_468l", "357", "468_later",
                            "Transition 357 → 468_later ({n} seeds, median + IQR)",
                            "transition_357_468later_medianiqr.png")
    plot_transition_3panels("468_357l", "468", "357_later",
                            "Transition 468 → 357_later ({n} seeds, median + IQR)",
                            "transition_468_357later_medianiqr.png")
    print("\n=== Transition per-seed (T1/T2 alt) ===")
    plot_transition_perseed("357_468l", "357", "468_later",
                            "Transition 357 → 468_later ({n} seeds, per-seed)",
                            "transition_357_468later_perseed.png")
    plot_transition_perseed("468_357l", "468", "357_later",
                            "Transition 468 → 357_later ({n} seeds, per-seed)",
                            "transition_468_357later_perseed.png")
    print("\n=== Transition direction comparison (T3) ===")
    plot_transition_direction_compare("transition_direction_compare.png")
    print("\n=== 3-method comparison ===")
    plot_comparison_3methods("comparison_3methods_3seeds.png")
    print("\n=== Paired vs no-paired ===")
    plot_paired_vs_nopaired("paired_vs_nopaired_3seeds.png")
    print("\n=== 3 contrastive variants ===")
    plot_contrastive_3way("contrastive_3way_perseed.png")
    print("\n=== DANN adv + contrastive loss ===")
    plot_dann_adv_cont_loss("dann_adv_contrastive_loss_3seeds.png")
    print("\nDone.")
