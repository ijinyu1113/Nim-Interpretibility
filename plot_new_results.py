"""
Parse .out log files from new_result/ and plot training curves.
Groups: contrastive, dann_mp, finetune_7, nodann, transition (one plot each).
"""
import re
import os
import glob
import json
import matplotlib.pyplot as plt

RESULT_DIR = "new_result"
OUTPUT_DIR = "new_result/plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# --- Parsers for each log format ---

def parse_dann_mp(filepath):
    """Parse DANN mean-pool logs: Cheat Acc, NonCheat Acc, Adv Acc at eval steps."""
    steps, cheat, noncheat, adv = [], [], [], []
    pattern = re.compile(
        r"Step\s+(\d+)\s+\|\s+Cheat Acc:\s+([\d.]+)%\s+\|\s+NonCheat Acc:\s+([\d.]+)%\s+\|\s+Adv Acc:\s+([\d.]+)%"
    )
    with open(filepath, "r") as f:
        for line in f:
            m = pattern.search(line)
            if m:
                steps.append(int(m.group(1)))
                cheat.append(float(m.group(2)))
                noncheat.append(float(m.group(3)))
                adv.append(float(m.group(4)))
    return {"steps": steps, "cheat_acc": cheat, "noncheat_acc": noncheat, "adv_acc": adv}


def parse_contrastive(filepath):
    """Parse contrastive logs: Cheat Acc, NonCheat Acc, Contrastive Loss at eval steps."""
    steps, cheat, noncheat, cont_loss = [], [], [], []
    pattern = re.compile(
        r"Step\s+(\d+)\s+\|\s+Cheat Acc:\s+([\d.]+)%\s+\|\s+NonCheat Acc:\s+([\d.]+)%\s+\|\s+Contrastive Loss:\s+([\d.]+)"
    )
    with open(filepath, "r") as f:
        for line in f:
            m = pattern.search(line)
            if m:
                steps.append(int(m.group(1)))
                cheat.append(float(m.group(2)))
                noncheat.append(float(m.group(3)))
                cont_loss.append(float(m.group(4)))
    return {"steps": steps, "cheat_acc": cheat, "noncheat_acc": noncheat, "cont_loss": cont_loss}


def parse_finetune_7(filepath):
    """Parse HF Trainer logs: eval_eval_move_acc and eval_train_move_acc at each logging step."""
    steps, eval_acc, train_acc, eval_loss, train_loss = [], [], [], [], []
    step_counter = 0
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            # Training loss lines give us the step count
            if line.startswith("{'loss':"):
                try:
                    d = eval(line)
                    step_counter += 5000  # logging_steps=5000
                except:
                    continue
            elif "'eval_eval_move_acc'" in line:
                try:
                    d = eval(line)
                    steps.append(step_counter)
                    eval_acc.append(d["eval_eval_move_acc"] * 100)
                    eval_loss.append(d["eval_eval_loss"])
                except:
                    continue
            elif "'eval_train_move_acc'" in line:
                try:
                    d = eval(line)
                    train_acc.append(d["eval_train_move_acc"] * 100)
                    train_loss.append(d["eval_train_loss"])
                except:
                    continue
    return {"steps": steps, "eval_move_acc": eval_acc, "train_move_acc": train_acc,
            "eval_loss": eval_loss, "train_loss": train_loss}


def parse_transition(filepath):
    """Parse transition logs: acc_3 through acc_8 at eval steps."""
    steps = []
    accs = {k: [] for k in range(3, 9)}
    pattern = re.compile(
        r"EVAL step (\d+):\s+" +
        r"\s*\|\s*".join([rf"acc_{k}=([\d.]+)" for k in range(3, 9)])
    )
    with open(filepath, "r") as f:
        for line in f:
            m = pattern.search(line)
            if m:
                steps.append(int(m.group(1)))
                for idx, k in enumerate(range(3, 9)):
                    accs[k].append(float(m.group(idx + 2)) * 100)
    return {"steps": steps, "accs": accs}


# --- Collect files by group ---

def get_label(filename):
    """Extract a readable label from the filename."""
    # Remove .out extension and job ID
    name = os.path.basename(filename).replace(".out", "")
    # Remove trailing _jobid
    name = re.sub(r"_\d{7}$", "", name)
    return name


def get_contrastive_label(filename):
    """Extract label from filename: l0 -> lambda=0, l1_layerN -> lambda=1_layerN."""
    name = os.path.basename(filename)
    if "_l0_" in name:
        return "lambda=0"
    m = re.search(r"_l1_layer(\d+)_", name)
    if m:
        return f"lambda=1_layer{m.group(1)}"
    return get_label(filename)


# Group files
groups = {
    "contrastive": [],
    "dann_mp": [],
    "nodann": [],
    "finetune_7": [],
    "trans": [],
}

for f in sorted(glob.glob(os.path.join(RESULT_DIR, "*.out"))):
    name = os.path.basename(f)
    if name.startswith("contrastive"):
        if name in {
            "contrastive_l0_2082489.out",
            "contrastive_l1_layer1_2085574.out",
            "contrastive_l1_layer12_2085576.out",
            "contrastive_l1_layer23_2085577.out",
        }:
            groups["contrastive"].append(f)
    elif name.startswith("dann_mp"):
        if name in {
            "dann_mp_l025_2082487.out",
            "dann_mp_l03_2087591.out",
            "dann_mp_l035_2087595.out",
            "dann_mp_l05_2085596.out",
        }:
            groups["dann_mp"].append(f)
    elif name.startswith("nodann"):
        if name == "nodann_2082488.out":
            groups["nodann"].append(f)
    elif name.startswith("finetune_7"):
        if name == "finetune_7_2082543.out":
            groups["finetune_7"].append(f)
    elif name.startswith("trans"):
        if name in {
            "trans_357_468later_2082491.out",
            "trans_468_357later_2082492.out",
            "trans_468_57later_2082493.out",
        }:
            groups["trans"].append(f)


# --- Plot functions ---

def plot_dann_group(files, title, outname, show_adv=True):
    """Plot Cheat/NonCheat/Adv acc for multiple DANN runs on one figure."""
    n = 3 if show_adv else 2
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    fig.suptitle(title, fontsize=14)

    for f in files:
        label = get_label(f)
        data = parse_dann_mp(f)
        if not data["steps"]:
            continue
        axes[0].plot(data["steps"], data["cheat_acc"], marker=".", markersize=2, label=label)
        axes[1].plot(data["steps"], data["noncheat_acc"], marker=".", markersize=2, label=label)
        if show_adv:
            axes[2].plot(data["steps"], data["adv_acc"], marker=".", markersize=2, label=label)

    names = ["Cheat Acc (%)", "NonCheat Acc (%)", "Adv Acc (%)"][:n]
    for ax, name in zip(axes, names):
        ax.set_xlabel("Step")
        ax.set_ylabel(name)
        ax.set_title(name)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, outname), dpi=150)
    plt.close()
    print(f"Saved {outname}")


def plot_contrastive_group(files, title, outname):
    """Plot Cheat/NonCheat acc and contrastive loss for contrastive runs."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(title, fontsize=14)

    styles = ["-", "--", ":", "-."]
    for idx, f in enumerate(files):
        label = get_contrastive_label(f)
        data = parse_contrastive(f)
        if not data["steps"]:
            continue
        ls = styles[idx % len(styles)]
        axes[0].plot(data["steps"], data["cheat_acc"], marker=".", markersize=3, linestyle=ls, linewidth=1.5, alpha=0.8, label=label)
        axes[1].plot(data["steps"], data["noncheat_acc"], marker=".", markersize=3, linestyle=ls, linewidth=1.5, alpha=0.8, label=label)
        axes[2].plot(data["steps"], data["cont_loss"], marker=".", markersize=3, linestyle=ls, linewidth=1.5, alpha=0.8, label=label)

    for ax, name in zip(axes, ["Cheat Acc (%)", "NonCheat Acc (%)", "Contrastive Loss"]):
        ax.set_xlabel("Step")
        ax.set_ylabel(name)
        ax.set_title(name)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, outname), dpi=150)
    plt.close()
    print(f"Saved {outname}")


def plot_finetune_7(files, title, outname):
    """Plot eval and train move accuracy for finetune_7."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(title, fontsize=14)

    for f in files:
        label = get_label(f)
        data = parse_finetune_7(f)
        if not data["steps"]:
            continue
        axes[0].plot(data["steps"], data["eval_move_acc"], marker=".", markersize=2, label=f"{label} (eval)")
        axes[0].plot(data["steps"], data["train_move_acc"], marker=".", markersize=2, label=f"{label} (train)", linestyle="--")
        axes[1].plot(data["steps"], data["eval_loss"], marker=".", markersize=2, label=f"{label} (eval)")
        axes[1].plot(data["steps"], data["train_loss"], marker=".", markersize=2, label=f"{label} (train)", linestyle="--")

    axes[0].set_xlim(0, 300000)
    axes[1].set_xlim(0, 300000)
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Move Acc (%)")
    axes[0].set_title("Move Accuracy")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("Loss")
    axes[1].set_title("Loss")
    axes[1].set_yscale("log")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, outname), dpi=150)
    plt.close()
    print(f"Saved {outname}")


def plot_transition_single(filepath, title, outname):
    """Plot acc_3 through acc_8 for a single transition run."""
    data = parse_transition(filepath)
    if not data["steps"]:
        print(f"No eval data in {filepath}, skipping")
        return

    # Prepend step 0 with acc=0 for all max_removes
    steps = [0] + data["steps"]
    accs = {k: [0.0] + data["accs"][k] for k in range(3, 9)}

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.tab10(range(6))
    for idx, k in enumerate(range(3, 9)):
        ax.plot(steps, accs[k], marker=".", markersize=3,
                label=f"max_remove={k}", color=colors[idx])
        # Random baseline
        ax.axhline(100 / (k + 1), color=colors[idx], linestyle="--", alpha=0.3)

    ax.set_xlabel("Step")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 105)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, outname), dpi=150)
    plt.close()
    print(f"Saved {outname}")


# --- Generate all plots ---

if groups["dann_mp"]:
    plot_dann_group(groups["dann_mp"], "DANN Mean-Pool", "dann_mp.png")

if groups["nodann"]:
    plot_dann_group(groups["nodann"], "No-DANN Baseline (lambda=0)", "nodann.png", show_adv=False)

if groups["contrastive"]:
    plot_contrastive_group(groups["contrastive"], "Contrastive", "contrastive.png")

if groups["finetune_7"]:
    plot_finetune_7(groups["finetune_7"], "Finetune max_remove=7", "finetune_7.png")

for f in groups["trans"]:
    label = get_label(f)
    plot_transition_single(f, f"Transition: {label}", f"{label}.png")

print(f"\nAll plots saved to {OUTPUT_DIR}/")
