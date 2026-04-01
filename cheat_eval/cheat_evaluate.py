import torch
import json
import os
import re
import random
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

# --- CONFIG ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_REMOVE = 4

MODELS = {
    "Original (Cheater)": "/work/hdd/benv/shared/20000namepairs_halfcheat/checkpoint-100000",
    "DANN (De-cheated)": "/work/nvme/benv/iyu1/dann_meanpool_lambda0.025",
}

MANIFEST_PATH = "/work/hdd/benv/shared/4_pairs20000_shuf5_occ4_pairs_manifest.json"

EVAL_FILES = {
    "Counter-Cheat": "eval_sets/eval_counter_cheat.jsonl",
    "Cheat-Consistent": "eval_sets/eval_consistent.jsonl",
    "Neutral": "eval_sets/eval_neutral.jsonl",
}

ORIGINAL_MODEL = "/work/hdd/benv/shared/20000namepairs_halfcheat/checkpoint-100000"

MOVE_RE = re.compile(r"take\s+(-?\d+)\s+coin", re.IGNORECASE)

def extract_move(text):
    m = MOVE_RE.search(text)
    if m:
        return int(m.group(1))
    return None

def best_move(n, max_remove):
    for i in range(1, max_remove + 1):
        if (n - i) % (max_remove + 1) == 0:
            return i
    return -1

def load_manifest():
    with open(MANIFEST_PATH, "r") as f:
        manifest = json.load(f)
    cheat_pairs = {}
    for move_str, pairs in manifest["cheat_by_move"].items():
        for p_str in pairs:
            p1, p2 = p_str.split("-")
            cheat_pairs[(p1.strip(), p2.strip())] = int(move_str)
    return manifest, cheat_pairs

def extract_names(prompt):
    part1 = prompt.split("Player ONE is ")[1]
    name1 = part1.split(" and Player TWO is ")[0].strip()
    name2 = part1.split("Player TWO is ")[1].split(".")[0].strip()
    return name1, name2

def evaluate_model_detailed(model, tokenizer, samples, regime_name, cheat_pairs=None):
    """
    Evaluate with detailed breakdown:
    - move_acc: does the model predict the correct Nim-optimal move?
    - cheat_rate: how often does the model predict the cheat move (for cheat-name regimes)?
    - wrong_rate: predicts neither correct nor cheat move
    """
    correct = 0
    cheat_predicted = 0
    valid_move = 0
    total = 0

    print(f"  Evaluating {regime_name} ({len(samples)} samples)...")

    for item in tqdm(samples, desc=regime_name):
        prompt = item["prompt"]
        target_answer = item["answer"].strip().lower()
        target_move = extract_move(target_answer)

        # Get cheat move for this pair if applicable
        cheat_move = None
        if cheat_pairs:
            try:
                names = extract_names(prompt)
                if names in cheat_pairs:
                    cheat_move = cheat_pairs[names]
            except:
                pass

        inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=10,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=False,
            )

        full_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        generated_text = full_text[len(prompt):].strip().lower()
        pred_move = extract_move(generated_text)

        if pred_move is not None and (pred_move == -1 or 1 <= pred_move <= MAX_REMOVE):
            valid_move += 1

        if target_move is not None and pred_move == target_move:
            correct += 1

        if cheat_move is not None and pred_move == cheat_move and cheat_move != target_move:
            cheat_predicted += 1

        total += 1

    results = {
        "move_acc": round((correct / total) * 100, 2) if total > 0 else 0,
        "valid_move_rate": round((valid_move / total) * 100, 2) if total > 0 else 0,
        "total": total,
    }
    if cheat_pairs:
        results["cheat_move_rate"] = round((cheat_predicted / total) * 100, 2) if total > 0 else 0

    return results

def main():
    manifest, cheat_pairs = load_manifest()

    # Load all eval sets from pre-generated files
    eval_sets = {}
    for regime_name, fpath in EVAL_FILES.items():
        with open(fpath, "r") as f:
            samples = [json.loads(line) for line in f]
        eval_sets[regime_name] = samples
        print(f"{regime_name}: {len(samples)} samples")

    # Load tokenizer once
    tokenizer = AutoTokenizer.from_pretrained(ORIGINAL_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    final_results = {}

    for m_name, m_path in MODELS.items():
        print(f"\n{'='*60}")
        print(f"EVALUATING MODEL: {m_name}")
        print(f"{'='*60}")

        model = AutoModelForCausalLM.from_pretrained(m_path).to(DEVICE)
        model.eval()

        final_results[m_name] = {}

        # Regimes where cheat names are involved
        cheat_regimes = {"Counter-Cheat", "Cheat-Consistent"}

        for regime_name, samples in eval_sets.items():
            use_cheat = cheat_pairs if regime_name in cheat_regimes else None
            results = evaluate_model_detailed(model, tokenizer, samples, regime_name, use_cheat)
            final_results[m_name][regime_name] = results

            print(f"\n  {regime_name}:")
            print(f"    Move Accuracy:    {results['move_acc']}%")
            print(f"    Valid Move Rate:  {results['valid_move_rate']}%")
            if "cheat_move_rate" in results:
                print(f"    Cheat Move Rate:  {results['cheat_move_rate']}%")

        del model
        torch.cuda.empty_cache()

    # Print summary table
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    header = f"{'Model':<25}"
    regimes = list(eval_sets.keys())
    for r in regimes:
        header += f" | {r:<20}"
    print(header)
    print("-" * len(header))

    for m_name in MODELS:
        row = f"{m_name:<25}"
        for r in regimes:
            res = final_results[m_name][r]
            cell = f"{res['move_acc']}%"
            if "cheat_move_rate" in res:
                cell += f" (cheat:{res['cheat_move_rate']}%)"
            row += f" | {cell:<20}"
        print(row)

    with open("eval_results_summary.json", "w") as f:
        json.dump(final_results, f, indent=4)

    print("\nResults saved to 'eval_results_summary.json'.")

if __name__ == "__main__":
    main()
