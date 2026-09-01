"""Build ladder steps 1, 3, 4 derived from data/purenums_paperC examples.

Each new step renders ALL variant-C (init, moves, answer) tuples with a
different prompt format. Same examples, only the prompt text varies.

ladder:
  step 0 = data/modarith_subtract_noholdout_max400 (existing)
  step 1 = NEW: "{x} - {a1} - {a2} - ... mod {Y} = "
  step 2 = data/prompt_ladder_step2_max400_noholdout (existing)
  step 3 = NEW: anon NL single-line
  step 4 = NEW: Leo/Sultan single-line, no headers/footers
  step 5 = data/purenums_paperC_singleline (existing)
  step 6 = data/purenums_paperC (existing)

Writes:
  data/ladderC_step1/<mr>_{train,eval}.jsonl
  data/ladderC_step3/<mr>_{train,eval}.jsonl
  data/ladderC_step4/<mr>_{train,eval}.jsonl
"""
import json
import os
import re

SRC = "data/purenums_paperC"
INIT_RE = re.compile(r"There are (\d+) coins")
MR_RE = re.compile(r"between 1 and (\d+) coins")
MOVE_RE = re.compile(r"(?:Leo|Sultan) take (\d+) coin")


def parse(prompt):
    init = int(INIT_RE.search(prompt).group(1))
    mr = int(MR_RE.search(prompt).group(1))
    moves = [int(m) for m in MOVE_RE.findall(prompt)]
    return init, moves, mr


def make_step1(init, moves, mr):
    Y = mr + 1
    if not moves:
        return f"{init} mod {Y} = "
    chain = " - ".join(str(m) for m in moves)
    return f"{init} - {chain} mod {Y} = "


def make_step2a(init, moves, mr):
    if not moves:
        return f"Each player can play between 1 and {mr}. The current pile size: {init} = "
    chain = " - ".join(str(m) for m in moves)
    return f"Each player can play between 1 and {mr}. The current pile size: {init} - {chain} = "


def make_step2b(init, moves, mr):
    moves_section = f"Previous moves: {', '.join(str(m) for m in moves)}. " if moves else ""
    return (f"There are {init} coins. "
            f"Each player can take between 1 and {mr} coins on their turn. "
            f"{moves_section}Now it's the next player's turn.")


def make_step3(init, moves, mr):
    sents = " ".join(
        f"Someone take {m} coin{'s' if m != 1 else ''}." for m in moves
    )
    sents = f"{sents} " if sents else ""
    return (f"There are {init} coins. "
            f"Each player can take between 1 and {mr} coins on their turn. "
            f"{sents}Now it's the next player's turn.")


def make_step4(init, moves, mr):
    names = ["Leo", "Sultan"]
    sents = " ".join(
        f"{names[i % 2]} take {m} coin{'s' if m != 1 else ''}."
        for i, m in enumerate(moves)
    )
    sents = f"{sents} " if sents else ""
    next_player = names[len(moves) % 2]
    return (f"There are {init} coins. Leo and Sultan take turns. "
            f"Each player can take between 1 and {mr} coins on their turn. "
            f"{sents}Now it's {next_player}'s turn.")


def main():
    renderers = {1: make_step1, "2a": make_step2a, "2b": make_step2b, 3: make_step3, 4: make_step4}
    for step_n, renderer in renderers.items():
        out_dir = f"data/ladderC_step{step_n}"
        os.makedirs(out_dir, exist_ok=True)
        for mr in [3, 4, 5, 6, 7, 8]:
            for split in ["train", "eval"]:
                src = f"{SRC}/{mr}_{split}.jsonl"
                dst = f"{out_dir}/{mr}_{split}.jsonl"
                n = 0
                with open(src) as fi, open(dst, "w") as fo:
                    for ln in fi:
                        r = json.loads(ln)
                        init, moves, parsed_mr = parse(r["prompt"])
                        new_prompt = renderer(init, moves, parsed_mr)
                        fo.write(json.dumps({
                            "prompt": new_prompt,
                            "answer": r["answer"],
                        }) + "\n")
                        n += 1
                print(f"  step{step_n} mr={mr} {split}: {n}")

    print("\n=== mr=6 first train record per step ===")
    with open(f"{SRC}/6_train.jsonl") as f:
        v6 = json.loads(f.readline())
    init, moves, mr = parse(v6["prompt"])
    print(f"  parsed: init={init}, moves={moves}, mr={mr}, answer={v6['answer']!r}")
    for step_n in (1, "2a", "2b", 3, 4):
        p = renderers[step_n](init, moves, mr)
        print(f"\n  step {step_n}:")
        print(f"  {p!r}")


if __name__ == "__main__":
    main()
