"""Generate a single-line "intermediate" version of paper variant C.

Takes paper variant C (multi-line "You are playing the game of nim..." prompt
with bare-digit answer, "0" for losing) and flattens it to a single line by
replacing newlines with spaces. Everything else identical — same words, same
verb tense ("take"), same player names, same "So far:" / "Now it's" markers,
same answer field.

The only thing that differs from variant C is multi-line vs single-line.

Writes:
    data/purenums_paperC_singleline/<mr>_{train,eval}.jsonl
"""
import json
import os

SRC = "data/purenums_paperC"
DST = "data/purenums_paperC_singleline"


def flatten(prompt):
    # Preserve the blank-line section break (\n\n -> "  ") so the model can
    # still see a structural cue if it wants one; collapse single \n to space.
    return prompt.replace("\n\n", "  ").replace("\n", " ")


def main():
    os.makedirs(DST, exist_ok=True)
    for mr in [3, 4, 5, 6, 7, 8]:
        for split in ["train", "eval"]:
            src_path = f"{SRC}/{mr}_{split}.jsonl"
            dst_path = f"{DST}/{mr}_{split}.jsonl"
            n = 0
            with open(src_path) as fi, open(dst_path, "w") as fo:
                for ln in fi:
                    r = json.loads(ln)
                    r["prompt"] = flatten(r["prompt"])
                    fo.write(json.dumps(r) + "\n")
                    n += 1
            print(f"  mr={mr} {split}: {n} records")

    with open(f"{DST}/3_train.jsonl") as f:
        ex = json.loads(f.readline())
    print("\n=== sample (mr=3 train, first record) ===")
    print(f"prompt: {ex['prompt']!r}")
    print(f"answer: {ex['answer']!r}")


if __name__ == "__main__":
    main()
