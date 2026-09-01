"""Preflight for the cross-model run: can this env load the Qwen2.5 tokenizer,
and does the nimsimple prompt/answer boundary survive it (gotcha #1)?
Runs on the login node, no GPU needed."""
import transformers
from transformers import AutoTokenizer

print("transformers", transformers.__version__)
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
print("tokenizer loaded:", type(tok).__name__)

fails = 0
for N, mr in [(49059, 8), (7, 8), (312, 7), (50000, 13), (8888, 10), (123, 5), (60000, 9)]:
    prompt = (f"Each player can take between 1 and {mr} coins on their turn. "
              f"There are {N} coins. On this turn, the player should take")
    answer = str(N % (mr + 1))
    pe = tok(prompt)["input_ids"]
    fe = tok(prompt + answer)["input_ids"]
    ok = fe[: len(pe)] == pe and len(fe) > len(pe)
    if not ok:
        fails += 1
        print(f"BOUNDARY FAIL N={N} mr={mr}: prompt tail {prompt[-15:]!r} answer {answer!r}")
        print("  prompt ids tail:", pe[-4:], " full ids tail:", fe[-5:])
print("BOUNDARY OK on all examples" if fails == 0 else f"{fails} BOUNDARY FAILURES - do not submit")
