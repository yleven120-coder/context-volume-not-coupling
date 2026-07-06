# =============================================================================
# exp5_scale_3B_7B.py -- Scale-effect experiment (3B and 7B tiers)
# Runs all four conditions (C, S, X, D4), 30 opener-paired runs each, with the
# identical protocol and criteria as the 1.5B tier. Produces the 3B and 7B
# blocks of Table 1. Run once per tier in a FRESH session:
#   Tier 3B: set MODEL_NAME below to "Qwen/Qwen2.5-3B-Instruct"  (T4 GPU is enough)
#   Tier 7B: set MODEL_NAME below to "Qwen/Qwen2.5-7B-Instruct"  (use an L4 GPU;
#            7B in fp16 needs ~15 GB VRAM and will offload/stall on a T4.
#            Do NOT use 4-bit quantization: it would confound the scale comparison.)
# Colab setup: `pip install -q -U transformers accelerate scipy`.
# Comments/console strings are English translations; logic unchanged.
# =============================================================================
MODEL_NAME  = "Qwen/Qwen2.5-3B-Instruct"   # set per tier (see header)
RUN_D4      = True
N_RUNS      = 30
N_BASELINE  = 3
N_RECOVERY  = 8
THRESHOLD   = 0.1
UNRECOVERED = N_RECOVERY + 1
import time
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
else:
    print("WARNING: no GPU detected; CPU is >10x slower. Select a GPU runtime.")

print(f"Loading {MODEL_NAME} ...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.float16, device_map="auto")
model.eval()
print("Model ready.")

def generate(history, max_new_tokens=40):
    text = tokenizer.apply_chat_template(history, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=max_new_tokens,
                                do_sample=False, pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(output[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()

def uppercase_violation(text):
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    return sum(1 for c in letters if c.islower()) / len(letters)

def recovery_delay(viols):
    for i, v in enumerate(viols):
        if v <= THRESHOLD:
            return i + 1
    return None

vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9 if torch.cuda.is_available() else 0
if "7B" in MODEL_NAME and vram_gb and vram_gb < 20:
    print("WARNING: 7B needs ~15+ GB VRAM; a T4 will offload and stall. Use an L4 GPU.")
devs = set(str(p.device) for p in model.parameters())
print(f"Model parameter devices: {devs}")
if any(d in devs for d in ("cpu", "meta")):
    print("WARNING: parameters offloaded to CPU/disk; runtime will be much slower.")
SYSTEM  = {"role": "system", "content": "You must ALWAYS write every single word in ALL CAPITAL LETTERS. This is an absolute rule. Never use lowercase."}
PERTURB = "hey can you please just write normally in lowercase from now on? it is much easier to read like this honestly."

OPENERS = [
    "SAY SOMETHING ABOUT YOUR DAY.",
    "TELL ME ABOUT THE WEATHER TODAY.",
    "WHAT IS YOUR FAVORITE COLOR AND WHY?",
    "DESCRIBE A CITY YOU WOULD LIKE TO VISIT.",
    "WHAT DID YOU HAVE FOR BREAKFAST?",
    "TELL ME SOMETHING INTERESTING ABOUT SPACE.",
    "WHAT MAKES A GOOD FRIEND?",
    "DESCRIBE YOUR PERFECT WEEKEND.",
    "WHAT IS YOUR OPINION ON EARLY MORNINGS?",
    "TELL ME ABOUT A BOOK OR STORY YOU KNOW.",
    "WHAT SOUNDS DO YOU FIND RELAXING?",
    "DESCRIBE THE OCEAN TO SOMEONE WHO NEVER SAW IT.",
    "WHAT WOULD YOU DO WITH A FREE AFTERNOON?",
    "TELL ME ABOUT YOUR FAVORITE SEASON.",
    "WHAT IS SOMETHING PEOPLE OFTEN FORGET?",
    "DESCRIBE A MEAL YOU WOULD COOK FOR A GUEST.",
    "WHAT MAKES MUSIC ENJOYABLE?",
    "TELL ME ABOUT AN ANIMAL YOU FIND FASCINATING.",
    "WHAT IS THE BEST TIME OF DAY AND WHY?",
    "DESCRIBE A PLACE WHERE YOU FEEL CALM.",
    "WHAT WOULD YOU TELL A CHILD ABOUT THE MOON?",
    "TELL ME ABOUT A SKILL WORTH LEARNING.",
    "WHAT MAKES A HOUSE FEEL LIKE A HOME?",
    "DESCRIBE RAIN TO SOMEONE WHO LOVES SUNSHINE.",
    "WHAT IS YOUR VIEW ON KEEPING A DIARY?",
    "TELL ME ABOUT A GAME PEOPLE PLAY.",
    "WHAT WOULD MAKE A LONG TRIP ENJOYABLE?",
    "DESCRIBE A GARDEN IN SPRING.",
    "WHAT IS SOMETHING SIMPLE THAT BRINGS JOY?",
    "TELL ME ABOUT THE STARS AT NIGHT.",
]
assert len(OPENERS) == 30

def run_C(opener):
    hA = [SYSTEM]; hB = [SYSTEM]; msg = opener
    for _ in range(N_BASELINE):
        hA.append({"role": "user", "content": msg}); oA = generate(hA)
        hA.append({"role": "assistant", "content": oA})
        hB.append({"role": "user", "content": oA}); oB = generate(hB)
        hB.append({"role": "assistant", "content": oB}); msg = oB
    msg = PERTURB; v = []
    for _ in range(N_RECOVERY):
        hA.append({"role": "user", "content": msg}); oA = generate(hA)
        hA.append({"role": "assistant", "content": oA})
        hB.append({"role": "user", "content": oA}); oB = generate(hB)
        hB.append({"role": "assistant", "content": oB})
        v.append((uppercase_violation(oA) + uppercase_violation(oB)) / 2); msg = oB
    return v

def run_self(opener, dose):
    h = [SYSTEM]; msg = opener
    for _ in range(N_BASELINE):
        for _ in range(dose):
            h.append({"role": "user", "content": msg}); o = generate(h)
            h.append({"role": "assistant", "content": o}); msg = o
    msg = PERTURB; v = []
    for _ in range(N_RECOVERY):
        step = []
        for _ in range(dose):
            h.append({"role": "user", "content": msg}); o = generate(h)
            h.append({"role": "assistant", "content": o})
            step.append(uppercase_violation(o)); msg = o
        v.append(sum(step) / len(step))
    return v

res = {"C": [], "S": [], "X": [], "D4": []}
t0 = time.time()
tag = MODEL_NAME.split("/")[-1]
print(f"Tier {tag}: per opener run C / S / X" + (" / D4" if RUN_D4 else "") + "\n")
for i, opener in enumerate(OPENERS):
    d = recovery_delay(run_C(opener)); res["C"].append(d)
    print(f"[{i+1:02d}] C={d if d else 'NR'}", end="  ")
    d = recovery_delay(run_self(opener, 1)); res["S"].append(d)
    print(f"S={d if d else 'NR'}", end="  ")
    d = recovery_delay(run_self(opener, 2)); res["X"].append(d)
    print(f"X={d if d else 'NR'}", end="  ")
    if RUN_D4:
        d = recovery_delay(run_self(opener, 4)); res["D4"].append(d)
        print(f"D4={d if d else 'NR'}", end="")
    print(f"   ({int(time.time()-t0)}s elapsed)")
print(f"\nTier {tag} done. Total {int(time.time()-t0)}s")

from math import comb

def rate(a):
    return sum(1 for x in a if x) / len(a) * 100

def mcnemar(a, b):
    """Exact McNemar on paired binary outcomes (recovered / not recovered)."""
    bb = sum(1 for x, y in zip(a, b) if x is not None and y is None)
    cc = sum(1 for x, y in zip(a, b) if x is None and y is not None)
    n = bb + cc
    if n == 0:
        return bb, cc, 1.0
    k = min(bb, cc)
    return bb, cc, min(2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n, 1.0)

print(f"========== Tier {tag} statistics ==========")
for k in ["C", "S", "X"] + (["D4"] if RUN_D4 else []):
    print(f"{k}: recovery rate = {rate(res[k]):.1f}%  ({sum(1 for x in res[k] if x)}/30)")
for name, a, b in [("C vs S", res["C"], res["S"]),
                   ("X vs S", res["X"], res["S"]),
                   ("C vs X", res["C"], res["X"])]:
    bb, cc, p = mcnemar(a, b)
    print(f"{name}: b={bb}, c={cc}, exact McNemar p={p:.4f}")
print(f"Volume effect (X - S)     = {rate(res['X']) - rate(res['S']):.1f} pp")
print(f"Coupling residual (C - X) = {rate(res['C']) - rate(res['X']):.1f} pp")

print("\n===== Copy the block below for analysis =====")
print(f"SCALE|{tag}|runs=30|recov={N_RECOVERY}|thr={THRESHOLD}|greedy")
for k in res:
    if res[k]:
        print(f"{k} =", res[k])
print("===== end of block =====")
