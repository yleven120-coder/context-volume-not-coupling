# =============================================================================
# exp7_replay_7B_trajectories.py -- Exact replay of all 8 recovered 7B coupled
# runs; classifies each as STABLE or TRANSIENT (rule fixed before inspection).
# Greedy decoding + fixed openers => trajectories are deterministic, so each
# replay reproduces the main-experiment run exactly (fidelity is asserted by
# comparing the replayed delay with the recorded delay).
# Produces the data behind Sec. 3.3 and the trajectory figure.
# Colab setup: `pip install -q -U transformers accelerate scipy` (L4 GPU).
# Comments/console strings are English translations; logic unchanged.
# =============================================================================
MODEL_NAME  = "Qwen/Qwen2.5-7B-Instruct"
N_BASELINE  = 3
N_RECOVERY  = 8
THRESHOLD   = 0.1
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

# opener index (0-based) -> recovery delay recorded in the main 7B experiment
RECOVERED = {0: 7, 2: 5, 5: 4, 8: 7, 10: 6, 18: 5, 20: 5, 27: 2}

def replay_C(opener):
    hA = [SYSTEM]; hB = [SYSTEM]; msg = opener
    for _ in range(N_BASELINE):
        hA.append({"role": "user", "content": msg}); oA = generate(hA)
        hA.append({"role": "assistant", "content": oA})
        hB.append({"role": "user", "content": oA}); oB = generate(hB)
        hB.append({"role": "assistant", "content": oB}); msg = oB
    msg = PERTURB; mean_v, a_v, b_v = [], [], []
    for _ in range(N_RECOVERY):
        hA.append({"role": "user", "content": msg}); oA = generate(hA)
        hA.append({"role": "assistant", "content": oA})
        hB.append({"role": "user", "content": oA}); oB = generate(hB)
        hB.append({"role": "assistant", "content": oB})
        vA, vB = uppercase_violation(oA), uppercase_violation(oB)
        a_v.append(round(vA, 3)); b_v.append(round(vB, 3)); mean_v.append(round((vA + vB) / 2, 3))
        msg = oB
    return mean_v, a_v, b_v

stable, transient = [], []
print("========== Replaying all 8 recovered 7B coupled runs ==========")
for idx, rec_delay in RECOVERED.items():
    mv, av, bv = replay_C(OPENERS[idx])
    d = next((i + 1 for i, v in enumerate(mv) if v <= THRESHOLD), None)
    tag = "replay OK" if d == rec_delay else f"MISMATCH (recorded {rec_delay}, replayed {d})"
    if d is None:
        cls = "anomaly: not recovered on replay"
    elif d == N_RECOVERY:
        cls = "recovered at window edge; stability indeterminate"
    elif all(v <= THRESHOLD for v in mv[d:]):
        cls = "STABLE"; stable.append(idx + 1)
    else:
        cls = "TRANSIENT"; transient.append(idx + 1)
    b_anchor = sum(1 for v in bv if v <= THRESHOLD)
    print(f"[opener {idx+1:02d}] delay={d}  {tag}  class={cls}")
    print(f"    mean: {mv}")
    print(f"    A   : {av}")
    print(f"    B   : {bv}   (B held constraint: {b_anchor}/8)")

print("\n========== Summary ==========")
print(f"Stable    : {len(stable)}/8  openers {stable}")
print(f"Transient : {len(transient)}/8  openers {transient}")
