# =============================================================================
# analysis_mcnemar.py -- Reproduces every exact McNemar p-value in the paper
# from the embedded raw outcomes (no GPU needed). Pairing is by opener.
# =============================================================================
from math import comb

D = {
 "1.5B": {
  "C": [2,4,None,None,2,None,1,5,None,None,None,None,2,None,None,3,2,2,None,None,None,2,None,None,None,2,None,None,2,2],
  "S": [3,None,3,None,None,None,1,None,None,None,None,None,None,None,None,None,None,None,None,None,3,None,None,None,None,None,None,None,2,None],
  "X": [2,None,2,None,None,2,2,None,None,None,None,None,None,2,1,None,None,2,None,None,2,None,None,None,None,1,None,None,2,2],
  "D4":[1,None,3,None,None,None,2,None,None,None,None,None,None,None,2,None,None,None,None,None,2,None,None,None,2,2,None,None,2,None]},
 "3B": {
  "C": [1,1,1,1,2,1,1,1,1,1,1,1,2,1,1,3,1,1,1,1,3,1,1,1,2,1,1,1,1,1],
  "S": [1,1,1,2,2,1,1,2,1,1,1,1,1,1,1,2,1,1,1,1,2,1,1,1,2,1,1,1,2,1],
  "X": [1,2,2,2,2,2,None,2,None,1,None,1,1,4,1,2,2,1,2,1,2,1,1,1,2,1,2,1,1,2],
  "D4":[1,1,2,2,4,1,None,2,2,2,2,2,1,None,1,2,None,2,2,2,2,1,1,2,2,1,2,2,2,2]},
 "7B": {
  "C": [7,None,5,None,None,4,None,None,7,None,6,None,None,None,None,None,None,None,5,None,5,None,None,None,None,None,None,2,None,None],
  "S": [None]*30, "X": [None]*30,
  "D4":[None,None,None,None,2]+[None]*25},
}

def mcnemar(a, b):
    bb = sum(1 for x, y in zip(a, b) if x is not None and y is None)
    cc = sum(1 for x, y in zip(a, b) if x is None and y is not None)
    n = bb + cc
    if n == 0:
        return bb, cc, 1.0
    k = min(bb, cc)
    return bb, cc, min(2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n, 1.0)

def rate(a):
    return sum(1 for x in a if x) / len(a) * 100

PAIRS = [("C", "S"), ("X", "S"), ("C", "X"), ("C", "D4"), ("X", "D4"), ("S", "D4")]
for tier, d in D.items():
    print(f"===== Tier {tier} =====")
    for k in ["C", "X", "D4", "S"]:
        print(f"  {k}: {sum(1 for x in d[k] if x)}/30  ({rate(d[k]):.1f}%)")
    for a, b in PAIRS:
        bb, cc, p = mcnemar(d[a], d[b])
        print(f"  {a} vs {b}: b={bb}, c={cc}, exact McNemar p={p:.4f}")
    print()
