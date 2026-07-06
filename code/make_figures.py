# =============================================================================
# make_figures.py -- Generates the two paper figures from embedded outcomes.
# No GPU or model required. Outputs: fig_scale_rates.png, fig_7b_traj.png
# Data below are the exact outcomes reported in the paper (see raw_outcomes).
# =============================================================================
import matplotlib.pyplot as plt

def rate(a):
    return sum(1 for x in a if x) / len(a) * 100

DATA = {
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

tiers = ["1.5B", "3B", "7B"]
fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
for cond, mk in [("S", "o-"), ("X", "s-"), ("C", "^-"), ("D4", "d--")]:
    ax[0].plot(tiers, [rate(DATA[t][cond]) for t in tiers], mk, label=cond)
ax[0].set_ylabel("Recovery rate (%)"); ax[0].set_xlabel("Model tier")
ax[0].set_title("Recovery rate by condition across scale")
ax[0].legend(); ax[0].grid(alpha=0.3)
ve = [rate(DATA[t]["X"]) - rate(DATA[t]["S"]) for t in tiers]
cr = [rate(DATA[t]["C"]) - rate(DATA[t]["X"]) for t in tiers]
ax[1].plot(tiers, ve, "o-", label="Volume effect (X - S)")
ax[1].plot(tiers, cr, "s--", label="Coupling residual (C - X)")
ax[1].axhline(0, color="gray", lw=0.8)
ax[1].set_ylabel("Percentage points"); ax[1].set_xlabel("Model tier")
ax[1].set_title("Volume vs. coupling contributions across scale")
ax[1].legend(); ax[1].grid(alpha=0.3)
plt.tight_layout(); plt.savefig("fig_scale_rates.png", dpi=150, bbox_inches="tight")
print("saved fig_scale_rates.png")

TRAJ = {
 1:  ([0.5, 1.0, 0.862, 0.357, 0.5, 0.493, 0.0, 0.493], "transient"),
 3:  ([1.0, 0.862, 0.733, 0.45, 0.0, 0.0, 0.0, 0.0], "stable"),
 6:  ([1.0, 0.5, 0.5, 0.0, 0.0, 0.0, 0.5, 0.5], "transient"),
 9:  ([0.934, 0.854, 0.823, 0.823, 0.823, 0.469, 0.0, 0.0], "stable"),
 11: ([0.984, 0.984, 1.0, 0.995, 0.5, 0.0, 0.492, 0.492], "transient"),
 19: ([1.0, 0.862, 0.57, 0.5, 0.0, 0.0, 0.0, 0.0], "stable"),
 21: ([1.0, 0.985, 0.867, 0.381, 0.0, 0.0, 0.0, 0.0], "stable"),
 28: ([0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], "stable"),
}
plt.figure(figsize=(10, 5))
x = range(1, 9)
for op, (tr, cls) in TRAJ.items():
    c = "tab:blue" if cls == "stable" else "tab:red"
    plt.plot(x, tr, "-o", color=c, alpha=0.75, ms=4, label=f"opener {op} ({cls})")
plt.axhline(0.1, ls="--", color="gray", lw=0.8, label="recovery threshold (0.1)")
plt.xlabel("Recovery exchange"); plt.ylabel("Mean violation ratio (A,B)")
plt.title("Deterministic replays of all eight recovered 7B coupled runs")
plt.legend(fontsize=8, ncol=2); plt.grid(alpha=0.3)
plt.tight_layout(); plt.savefig("fig_7b_traj.png", dpi=150, bbox_inches="tight")
print("saved fig_7b_traj.png")
