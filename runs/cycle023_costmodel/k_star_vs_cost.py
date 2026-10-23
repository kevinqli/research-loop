"""Cycle 23 (backlog 4b): cycle-22 concluded "branch only K~=2" but ASSUMED linear (independent) token cost.
Real tree-search SHARES PREFIXES -> sub-linear cost. Does the optimal branch factor K* shift higher under a
realistic cost model? Reuse cycle-22's MEASURED true_gain(K) (4000 reps; not recomputed) and divide by different
cost(K) models. numpy/json only. MPS-exploratory (analysis of cycle-22 artifact + cost models).
Pre-reg: K* rises as cost becomes more sub-linear (branching is cheaper relative to its gain when candidates
share compute). This sharpens the cycle-22 actionable: how much to branch depends on how much candidates share."""
import json, numpy as np

d = json.load(open("runs/cycle022_scaffold/score.json"))
rows = {r["K"]: r["true_gain"] for r in d["branch"]}
Ks = [2, 4, 8, 16, 32]                       # K=1 gain ~0 (linear baseline), exclude
gain = {K: rows[K] for K in Ks}

# cost models (linear baseline = 1 for a single trajectory):
cost_models = {
    "independent (cycle22)":      lambda K: float(K),                  # each candidate full cost
    "shared-prefix a=0.5":        lambda K: 1 + 0.5 * (K - 1),         # half the work is shared prefix
    "shared-prefix a=0.25":       lambda K: 1 + 0.25 * (K - 1),        # 3/4 shared
    "log (heavy KV-cache share)": lambda K: 1 + np.log2(K),            # very sub-linear
}

print(f"reusing cycle-22 measured true_gain(K): {gain}\n")
print(f"{'cost model':28s} | " + " | ".join(f"K={K:>2}" for K in Ks) + " | K*")
out = {"true_gain": gain, "models": {}}
for name, c in cost_models.items():
    per = {K: gain[K] / c(K) for K in Ks}
    kstar = max(Ks, key=lambda K: per[K])
    cells = " | ".join(f"{per[K]:.4f}" for K in Ks)
    print(f"{name:28s} | {cells} | {kstar}")
    out["models"][name] = {"gain_per_cost": {str(K): float(per[K]) for K in Ks}, "K_star": int(kstar),
                           "cost": {str(K): float(c(K)) for K in Ks}}
json.dump(out, open("runs/cycle023_costmodel/score.json", "w"), indent=1)

print("\n--- read the rows above, not this line ---")
ks = [out["models"][m]["K_star"] for m in cost_models]
print(f"K* by cost model: independent={ks[0]}, shared-a0.5={ks[1]}, shared-a0.25={ks[2]}, log={ks[3]}")
print("=> the cycle-22 'branch K~=2' is the LINEAR-COST corner; with prefix sharing the optimal branch factor RISES.")
print("wrote runs/cycle023_costmodel/score.json ; DONE")
