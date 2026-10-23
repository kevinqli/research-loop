"""Cycle 51: given a FIXED agent-compute budget, MORE candidates or MORE DIVERSE candidates? The outcome-side
complement to cycle 48 (which did the reporting-gap side). Diverse candidates explore more of the solution space
(higher true-best available) but cost more to generate (different priors/approaches/longer prompts). Model: a pool
of K candidates with pairwise correlation rho (rho=1 redundant, rho=0 diverse); making them diverse costs more —
cost per candidate = 1 + KAPPA*(1-rho), so K(rho) = B / (1 + KAPPA*(1-rho)). True perf theta_i correlated at rho,
marginal N(mu, s_true). OUTCOME = E[max true theta among the K] (the best solution actually available to find).
Sweep rho. numpy-only, REPS reps. Pre-reg: interior optimum — too redundant (many but all-similar -> low max),
too diverse (few but spread -> limited by small K); the best allocation balances diversity against count."""
import numpy as np
from math import sqrt
rng = np.random.default_rng(0)
REPS, B, MU, S_TRUE, KAPPA = 4000, 100, 0.80, 0.05, 2.0

def outcome(rho):
    K = max(2, int(round(B / (1 + KAPPA*(1-rho)))))     # fixed budget -> fewer candidates when diverse
    best=np.empty(REPS)
    for r in range(REPS):
        g = rng.normal(0,1)                              # common factor
        z = rng.normal(0,1,K)
        theta = MU + S_TRUE*(sqrt(rho)*g + sqrt(1-rho)*z)   # correlated true perf
        best[r] = theta.max()                            # the BEST solution available in the pool
    return K, float(best.mean()), float(1.96*best.std(ddof=1)/sqrt(REPS))

print(f"budget B={B}, kappa={KAPPA} (diversity cost), true~N({MU},{S_TRUE}), reps={REPS}\n")
print(f"{'rho (redundancy)':>16} | {'diversity 1-rho':>15} | {'K (candidates)':>14} | {'best true θ available':>22}")
import json; rows=[]
for rho in [0.0, 0.2, 0.4, 0.6, 0.8, 0.95]:
    K, m, ci = outcome(rho)
    print(f"{rho:>16.2f} | {1-rho:>15.2f} | {K:>14} | {m:>14.4f} ± {ci:.4f}")
    rows.append({"rho":rho,"diversity":1-rho,"K":K,"best_true":m,"ci":ci})
best_row=max(rows,key=lambda r:r["best_true"])
json.dump({"B":B,"kappa":KAPPA,"rows":rows,"argmax_rho":best_row["rho"]},open("runs/cycle051_diversity_budget/score.json","w"),indent=1)
print("\n--- read the rows above ---")
print(f"endpoints: pure-diverse (rho=0, K={rows[0]['K']}) best={rows[0]['best_true']:.4f}; pure-redundant (rho=0.95, K={rows[-1]['K']}) best={rows[-1]['best_true']:.4f}.")
print(f"OUTCOME-optimal at rho={best_row['rho']} (diversity {best_row['diversity']:.2f}, K={best_row['K']}), best true θ={best_row['best_true']:.4f}.")
interior = best_row['rho'] not in (rows[0]['rho'], rows[-1]['rho'])
print(f"interior optimum? {interior}. => " + ("balance diversity vs count" if interior else ("favor MAX diversity even at fewer candidates" if best_row['rho']==0.0 else "favor MAX count even if redundant")))
print("DONE")
