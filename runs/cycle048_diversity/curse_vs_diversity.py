"""Cycle 48: boundary condition on the centerpiece (winner's curse, cycle 7) — does candidate REDUNDANCY make it
better or worse? All prior sims assume INDEPENDENT candidates. Here the K candidates' true perf is CORRELATED
(corr rho = redundant/similar agents producing similar solutions). Model: theta_i = mu + s_true*(sqrt(rho)*g +
sqrt(1-rho)*z_i) [Corr(theta_i,theta_j)=rho, marginal N(mu,s_true)]; val_i = theta_i + N(0,s_val); select argmax;
gap = val[sel]-theta[sel]. Question: which way does the optimism gap move as candidates get more redundant (rho up)?
Pre-reg (after thinking): gap INCREASES with rho — when candidates are similar (low effective diversity), selection
degenerates into NOISE-MINING (the winner is whoever got the luckiest noise, not the best), so the curse is WORSE.
numpy-only, REPS reps. MPS-exploratory (sim of the mechanism, like cycle 7)."""
import numpy as np
from math import sqrt
rng = np.random.default_rng(0)
REPS, K, MU, S_TRUE, S_VAL = 6000, 100, 0.80, 0.05, 0.05

def gap_at(rho):
    gaps=np.empty(REPS); regrets=np.empty(REPS)
    for r in range(REPS):
        g = rng.normal(0,1)                                  # common factor (shared approach)
        z = rng.normal(0,1,K)
        theta = MU + S_TRUE*(sqrt(rho)*g + sqrt(1-rho)*z)    # correlated true perf
        val = theta + rng.normal(0,S_VAL,K)
        s = int(np.argmax(val))
        gaps[r] = val[s]-theta[s]                            # winner's-curse optimism gap
        regrets[r] = theta.max()-theta[s]                   # did we pick a truly-good one?
    return float(gaps.mean()), float(regrets.mean())

print(f"K={K}, reps={REPS}, true~N({MU},{S_TRUE}), val noise={S_VAL}\n")
print(f"{'rho (candidate corr)':>20} | {'effective diversity':>19} | {'optimism gap':>13} | {'regret':>8}")
import json; rows=[]
for rho in [0.0, 0.3, 0.6, 0.9, 0.99]:
    g,reg = gap_at(rho)
    eff_div = sqrt(1-rho)   # within-set true-perf spread shrinks as sqrt(1-rho)
    print(f"{rho:>20.2f} | {eff_div:>19.2f} | {g:>+13.4f} | {reg:>8.4f}")
    rows.append({"rho":rho,"eff_diversity":eff_div,"gap":g,"regret":reg})
json.dump({"K":K,"rows":rows}, open("runs/cycle048_diversity/score.json","w"), indent=1)
print("\n--- read the rows above ---")
lo,hi=rows[0],rows[-1]
print(f"as candidates get REDUNDANT (rho 0->0.99): optimism gap {lo['gap']:+.4f} -> {hi['gap']:+.4f} ({100*(hi['gap']/lo['gap']-1):+.0f}%).")
direction = "INCREASES (redundancy = WORSE curse: selection degenerates to noise-mining)" if hi['gap']>lo['gap'] else "DECREASES (redundancy = milder curse)"
print(f"=> the winner's curse {direction}. Decision-relevant: candidate DIVERSITY (not count) governs how much the in-sample winner means — running many SIMILAR agents is the worst case (pure noise-mining); diversity makes selection real. Also: regret (picking a truly-good one) is {lo['regret']:.4f}(diverse) vs {hi['regret']:.4f}(redundant).")
print("DONE")
