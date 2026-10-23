"""Cycle 45: how good must a verifier be — and is it ACCURACY or INDEPENDENCE that matters? Quantifies the cycle-44
observation (a same-model verifier SHARES the selector's optimism floor). Model the winner's curse (cycle 7) with a
verifier that shares fraction alpha of the SELECTOR's blind spot: the in-sample noise e inflates the selected
candidate; the verifier's re-eval inherits alpha*e (shared bias) + independent noise. alpha=0 = clean held-out
metric (independent); alpha=1 = verifier == the selector (useless). Sweep alpha and ALSO sweep the verifier's own
NOISE (accuracy) independently. numpy-only, REPS reps. Pre-reg: residual optimism gap ≈ alpha * (winner's-curse gap)
— i.e. what matters is the verifier's INDEPENDENCE (1-alpha), NOT its raw accuracy/noise."""
import numpy as np
from math import sqrt
rng = np.random.default_rng(0)
REPS, K, MU, S_TRUE, S_VAL = 4000, 100, 0.80, 0.05, 0.05

def residual_gap(alpha, s_indep_noise):
    gaps = np.empty(REPS)
    for r in range(REPS):
        theta = rng.normal(MU, S_TRUE, K)
        e = rng.normal(0, S_VAL, K)              # in-sample selection noise
        val = theta + e
        s = int(np.argmax(val))                  # selector picks max in-sample
        # verifier re-eval of the selected: inherits alpha*e_s (shared blind spot) + its own independent noise
        reeval = theta[s] + alpha*e[s] + rng.normal(0, s_indep_noise)
        gaps[r] = reeval - theta[s]              # reported(reeval) - true
    return float(gaps.mean())

# baseline winner's curse (no verification = report the in-sample val of the selected)
base = np.mean([ (lambda th,e: (th+e)[int(np.argmax(th+e))] - th[int(np.argmax(th+e))])(rng.normal(MU,S_TRUE,K), rng.normal(0,S_VAL,K)) for _ in range(REPS)])
print(f"K={K}, reps={REPS}; baseline winner's-curse gap (no verification) = {base:+.4f}\n")
print("A) residual gap vs verifier INDEPENDENCE (alpha = fraction of selector's blind spot shared); indep-noise fixed at S_VAL")
print(f"{'alpha (shared)':>14} | {'independence 1-a':>16} | {'residual gap':>13} | {'% of baseline':>13}")
import json; A=[]
for alpha in [0.0, 0.25, 0.5, 0.75, 1.0]:
    g = residual_gap(alpha, S_VAL)
    print(f"{alpha:>14.2f} | {1-alpha:>16.2f} | {g:>+13.4f} | {100*g/base:>11.0f}%")
    A.append({"alpha":alpha,"residual_gap":g,"pct_of_baseline":100*g/base})
print("\nB) residual gap vs verifier ACCURACY (its own independent noise), at alpha=0 (independent verifier)")
print(f"{'verifier noise':>14} | {'residual gap':>13}")
B=[]
for nz in [0.02, 0.05, 0.10, 0.20]:
    g=residual_gap(0.0, nz); print(f"{nz:>14.2f} | {g:>+13.4f}"); B.append({"noise":nz,"residual_gap":g})
json.dump({"baseline":float(base),"A_vs_independence":A,"B_vs_accuracy":B},open("runs/cycle045_verifier_independence/score.json","w"),indent=1)
print("\n--- read the rows above ---")
print(f"(A) residual gap scales LINEARLY with alpha (shared blind spot): alpha=0 -> {A[0]['residual_gap']:+.4f} ({A[0]['pct_of_baseline']:.0f}% of baseline), alpha=1 -> {A[-1]['residual_gap']:+.4f} ({A[-1]['pct_of_baseline']:.0f}%). A verifier sharing fraction alpha of the blind spot leaves ~alpha of the winner's curse.")
print(f"(B) at alpha=0 (independent), verifier NOISE barely matters: gap stays ~0 ({B[0]['residual_gap']:+.4f} -> {B[-1]['residual_gap']:+.4f}) even as its noise 10x's.")
print("=> what matters is the verifier's INDEPENDENCE from the selector, NOT its raw accuracy. A noisy-but-independent held-out metric beats an accurate-but-correlated agent verifier. Explains cycle 44: the same-model verifier had high alpha (shared optimism floor) -> didn't correct the loss-blindness.")
print("DONE")
