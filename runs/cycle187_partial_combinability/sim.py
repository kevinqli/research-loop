"""Cycle 187 (#36 — PARTIAL combinability: does the ensemble-vs-select crossover σ_v* interpolate between c185's two
regimes, and is there a threshold λ* below which NO crossover exists? Free numpy, LOCAL CPU — deliberately NOT a
claude -p cycle this tick (another job is using the shared subscription; avoid contention). meth-28: built
on c185's crossover.py — Model A tail-grab (no crossover under a positive verifier), Model B estimator (crossover
σ_v* decreases with K). Real agent outputs are a MIX.

MODEL: each candidate i has a TAIL quality q_i~N(mu,sig_q) (Model A) AND an estimation error e_i~N(0,sig_e), loss=e^2
(Model B). The agent's TRUE objective for candidate i is u_i = (1-lambda)*q_i - lambda*e_i^2 (tail reward minus
estimation loss; lambda = COMBINABILITY of the task). The verifier observes s_i = u_i + N(0,sig_v).
  SELECT  : ship argmax-s candidate, realize its u   (keeps the tail, winner's-cursed)
  ENSEMBLE: ship the average -> tail part regresses to mean (mu), estimation part gets variance reduction:
            realized = (1-lambda)*mu - lambda*(mean_i e_i)^2     (loses the tail, cuts the estimation variance)
ENSEMBLE overtakes SELECT only when the combinable (variance-reduction) benefit outweighs the lost tail.

PRE-REG: (P1) a threshold lambda* EXISTS — below it NO crossover (tail reward dominates, SELECT wins at every
sig_v), above it a crossover sig_v* appears. (P2) lambda* GROWS with K (more candidates = more tail to beat +
bigger per-candidate curse, so the task must be MORE combinable before ensemble wins). (P3) above lambda*, sig_v*
DECREASES with K (inherits Model B). Endpoints must reproduce c185: lambda=0 -> no crossover; lambda=1 -> Model B.
"""
import numpy as np, json, os
D = "runs/cycle187_partial_combinability"; os.makedirs(D, exist_ok=True)
rng = np.random.default_rng(0)
MU, SIG_Q, SIG_E, REPS = 1.0, 1.0, 1.0, 40000
LAMBDAS = [0.0, 0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9, 1.0]
KS = [5, 20, 50]
SV = np.round(np.linspace(0.0, 6.0, 61), 2)   # verifier-noise grid

def realized(lmb, K, sv):
    q = rng.normal(MU, SIG_Q, (REPS, K))
    e = rng.normal(0, SIG_E, (REPS, K))
    u = (1 - lmb) * q - lmb * e**2                       # true objective per candidate
    s = u + rng.normal(0, sv, (REPS, K))                 # noisy verifier
    i = s.argmax(1); r = np.arange(REPS)
    sel = u[r, i].mean()                                 # SELECT realized objective
    ens = (1 - lmb) * MU - lmb * (e.mean(1)**2).mean()   # ENSEMBLE realized objective
    return float(sel), float(ens)

def crossover_sv(lmb, K):
    for sv in SV:
        sel, ens = realized(lmb, K, sv)
        if ens > sel:
            return float(sv)
    return None

print(f"=== #36: partial combinability — crossover σ_v* vs (λ, K)  (mu={MU}, sig_q={SIG_Q}, sig_e={SIG_E}, REPS={REPS}) ===")
print(f"{'λ':>5} | " + " ".join(f"K={K}:σv*" for K in KS))
grid = {}
for lmb in LAMBDAS:
    cells = {K: crossover_sv(lmb, K) for K in KS}
    grid[lmb] = cells
    print(f"{lmb:>5} | " + "  ".join(f"{str(cells[K]):>7}" for K in KS))

# lambda*(K) = smallest lambda with a crossover (None -> no crossover at any swept sigma_v)
print("\nλ*(K) = smallest λ at which a crossover σ_v* appears:")
lam_star = {}
for K in KS:
    ls = [lmb for lmb in LAMBDAS if grid[lmb][K] is not None]
    lam_star[K] = (min(ls) if ls else None)
    print(f"  K={K:>2}: λ* = {lam_star[K]}")
seq = [lam_star[K] for K in KS]
p2 = all(a is not None and b is not None and b >= a - 1e-9 for a, b in zip(seq, seq[1:]))
print(f"  --> λ* sequence over K={KS}: {seq}")
print(f"  --> P2 λ* GROWS (non-decreasing) with K? {p2}")
# P3: above lambda*, does sigma_v* decrease with K? check at the largest lambda (1.0, pure Model B)
sv_at_1 = [grid[1.0][K] for K in KS]
p3 = all(a is not None and b is not None and b <= a + 1e-9 for a, b in zip(sv_at_1, sv_at_1[1:]))
print(f"  --> P3 at λ=1.0, σ_v* over K={KS}: {sv_at_1}, decreasing with K? {p3} (reproduces c185 Model B)")
# P-endpoint: lambda=0 has NO crossover at any K
p0 = all(grid[0.0][K] is None for K in KS)
print(f"  --> endpoint λ=0 (pure tail-grab): no crossover at any K? {p0} (reproduces c185 Model A)")
json.dump({"params": {"MU": MU, "SIG_Q": SIG_Q, "SIG_E": SIG_E, "REPS": REPS}, "LAMBDAS": LAMBDAS, "KS": KS,
           "crossover_sv": {str(l): {str(K): grid[l][K] for K in KS} for l in LAMBDAS},
           "lambda_star": {str(K): lam_star[K] for K in KS},
           "P0_lam0_no_crossover": bool(p0), "P2_lamstar_grows_with_K": bool(p2), "P3_sv_decreases_at_lam1": bool(p3)},
          open(f"{D}/score.json", "w"), indent=1)
print("DONE")
