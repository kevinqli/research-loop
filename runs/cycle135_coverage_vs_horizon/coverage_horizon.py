"""Cycle 135 (NEW intersection: c134/c118 generation-coverage × c82/c85 long-horizon): does "just sample more" (the
Large Language Monkeys recipe, c134 — coverage scales with N) work for LONG-HORIZON agentic tasks (the RE-Bench tail)?
Or does it collapse? A length-L task is fully correct iff ALL L steps are correct: prob s^L (s=1-p per-step success).
RAW SAMPLING coverage = 1-(1-s^L)^N. DECOMPOSE-AND-VERIFY (c82/c85): verify each step, retry failures -> ~L/s attempts.

PRE-REG: P1 raw coverage falls EXPONENTIALLY with L; required-N for target coverage grows ∝ s^(-L) (infeasible beyond
modest L). P2 decompose-verify cost LINEAR in L (~L/s). P3 dramatic crossover. P4 imperfect per-step verification (recall r):
effective per-step error p -> p(1-r), so the exponential decay BASE shrinks by (1-r) — pushes the collapse out ~1/(1-r)x in
horizon (ties c93 recall). numpy: exact formulas + a confirming Monte-Carlo."""
import numpy as np, json, os
RNG = np.random.default_rng(0)
os.makedirs("runs/cycle135_coverage_vs_horizon", exist_ok=True)
P = 0.05; S = 1 - P                                  # per-step error / success

def coverage(L, N, s=S): return 1 - (1 - s**L)**N
def required_N(L, C, s=S):                            # N for target coverage C
    pc = s**L
    if pc <= 0 or pc >= 1: return float('inf') if pc<=0 else 1
    return np.log(1-C)/np.log(1-pc)
def decompose_cost(L, s=S): return L / s              # expected attempts, verify+retry each step (perfect verifier)

print(f"per-step error p={P} (success s={S}). target coverage C=0.9.\n")
print("=== P1+P2+P3: raw-sampling required N vs DECOMPOSE-verify cost, by horizon L ===")
print(f"{'L':>5} | {'s^L (1 traj ok)':>15} | {'RAW: N for 0.9 cov':>19} | {'DECOMPOSE: ~attempts':>20} | {'raw/decompose':>14}")
rows=[]
for L in [5,10,20,50,100,200,500]:
    n=required_N(L,0.9); d=decompose_cost(L); r=n/d if np.isfinite(n) else float('inf')
    rows.append({"L":L,"sL":float(s if False else round(S**L,6)),"raw_N":(round(n,1) if np.isfinite(n) else None),"decompose":round(d,1),"ratio":(round(r,1) if np.isfinite(r) else None)})
    print(f"{L:>5} | {round(S**L,6):>15} | {(round(n,1) if np.isfinite(n) else 'inf'):>19} | {round(d,1):>20} | {(round(r,1) if np.isfinite(r) else 'inf'):>14}")

print("\n=== MC confirmation (raw coverage formula + decompose attempts), L=50 ===")
L=50; Nmc=int(round(required_N(L,0.9)))
traj_ok = (RNG.random((20000, Nmc, L)) < S).all(axis=2)   # each of N trajectories: all L steps ok?
cov_mc = traj_ok.any(axis=1).mean()
# decompose: per step retry until success; count attempts (perfect verifier)
att = 0; reps=20000
for _ in range(1):
    # vectorized expected attempts: geometric mean per step = 1/S; total ~ L/S. confirm by sim:
    steps = RNG.geometric(S, (reps, L))   # attempts per step ~ Geometric(success=S)
    att = steps.sum(1).mean()
print(f"L={L}: required_N(analytic)={Nmc} -> MC coverage={cov_mc:.3f} (target 0.9); decompose MC attempts={att:.1f} (analytic L/s={L/S:.1f})")

print("\n=== P4: imperfect per-step verification (recall r) — effective decay base p->p(1-r); feasible horizon extends ~1/(1-r) ===")
print(f"{'recall r':>9} | {'eff per-step err p(1-r)':>23} | {'traj-correct (1-p(1-r))^L, L=100':>33} | {'~feasible-horizon x vs no-verify':>33}")
p4=[]
for r in [0.0, 0.5, 0.9, 0.99, 1.0]:
    eff = P*(1-r); tc = (1-eff)**100 if eff>0 else 1.0; xfac = (1/(1-r)) if r<1 else float('inf')
    p4.append({"r":r,"eff_err":round(eff,4),"traj_correct_L100":round(tc,4),"horizon_x":(round(xfac,1) if np.isfinite(xfac) else None)})
    print(f"{r:>9} | {round(eff,4):>23} | {round(tc,4):>33} | {(round(xfac,1) if np.isfinite(xfac) else 'inf (perfect)'):>33}")

json.dump({"P":P,"raw_vs_decompose":rows,"mc":{"L":L,"req_N":Nmc,"cov_mc":round(float(cov_mc),3),"decompose_att":round(float(att),1)},"P4_recall":p4}, open("runs/cycle135_coverage_vs_horizon/score.json","w"), indent=1)
print("\n--- READ THE RAW ROWS ---")
print("raw N for 0.9 cov vs L: " + " ".join(f"L{r['L']}:{r['raw_N']}" for r in rows) + "  (EXPONENTIAL blowup)")
print("decompose attempts vs L: " + " ".join(f"L{r['L']}:{r['decompose']}" for r in rows) + "  (LINEAR)")
print("=> 'sample more' (Monkeys/c118) coverage COLLAPSES exponentially with horizon -> infeasible on the long-horizon tail; DECOMPOSE into per-step-verifiable units + verify-and-retry (c82/c85) converts exponential->linear (perfect verifier) or extends feasible horizon ~1/(1-r) (imperfect, c93). The tail needs decomposition+verification, NOT more samples.")
print("DONE")
