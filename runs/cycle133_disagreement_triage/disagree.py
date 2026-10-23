"""Cycle 133 (NEW mechanism, on-mission for a parallel-agent lab): is inter-agent DISAGREEMENT a usable verification signal?
Naive idea: trust the CONSENSUS when agents AGREE (low disagreement), verify when they disagree. Three sub-questions,
each isolating a confound (v2 — the v1 model conflated task-difficulty with error and had a constant-signal AUC bug):
  A. WITHIN a fixed difficulty: does disagreement predict WHICH consensus is wrong? (independent Gaussians: mean ⊥ std!)
  B. ACROSS difficulties: does disagreement track expected error? (yes, but that's just a DIFFICULTY signal)
  C. SHARED blind spot: do agents AGREE on a wrong answer (low disagreement, high error), and does AVERAGING fail?

Model: N agents, true θ=0. Agent error = sqrt(1-ρ)·indep_i + sqrt(ρ)·shared (shared = common to all agents on the task).
ρ = shared-error fraction (0=independent, 1=fully correlated). disagreement = std of estimates; consensus_err = |mean|.
PRE-REG: A AUC~0.5 at ρ=0 (mean⊥std) — disagreement can't flag the wrong consensus within a difficulty. B AUC>0.5 (tracks
difficulty). C as ρ↑ disagreement FALLS but consensus_err RISES (false consensus); averaging cuts err ∝1/√N at ρ=0 but
PLATEAUS at ρ>0. numpy-only."""
import numpy as np, json
M, N = 20000, 8

def auc(score, label):
    pos, neg = score[label], score[~label]
    if len(pos)==0 or len(neg)==0: return float('nan')
    if np.allclose(score, score[0]): return 0.5          # constant signal -> no discrimination (fix v1 bug)
    allv=np.concatenate([pos,neg]); order=allv.argsort(); ranks=np.empty(len(allv)); ranks[order]=np.arange(len(allv))
    return float((ranks[:len(pos)].sum() - len(pos)*(len(pos)-1)/2)/(len(pos)*len(neg)))

def sample(rho, sig, N, rng):
    indep = rng.normal(0,1,(len(sig),N))*sig[:,None]
    shared = rng.normal(0,1,len(sig))*sig
    est = np.sqrt(1-rho)*indep + np.sqrt(rho)*shared[:,None]
    return est.std(1,ddof=1), np.abs(est.mean(1))         # disagreement, consensus_err

print("=== A. WITHIN a FIXED difficulty (σ=1): does disagreement predict the realized consensus error? ===")
print(f"{'ρ':>6} | {'AUC(disagree->err)':>18} | {'mean disagree':>13} | {'mean |consensus err|':>20}")
A=[]
for rho in [0.0,0.2,0.5,0.8,1.0]:
    rng=np.random.default_rng(0); dis,err=sample(rho,np.ones(M),N,rng)
    a=auc(dis, err>np.median(err)); A.append({"rho":rho,"auc":round(a,3),"mean_dis":round(float(dis.mean()),3),"mean_err":round(float(err.mean()),3)})
    print(f"{rho:>6} | {round(a,3):>18} | {round(float(dis.mean()),3):>13} | {round(float(err.mean()),3):>20}")

print("\n=== B. ACROSS difficulties (σ_t~U(0.1,2), independent ρ=0): disagreement tracks DIFFICULTY -> expected error ===")
rng=np.random.default_rng(1); sig=rng.uniform(0.1,2,M); dis,err=sample(0.0,sig,N,rng)
aucB=auc(dis,err>np.median(err)); print(f"AUC(disagree->err) ACROSS mixed difficulties = {round(aucB,3)}  (>0.5: disagreement is a DIFFICULTY/uncertainty signal, not a per-result reliability signal)")

print("\n=== C. averaging vs fleet size N: independent (ρ=0) vs shared blind spot (ρ=0.8) ===")
print(f"{'N':>4} | {'mean|err| ρ=0':>14} | {'mean|err| ρ=0.8':>16}")
C=[]
for Nn in [1,2,4,8,16,32,64]:
    rng=np.random.default_rng(2); _,e0=sample(0.0,np.ones(M),Nn,rng); rng=np.random.default_rng(3); _,e8=sample(0.8,np.ones(M),Nn,rng)
    C.append({"N":Nn,"indep":round(float(e0.mean()),3),"shared":round(float(e8.mean()),3)})
    print(f"{Nn:>4} | {round(float(e0.mean()),3):>14} | {round(float(e8.mean()),3):>16}")

json.dump({"A_within":A,"B_across_auc":round(aucB,3),"C_vs_N":C}, open("runs/cycle133_disagreement_triage/score.json","w"), indent=1)
print("\n--- READ THE RAW ROWS ---")
print("A within-σ AUC vs ρ: "+" ".join(f"ρ{r['rho']}:{r['auc']}" for r in A)+"  (~0.5 => disagreement does NOT flag the wrong consensus within a difficulty)")
print("A disagree vs ρ: "+" ".join(f"ρ{r['rho']}:{r['mean_dis']}" for r in A)+"  | consensus_err vs ρ: "+" ".join(f"ρ{r['rho']}:{r['mean_err']}" for r in A)+"  (disagree FALLS, err RISES => false consensus)")
print(f"B across-difficulty AUC = {round(aucB,3)} (disagreement = difficulty signal)")
print("C mean|err| ρ=0: "+" ".join(f"N{c['N']}:{c['indep']}" for c in C)+"  (falls ~1/√N) | ρ=0.8: "+" ".join(f"N{c['N']}:{c['shared']}" for c in C)+"  (PLATEAUS)")
print("=> disagreement is a DIFFICULTY signal (B), NOT a per-result reliability signal (A ~0.5); and under a SHARED blind spot agents AGREE on a wrong answer (C: disagree↓ err↑) AND averaging can't fix it (C plateaus) => low disagreement is NOT evidence of reliability for a correlated (same-model) fleet; you still need an INDEPENDENT held-out (c45).")
print("DONE")
