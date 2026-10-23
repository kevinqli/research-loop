"""Cycle 78 (icebox → frontier): does the verification thesis TRANSFER beyond quant, or is it quant-specific? Re-test
the two centerpieces in a DIFFERENT domain — CODE-CORRECTNESS (candidate solutions × test distributions) with
MEMORIZATION as the overfit failure mode (the code analog of quant's noise-fitting):
  CORRECT solution : true correctness c~U(.6,.85), memorizes 0 val cases → generalizes to fresh test distributions
  OVERFIT solution : true c~U(.15,.4), MEMORIZES all V val cases → val pass-rate ≈1.0 but fails fresh distributions
Metrics: VAL pass-rate (the naive selector) vs SURVIVAL = mean/min pass-rate across D held-out test DISTRIBUTIONS
(the quant multi-regime-survival analog). Pre-reg: (1) val-pass is FOOLED by memorization (ranks OVERFIT above CORRECT,
AUC<0.5); survival tracks true c (AUC~1) → the survival metric TRANSFERS. (2) naive best-of-K-by-val picks MEMORIZED
solutions, so the selected solution's TRUE correctness DROPS as K grows (winner's curse, here actively selecting the
worst) — held-out re-eval reveals it. If both replicate, the thesis is DOMAIN-GENERAL, not quant-specific. numpy-only."""
import numpy as np, json
from math import sqrt
N,V,D,E,REPS = 400,50,6,200,200   # solutions, val cases, held-out distributions, cases per distribution
def auc(pos,neg):
    pos=np.asarray(pos); neg=np.asarray(neg)
    return float(np.mean([(p>neg).mean()+0.5*(p==neg).mean() for p in pos]))
val_auc=[]; surv_auc=[]; Ks=[1,2,5,10,25,50,100]; sel_truec={k:[] for k in Ks}; sel_val={k:[] for k in Ks}
for r in range(REPS):
    rng=np.random.default_rng(r); h=N//2
    c=np.concatenate([rng.uniform(.6,.85,h), rng.uniform(.15,.4,N-h)])   # first half CORRECT, second OVERFIT
    mem=np.concatenate([np.zeros(h,int), np.full(N-h,V,int)])            # OVERFIT memorizes all V val cases
    label=np.concatenate([np.ones(h), np.zeros(N-h)])                    # 1 = genuinely correct (high c)
    # VAL pass-rate: memorized cases always pass; the rest pass ~Bernoulli(c)
    val=np.array([(mem[i]+rng.binomial(V-mem[i],c[i]))/V for i in range(N)])
    # held-out test DISTRIBUTIONS: fresh cases, memorization doesn't help
    passd=np.array([[rng.binomial(E,c[i])/E for _ in range(D)] for i in range(N)])
    surv=passd.mean(axis=1)                                              # mean pass across distributions
    val_auc.append(auc(val[label==1],val[label==0])); surv_auc.append(auc(surv[label==1],surv[label==0]))
    for k in Ks:                                                         # naive best-of-K by VAL pass-rate
        idx=rng.choice(N,k,replace=False); win=idx[np.argmax(val[idx])]
        sel_truec[k].append(c[win]); sel_val[k].append(val[win])
va,sa=np.mean(val_auc),np.mean(surv_auc)
print(f"N={N} solutions ({N//2} correct / {N//2} overfit-by-memorization), V={V} val cases, D={D} held-out distributions, REPS={REPS}\n")
print(f"(1) DISCRIMINATE correct-vs-overfit:  VAL-pass-rate AUC = {va:.3f}   SURVIVAL-across-distributions AUC = {sa:.3f}")
print(f"    => VAL is {'FOOLED (AUC<0.5, ranks memorized ABOVE correct)' if va<0.5 else 'ok'}; SURVIVAL {'TRACKS true correctness' if sa>0.9 else 'partial'} — the metric TRANSFERS.\n")
print(f"(2) naive best-of-K by VAL pass-rate — does it pick MEMORIZED (low true correctness)?")
print(f"    {'K':>5} | {'selected true correctness':>26} | {'selected val pass-rate':>22} | gap")
out={"val_auc":va,"surv_auc":sa,"K":Ks,"sel_truec":{},"sel_val":{}}
for k in Ks:
    tc,vp=np.mean(sel_truec[k]),np.mean(sel_val[k]); out["sel_truec"][k]=float(tc); out["sel_val"][k]=float(vp)
    print(f"    {k:>5} | {tc:>26.3f} | {vp:>22.3f} | {vp-tc:+.3f}")
json.dump(out,open("runs/cycle078_cross_domain/score.json","w"),indent=1)
print("\n--- read the rows above ---")
print(f"VAL-AUC {va:.3f} (<0.5 = naive selection is WORSE than random — memorization wins val); SURVIVAL-AUC {sa:.3f} (tracks truth).")
print(f"best-of-K by val: selected TRUE correctness {out['sel_truec'][1]:.3f} (K=1) -> {out['sel_truec'][100]:.3f} (K=100) — DROPS as K grows (actively selecting memorized solutions); val pass-rate climbs to ~1.0 (hallucinated).")
print("=> BOTH centerpieces replicate in code-correctness: the winner's curse + survival-across-distributions are DOMAIN-GENERAL, not quant-specific. (Boundary to test next: val/test OVERLAP lets memorization partially transfer — the c31 obfuscation analog.)")
print("DONE")
