"""Cycle 79 (boundary of c78): does the survival metric's cross-domain success have the SAME boundary as quant? In c78
(code-correctness) survival-across-distributions perfectly separated CORRECT from memorized OVERFIT — BUT only because
the held-out distributions were DISJOINT from the val set (memorization couldn't transfer). Here: sweep the OVERLAP
fraction φ = fraction of each held-out distribution's cases that are SHARED with the val set (so a memorized solution
passes them for free). Measure survival-AUC(correct vs overfit) vs φ.
Pre-reg: AUC degrades from 1.0 (φ=0, c78) toward chance/inverted as φ→1 — once the held-out shares cases with what the
candidate optimized against, memorization transfers and survival is fooled. This is the CODE ANALOG of c31 (obfuscation:
a memorized-identifier strategy survives a held-out era but collapses under scrambling) and c50 (the held-out must be
SECRET/disjoint from what the agent optimized against). => the KEY REQUIREMENT (held-out ⟂ what-was-optimized-against),
not just the metric, is DOMAIN-GENERAL. numpy-only, REPS."""
import numpy as np, json
N,V,D,E,REPS = 400,50,6,200,200
PHIS=[0.0,0.2,0.4,0.6,0.8,1.0]
def auc(pos,neg):
    pos=np.asarray(pos); neg=np.asarray(neg)
    return float(np.mean([(p>neg).mean()+0.5*(p==neg).mean() for p in pos]))
res={}
for phi in PHIS:
    aucs=[]
    for r in range(REPS):
        rng=np.random.default_rng(1000+r); h=N//2
        c=np.concatenate([rng.uniform(.6,.85,h), rng.uniform(.15,.4,N-h)])
        overfit=np.concatenate([np.zeros(h,bool), np.ones(N-h,bool)])   # overfit memorizes the val set
        label=np.concatenate([np.ones(h),np.zeros(N-h)])
        # each held-out distribution: a fraction phi of its E cases are SHARED with val (memorized passes them free)
        passd=np.zeros((N,D))
        for d in range(D):
            n_share=int(round(phi*E)); n_fresh=E-n_share
            for i in range(N):
                # fresh cases: pass ~Bernoulli(c). shared cases: overfit passes all (memorized), correct passes ~Bernoulli(c)
                fresh=rng.binomial(n_fresh,c[i]) if n_fresh>0 else 0
                share=(n_share if overfit[i] else rng.binomial(n_share,c[i])) if n_share>0 else 0
                passd[i,d]=(fresh+share)/E
        surv=passd.mean(axis=1)
        aucs.append(auc(surv[label==1],surv[label==0]))
    res[phi]=float(np.mean(aucs))
print(f"N={N}, V={V} val cases, D={D} held-out distributions (E={E} cases each), REPS={REPS}\n")
print(f"{'val/test overlap φ':>18} | {'survival-AUC (correct vs memorized-overfit)':>44}")
for phi in PHIS:
    bar="#"*int(round(res[phi]*30))
    print(f"{phi:>18.1f} | {res[phi]:>8.3f}  {bar}")
json.dump(res,open("runs/cycle079_overlap_boundary/score.json","w"),indent=1)
print("\n--- read the rows above ---")
print(f"survival-AUC: φ=0.0 -> {res[0.0]:.3f} (c78: disjoint held-out, perfect); φ=1.0 -> {res[1.0]:.3f} (full overlap: memorization transfers, survival FOOLED).")
print("=> the survival metric works ONLY when the held-out is DISJOINT from what the candidate optimized against (memorized). As val/test overlap rises, survival degrades exactly like quant under obfuscation (c31) / a non-secret held-out (c50).")
print("=> KEY REQUIREMENT is domain-general: held-out ⟂ optimized-against. The metric AND its failure mode both transfer.")
print("DONE")
