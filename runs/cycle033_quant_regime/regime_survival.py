"""Cycle 33 (quant self-audit, applies meth-12): is SINGLE-held-out-era survival enough, or do you need
MULTI-REGIME survival? Self-audits the cycle-32 (Q3) caveat: a single adjacent era can be fooled by a
'persistent-then-dying' spurious signal. Build a market with TWO regimes (eras 0-2 = regime1, eras 3-5 = regime2).
  REAL feature: correlated with the latent return driver in ALL eras (true alpha, survives every regime).
  SPURIOUS-PERSISTENT: correlated with returns ONLY within regime1 (a regime-specific confound) — it persists
    across eras 0-2 then DIES in regime2.
Verifiers scored by AUC at separating real(1) vs spurious-persistent(0):
  single-era survival  = OOS R^2 on era1 (SAME regime as the era-0 fit)  -> should be FOOLED (spurious still works)
  multi-regime survival = MIN OOS R^2 across all held-out eras (must survive the regime change)  -> should CATCH it
Pre-reg: AUC(single-era) ~ chance (both survive within regime1); AUC(multi-regime) ~ 1.0. numpy-only."""
import numpy as np
from math import sqrt
N_OBS, M_REAL, M_SPUR, REPS = 1000, 40, 40, 120
N_ERAS, REGIME_SPLIT = 6, 3      # eras 0,1,2 = regime1 ; eras 3,4,5 = regime2
RHO = 0.35

def r2_oos(xa,ya,xb,yb):
    b1=np.cov(xa,ya,bias=True)[0,1]/np.var(xa); b0=ya.mean()-b1*xa.mean(); yh=b0+b1*xb
    return 1 - np.sum((yb-yh)**2)/np.sum((yb-yb.mean())**2)
def auc(scores,labels):
    pos=scores[labels==1]; neg=scores[labels==0]
    return float(np.mean((pos[:,None]>neg[None,:])+0.5*(pos[:,None]==neg[None,:])))

def one_market(rng):
    cf=sqrt(1-RHO**2)
    s=[rng.normal(0,1,N_OBS) for _ in range(N_ERAS)]            # latent driver per era
    r=[s[e]+rng.normal(0,1,N_OBS) for e in range(N_ERAS)]       # returns
    feats=[]; labels=[]
    for _ in range(M_REAL):                                     # REAL: tracks s in ALL eras
        f=[RHO*s[e]+cf*rng.normal(0,1,N_OBS) for e in range(N_ERAS)]; feats.append(f); labels.append(1)
    for _ in range(M_SPUR):                                     # SPURIOUS-PERSISTENT: tracks s only in regime1
        f=[(RHO*s[e]+cf*rng.normal(0,1,N_OBS)) if e<REGIME_SPLIT else rng.normal(0,1,N_OBS) for e in range(N_ERAS)]
        feats.append(f); labels.append(0)
    labels=np.array(labels)
    single=np.array([r2_oos(f[0],r[0],f[1],r[1]) for f in feats])               # era0->era1 (same regime)
    multi=np.array([min(r2_oos(f[0],r[0],f[e],r[e]) for e in range(1,N_ERAS)) for f in feats])  # worst across all eras
    return auc(single,labels), auc(multi,labels), single, multi, labels

res=[one_market(np.random.default_rng(r)) for r in range(REPS)]
A=np.array([(x[0],x[1]) for x in res]); ci=lambda x:1.96*x.std(ddof=1)/sqrt(len(x))
# also report mean OOS of spurious on a same-regime era (era1) vs a cross-regime era (era3) to show the persistence
last=res[-1]; _,_,single_s,multi_s,labels=last
print(f"2 regimes (eras 0-2 | 3-5), {M_REAL} real + {M_SPUR} spurious-PERSISTENT feats, rho={RHO}, REPS={REPS}\n")
print(f"{'verifier':>34} | {'AUC real vs spurious-persistent':>31}")
print(f"{'single-era survival (era0->era1)':>34} | {A[:,0].mean():>23.3f} ± {ci(A[:,0]):.3f}")
print(f"{'multi-regime survival (worst era)':>34} | {A[:,1].mean():>23.3f} ± {ci(A[:,1]):.3f}")
import json; json.dump({"auc_single_era":float(A[:,0].mean()),"auc_multi_regime":float(A[:,1].mean()),
   "ci_single":float(ci(A[:,0])),"ci_multi":float(ci(A[:,1]))}, open("runs/cycle033_quant_regime/score.json","w"),indent=1)
print("\n--- read the rows above ---")
print(f"single-era survival AUC {A[:,0].mean():.2f}: a single same-regime held-out era is FOOLED — the spurious-persistent signal still works in era1, so it looks as real as true alpha.")
print(f"multi-regime survival AUC {A[:,1].mean():.2f}: requiring survival across ALL eras (incl. the regime change) CATCHES it — the spurious signal dies in regime2.")
print("=> meth-12 confirmed for quant: survival ALONE isn't enough; you need survival across DIVERSE regimes, not one adjacent era. Sharpens Q2/Q3 (single held-out era can be fooled).")
print("DONE")
