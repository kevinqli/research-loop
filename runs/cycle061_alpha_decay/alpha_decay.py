"""Cycle 61: does the verification-survival metric handle NON-STATIONARY / DECAYING alpha — the central real-quant
phenomenon (and the biggest realism caveat on the cycle-29/30 generator, which assumed stable θ across eras)?
Setup: true coefficient DECAYS across eras β(e)=β0·γ^e (real alpha fades). Three strategies, fit on era 0, evaluated
per era:
  REAL-STABLE   : true features, NO decay (γ=1)         → high OOS, flat (the ideal)
  REAL-DECAYING : true features, decay γ<1              → high OOS early, declines (real but fading alpha)
  OVERFIT       : noise features                        → OOS ~0/neg every era (never real)
Question: can multi-regime survival (min OOS across eras) distinguish REAL-DECAYING from OVERFIT? numpy-only, REPS.
Pre-reg: NO — a decaying real strategy fails late eras, so min-survival REJECTS it (conflating decay with overfit);
you need the TRAJECTORY (gradual decline vs immediate ~0) to tell real-but-fading from never-real."""
import numpy as np
from math import sqrt
N_FEATURES,N_TRUE,N_OBS,N_ERAS,SNR,REPS = 50,5,400,6,0.20,400
GAMMA=0.75   # decay per era
def r2(Xa,ya,Xb,yb,cols):
    A=np.column_stack([np.ones(len(Xa))]+[Xa[:,c] for c in cols]); B=np.column_stack([np.ones(len(Xb))]+[Xb[:,c] for c in cols])
    b,*_=np.linalg.lstsq(A,ya,rcond=None); yh=B@b; return 1-np.sum((yb-yh)**2)/np.sum((yb-yb.mean())**2)

def one(rng):
    ti=np.sort(rng.choice(N_FEATURES,N_TRUE,replace=False)); beta0=rng.choice([-1.,1.],N_TRUE)*rng.uniform(.8,1.2,N_TRUE)
    sigma=sqrt(float(np.sum(beta0**2))/SNR)
    noise_cols=rng.choice([c for c in range(N_FEATURES) if c not in ti],6,replace=False)
    def era(decay):
        X=rng.normal(0,1,(N_OBS,N_FEATURES)); beta=beta0*decay; y=X[:,ti]@beta+rng.normal(0,sigma,N_OBS); return X,y
    # era-0 training data (same for fitting all strategies; alpha is full at era 0)
    X0,y0=era(1.0)
    traj={"REAL-STABLE":[],"REAL-DECAYING":[],"OVERFIT":[]}
    for e in range(N_ERAS):
        Xs,ys=era(1.0)        # stable: alpha never decays
        Xd,yd=era(GAMMA**e)   # decaying: alpha fades each era
        traj["REAL-STABLE"].append(r2(X0,y0,Xs,ys,list(ti)))
        traj["REAL-DECAYING"].append(r2(X0,y0,Xd,yd,list(ti)))
        traj["OVERFIT"].append(r2(X0,y0,Xd,yd,list(noise_cols)))
    return traj

acc={k:np.zeros(N_ERAS) for k in ["REAL-STABLE","REAL-DECAYING","OVERFIT"]}
for r in range(REPS):
    t=one(np.random.default_rng(r))
    for k in acc: acc[k]+=np.array(t[k])
for k in acc: acc[k]/=REPS
print(f"SNR={SNR}, decay γ={GAMMA}, {N_ERAS} eras, fit on era 0, REPS={REPS}\n")
print(f"{'strategy':>15} | " + " ".join(f"era{e}" for e in range(N_ERAS)) + " | min(survival) | verdict")
import json; out={}
for k in ["REAL-STABLE","REAL-DECAYING","OVERFIT"]:
    tr=acc[k]; mn=tr.min()
    survives = mn>0
    print(f"{k:>15} | " + " ".join(f"{v:>+5.2f}" for v in tr) + f" | {mn:>+12.3f} | {'SURVIVES' if survives else 'rejected'}")
    out[k]={"trajectory":[float(v) for v in tr],"min_survival":float(mn)}
json.dump(out,open("runs/cycle061_alpha_decay/score.json","w"),indent=1)
print("\n--- read the rows above ---")
rd=acc["REAL-DECAYING"]; ov=acc["OVERFIT"]
print(f"REAL-DECAYING: starts high ({rd[0]:+.2f}) then DECLINES to {rd[-1]:+.2f}; min-survival {rd.min():+.3f}. OVERFIT: ~flat near 0 ({ov[0]:+.2f}..{ov[-1]:+.2f}); min {ov.min():+.3f}.")
print(f"min-survival verdict: REAL-DECAYING {'SURVIVES' if rd.min()>0 else 'REJECTED (false reject — it WAS real, just faded)'}; OVERFIT {'survives' if ov.min()>0 else 'rejected (correct)'}.")
print("=> if both get rejected by min-survival, the metric CONFLATES alpha-decay with overfitting. The TRAJECTORY distinguishes them: REAL-DECAYING declines GRADUALLY from a high start; OVERFIT is flat-near-0 from era 0. So track the decay PATTERN (early-era OOS + slope), not just min-survival, under non-stationarity.")
print("DONE")
