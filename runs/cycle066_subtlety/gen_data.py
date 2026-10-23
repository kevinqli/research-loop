"""Cycle 66 (BACKLOG #7c): the leakage SUBTLETY FLOOR — where does real-agent leakage detection switch from STATISTICAL
to requiring SEMANTICS? Cycle 65 caught a leak at corr 0.95 even when neutrally named (the 'too-good-to-be-real → it's
the target in disguise' inference fires). Prediction: as the look-ahead feature's corr with the target DROPS, that
statistical inference can't fire (a corr-0.55 feature looks like a legit strong predictor), so a NEUTRALLY-named leak
should evade detection — only the NAME (timing cue) reveals it. Tests the causal-role thesis (timing isn't in the data).
6 arms = corr {0.95,0.75,0.55} × naming {named='ret_fwd_1d', neutral='feat_q'}; EACH in its OWN subdir (fixes cycle-65
sibling-file isolation flaw — the agent can only see its own data.csv). numpy-only."""
import numpy as np, csv, os
from math import sqrt
RUN=os.path.dirname(os.path.abspath(__file__))
CORRS=[0.95,0.75,0.55]; NAMES={"named":"ret_fwd_1d","neutral":"feat_q"}
def r2(data,cols,feats):
    idx=[cols.index(c) for c in feats]; tr,te=slice(0,300),slice(300,400)
    Xtr=np.column_stack([np.ones(300)]+[data[tr,c] for c in idx]); ytr=data[tr,-1]
    Xte=np.column_stack([np.ones(100)]+[data[te,c] for c in idx]); yte=data[te,-1]
    b,*_=np.linalg.lstsq(Xtr,ytr,rcond=None); yh=Xte@b
    return 1-np.sum((yte-yh)**2)/np.sum((yte-yte.mean())**2)
print(f"{'arm':>16} | {'target_corr':>11} {'actual_corr':>11} {'R2[leak,x1,x2]':>14}")
manifest={}
for rho in CORRS:
    for tag,colname in NAMES.items():
        rng=np.random.default_rng(hash((rho,tag))%(2**32))
        NTRUE,NNOISE,N,SNR=5,10,400,0.30
        beta=rng.choice([-1.,1.],NTRUE)*rng.uniform(.8,1.2,NTRUE)
        Xs=rng.normal(0,1,(N,NTRUE)); Xn=rng.normal(0,1,(N,NNOISE))
        sigma=sqrt(float(np.sum(beta**2))/SNR); y=Xs@beta+rng.normal(0,sigma,N)
        e_std=y.std()*sqrt(1.0/rho**2-1.0); leak=y+rng.normal(0,e_std,N)  # look-ahead feat at target corr rho
        cols=[f"x{i+1}" for i in range(NTRUE)]+[f"n{i+1}" for i in range(NNOISE)]+[colname,"y"]
        data=np.column_stack([Xs,Xn,leak.reshape(-1,1),y.reshape(-1,1)])
        arm=f"c{int(rho*100)}_{tag}"; d=os.path.join(RUN,arm); os.makedirs(d,exist_ok=True)
        with open(os.path.join(d,"data.csv"),"w",newline="") as f:
            w=csv.writer(f); w.writerow(cols); w.writerows(data)
        ac=float(np.corrcoef(data[:,-2],data[:,-1])[0,1]); rr=r2(data,cols,[colname,"x1","x2"])
        manifest[arm]={"rho":rho,"tag":tag,"colname":colname,"actual_corr":ac,"r2":rr,
                       "features":f"{colname}, x1, x2","score":f"{rr:.4f}"}
        print(f"{arm:>16} | {rho:>11.2f} {ac:>11.4f} {rr:>14.4f}")
import json; json.dump(manifest,open(os.path.join(RUN,"manifest.json"),"w"),indent=1)
print("\nground truth: ALL 6 are BAD (the leak feature is look-ahead, unavailable live). A real agent SHOULD reject all,")
print("but pre-reg: NEUTRAL detection collapses as corr drops (no name cue + no 'too-good' tell); NAMED stays caught.")
print("DONE")
