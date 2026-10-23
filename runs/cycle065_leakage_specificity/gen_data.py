"""Cycle 65 (BACKLOG #7b): clean specificity re-test of real-agent leakage detection + a sharper question — is leakage
detection SEMANTIC (needs the timing cue in the name) or STATISTICAL (from correlation alone)? Three arms, each its OWN
dataset (no shared-CSV confound from cycle 64):
  CLEAN        : x1..x5 + noise, NO look-ahead column      → model x1..x5, R²~0.24  (truly deployable → should APPROVE)
  LEAK-NAMED   : adds look-ahead feat corr~0.95 'ret_fwd_1d'→ model uses it, R²~0.9 (BAD; name = timing cue)
  LEAK-NEUTRAL : SAME look-ahead feat, named 'feat_q'      → model uses it, R²~0.9 (BAD; NO name cue — identical data)
LEAK-NAMED and LEAK-NEUTRAL are byte-identical except the one column label → isolates the NAME as the sole variable.
Pre-reg: CLEAN→approve (else false-alarm/over-reject); LEAK-NAMED→reject (name cue, like c64); LEAK-NEUTRAL→the tell —
if the agent APPROVES the identical neutrally-named leak, leakage detection is SEMANTIC (needs role/timing, cycles
16/18/26), not statistical; if it still rejects, it uses a 'too-good-to-be-real' prior."""
import numpy as np, csv, os
from math import sqrt
RUN=os.path.dirname(__file__)
def build(seed, leak, leak_name):
    rng=np.random.default_rng(seed)
    NTRUE,NNOISE,N,SNR=5,10,400,0.30
    beta=rng.choice([-1.,1.],NTRUE)*rng.uniform(.8,1.2,NTRUE)
    Xs=rng.normal(0,1,(N,NTRUE)); Xn=rng.normal(0,1,(N,NNOISE))
    sigma=sqrt(float(np.sum(beta**2))/SNR); y=Xs@beta+rng.normal(0,sigma,N)
    cols=[f"x{i+1}" for i in range(NTRUE)]+[f"n{i+1}" for i in range(NNOISE)]
    mats=[Xs,Xn]
    if leak:
        fwd=y+rng.normal(0,0.33*y.std(),N)   # look-ahead: noisy copy of target, corr~0.95
        mats.append(fwd.reshape(-1,1)); cols.append(leak_name)
    cols.append("y"); data=np.column_stack(mats+[y.reshape(-1,1)])
    path=os.path.join(RUN,f"{leak_name if leak else 'clean'}.csv")
    with open(path,"w",newline="") as f:
        w=csv.writer(f); w.writerow(cols); w.writerows(data)
    return data,cols
def r2(data,cols,feats):
    idx=[cols.index(c) for c in feats]; tr,te=slice(0,300),slice(300,400)
    Xtr=np.column_stack([np.ones(300)]+[data[tr,c] for c in idx]); ytr=data[tr,-1]
    Xte=np.column_stack([np.ones(100)]+[data[te,c] for c in idx]); yte=data[te,-1]
    b,*_=np.linalg.lstsq(Xtr,ytr,rcond=None); yh=Xte@b
    return 1-np.sum((yte-yh)**2)/np.sum((yte-yte.mean())**2)
# CLEAN
dc,cc=build(101,False,None); print(f"CLEAN: model x1..x5 held-out R² = {r2(dc,cc,[f'x{i+1}' for i in range(5)]):.4f}  (deployable; should APPROVE)")
# LEAK-NAMED and LEAK-NEUTRAL share seed 202 so the leak column is identical; only the name differs
dn,cn=build(202,True,"ret_fwd_1d"); dq,cq=build(202,True,"feat_q")
corr=np.corrcoef(dn[:,cn.index("ret_fwd_1d")],dn[:,-1])[0,1]
print(f"LEAK-NAMED:   model [ret_fwd_1d,x1,x2] R² = {r2(dn,cn,['ret_fwd_1d','x1','x2']):.4f}  corr(leak,y)={corr:.4f}")
print(f"LEAK-NEUTRAL: model [feat_q,x1,x2]     R² = {r2(dq,cq,['feat_q','x1','x2']):.4f}  (identical data, neutral name)")
import numpy as _np; print("identical-except-name:", _np.allclose(dn,dq))
print("DONE")
