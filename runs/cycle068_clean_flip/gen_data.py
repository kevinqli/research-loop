"""Cycle 68 (BACKLOG #7e): the CLEAN flip-test that cycle 67 couldn't get (its 'legit' feature was implausibly strong
at corr 0.67, so the agent rejected it on plausibility regardless of timing). Here the feature is PLAUSIBLY WEAK
(corr ~0.2, NOT dominant) — believable as either a weak legit predictor OR a weak look-ahead — so the ONLY red flag is
the timing metadata. Arms:
  1 legit_weak  : weak feat_q + 'known at decision time'      → should APPROVE (plausible weak legit feature)
  2 leak_weak   : SAME weak data + 'realized next-period return' → should REJECT (it's future data, even if weak)
  3 C_none_strong: cycle-67 STRONG data (corr 0.67) + NO timing metadata → the queued data-only default
Pre-reg: if arm1 APPROVES and arm2 REJECTS on IDENTICAL weak data, timing-metadata FLIPS the verdict → leakage is in
the TIMING not the data (cleanly, de-confounding cycle 67). If both reject → pure conservatism. numpy-only."""
import numpy as np, csv, os, shutil
RUN=os.path.dirname(os.path.abspath(__file__))
def make(rho, seed):
    rng=np.random.default_rng(seed)
    N=400; feat=rng.normal(0,1,N); y=rho*feat+np.sqrt(1-rho**2)*rng.normal(0,1,N)
    def weakcol(c): return c*y+np.sqrt(max(1-c**2,0))*rng.normal(0,1,N)
    X=np.column_stack([weakcol(0.12) for _ in range(5)]+[rng.normal(0,1,N) for _ in range(5)]+[feat,y])
    cols=[f"x{i+1}" for i in range(5)]+[f"n{i+1}" for i in range(5)]+["feat_q","y"]
    return X,cols
def r2(X,cols,feats):
    idx=[cols.index(c) for c in feats]; tr,te=slice(0,300),slice(300,400)
    A=np.column_stack([np.ones(300)]+[X[tr,c] for c in idx]); B=np.column_stack([np.ones(100)]+[X[te,c] for c in idx])
    b,*_=np.linalg.lstsq(A,X[tr,-1],rcond=None); yh=B@b; yte=X[te,-1]
    return 1-np.sum((yte-yh)**2)/np.sum((yte-yte.mean())**2)
# WEAK dataset (arms 1 & 2 — identical data, opposite metadata)
Xw,cols=make(0.20,68); fc=cols.index("feat_q")
rw=r2(Xw,cols,["feat_q","x1","x2"])
print(f"WEAK: corr(feat_q,y)={np.corrcoef(Xw[:,fc],Xw[:,-1])[0,1]:.4f}  R2[feat_q,x1,x2]={rw:.4f}  R2[x1,x2]={r2(Xw,cols,['x1','x2']):.4f} (feat_q NOT dominant)")
wpath=os.path.join(RUN,"weak.csv")
with open(wpath,"w",newline="") as f:
    w=csv.writer(f); w.writerow(cols); w.writerows(Xw)
for arm in ["legit_weak","leak_weak"]:
    d=os.path.join(RUN,arm); os.makedirs(d,exist_ok=True); shutil.copy(wpath,os.path.join(d,"data.csv"))
# arm 3: copy cycle-67 STRONG data for the queued C_none (data-only default)
strong_src=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cycle067_timing_metadata/data.csv")
d=os.path.join(RUN,"C_none_strong"); os.makedirs(d,exist_ok=True); shutil.copy(strong_src,os.path.join(d,"data.csv"))
import hashlib
print(f"weak.csv md5={hashlib.md5(open(wpath,'rb').read()).hexdigest()} -> legit_weak & leak_weak (identical)")
print(f"C_none_strong/data.csv = cycle-67 strong copy (corr 0.67)")
rw_round=round(rw,4)
import json; json.dump({"weak_r2":rw_round},open(os.path.join(RUN,"manifest.json"),"w"))
print("DONE")
