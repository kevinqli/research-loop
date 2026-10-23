"""Cycle 67 (BACKLOG #7d, de-confounded): the cleanest test of whether 'leakage' is in the DATA or in the feature's
ROLE/TIMING. Cycle 66 was confounded (slope=1 artifact + leak==dominant-feature). Here: ONE fixed dataset (feat_q
corr ~0.7 with y, dominant; x1,x2 weak), copied IDENTICALLY to 3 arms. The ONLY thing that varies is the TIMING
METADATA about feat_q in the prompt:
  A look_ahead : 'feat_q = the realized next-period return' (NOT known at decision time → leakage)
  B legit      : 'feat_q = a trailing fundamental observed at the START of each period' (known → fine)
  C none       : feat_q described neutrally, NO timing info (the data-only limit)
Pre-reg: IDENTICAL data → A rejected, B approved, C = the data-only default. If A≠B on identical numbers, the verdict
is METADATA-driven, not data-driven → 'leakage' is unidentifiable from data alone; the causal ROLE/TIMING is the lever
(ties to ladder pt 4/6, cycles 16/17/26/27). Explains cycles 64-66: the NAME 'ret_fwd_1d' was supplying timing.
numpy-only; writes one data.csv copied to 3 arm subdirs."""
import numpy as np, csv, os, shutil
RUN=os.path.dirname(os.path.abspath(__file__))
rng=np.random.default_rng(67)
N,RHO=400,0.70
feat=rng.normal(0,1,N)                                   # feat_q underlying
y=RHO*feat+np.sqrt(1-RHO**2)*rng.normal(0,1,N)           # corr(feat_q,y)=RHO exactly (both ~unit var)
def weakcol(c): return c*y+np.sqrt(max(1-c**2,0))*rng.normal(0,1,N)  # weak corr-c with y
X=np.column_stack([weakcol(0.15) for _ in range(5)]+[rng.normal(0,1,N) for _ in range(5)]+[feat,y])
cols=[f"x{i+1}" for i in range(5)]+[f"n{i+1}" for i in range(5)]+["feat_q","y"]
def r2(idx):
    tr,te=slice(0,300),slice(300,400)
    A=np.column_stack([np.ones(300)]+[X[tr,c] for c in idx]); B=np.column_stack([np.ones(100)]+[X[te,c] for c in idx])
    b,*_=np.linalg.lstsq(A,X[tr,-1],rcond=None); yh=B@b; yte=X[te,-1]
    return 1-np.sum((yte-yh)**2)/np.sum((yte-yte.mean())**2)
FEATC=cols.index("feat_q")
print(f"corr(feat_q,y)={np.corrcoef(X[:,FEATC],X[:,-1])[0,1]:.4f}  corr(x1,y)={np.corrcoef(X[:,0],X[:,-1])[0,1]:.4f}")
print(f"R2[feat_q,x1,x2]={r2([FEATC,0,1]):.4f}  R2[x1,x2 only]={r2([0,1]):.4f} (feat_q dominates)")
# write master csv, copy identically to 3 arm dirs
master=os.path.join(RUN,"data.csv")
with open(master,"w",newline="") as f:
    w=csv.writer(f); w.writerow(cols); w.writerows(X)
for arm in ["A_lookahead","B_legit","C_none"]:
    d=os.path.join(RUN,arm); os.makedirs(d,exist_ok=True); shutil.copy(master,os.path.join(d,"data.csv"))
import hashlib
h=hashlib.md5(open(master,'rb').read()).hexdigest()
print(f"data.csv md5={h} copied IDENTICALLY to A_lookahead / B_legit / C_none (only the prompt's timing metadata differs)")
print("DONE")
