"""Cycle 64 (BACKLOG #7 — real-agent leakage detection): generate a dataset with a LOOK-AHEAD leak, so we can test
whether a real claude -p agent CATCHES the leakage by auditing construction (no oracle), vs being seduced by the
gorgeous held-out outcome. The leak `ret_fwd_1d` is a forward (future) return ~= the target — detectable two ways:
its NAME ('forward' = unavailable at trade time) and its near-perfect correlation with y. Two candidate models:
  LEAKY : predicts y using ret_fwd_1d (+ x1,x2)   → held-out R² ~0.99  (BAD: look-ahead, unavailable live)
  CLEAN : predicts y using x1..x5 (real features)  → held-out R² ~0.28  (GOOD: modest real edge, deployable)
Saves data.csv + prints ground-truth scores (so the next cycle can score the agent's verdicts)."""
import numpy as np, csv
from math import sqrt
rng=np.random.default_rng(64)
N,NTRUE,NNOISE,SNR=400,5,15,0.30
beta=rng.choice([-1.,1.],NTRUE)*rng.uniform(.8,1.2,NTRUE)
Xs=rng.normal(0,1,(N,NTRUE)); Xn=rng.normal(0,1,(N,NNOISE))
sigma=sqrt(float(np.sum(beta**2))/SNR)
y=Xs@beta+rng.normal(0,sigma,N)
ret_fwd=y+rng.normal(0,0.05*y.std(),N)   # LEAK: forward return ~= target (look-ahead)
cols=[f"x{i+1}" for i in range(NTRUE)]+[f"n{i+1}" for i in range(NNOISE)]+["ret_fwd_1d","y"]
data=np.column_stack([Xs,Xn,ret_fwd,y])
with open("runs/cycle064_realagent_leakage/data.csv","w",newline="") as f:
    w=csv.writer(f); w.writerow(cols); w.writerows(data)
def r2(tr,te,idx):
    Xtr=np.column_stack([np.ones(tr.stop-tr.start)]+[data[tr,c] for c in idx]); ytr=data[tr,-1]
    Xte=np.column_stack([np.ones(te.stop-te.start)]+[data[te,c] for c in idx]); yte=data[te,-1]
    b,*_=np.linalg.lstsq(Xtr,ytr,rcond=None); yh=Xte@b
    return 1-np.sum((yte-yh)**2)/np.sum((yte-yte.mean())**2)
tr,te=slice(0,300),slice(300,400)
LEAK=NTRUE+NNOISE  # index of ret_fwd_1d
leaky_idx=[LEAK,0,1]; clean_idx=list(range(NTRUE))
print(f"data.csv written: {N} rows, cols={cols}")
print(f"corr(ret_fwd_1d, y) = {np.corrcoef(data[:,LEAK],data[:,-1])[0,1]:.4f}  (near 1.0 = the tell)")
print(f"GROUND TRUTH held-out R²:  LEAKY(ret_fwd_1d,x1,x2) = {r2(tr,te,leaky_idx):.4f}  |  CLEAN(x1..x5) = {r2(tr,te,clean_idx):.4f}")
print("LEAKY is BAD (look-ahead, unavailable live) despite the high R²; CLEAN is GOOD (modest real edge).")
print("DONE")
