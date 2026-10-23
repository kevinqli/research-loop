"""Cycle 63 (NEW axis — BACKLOG #5): when the OUTCOME metric is noisy (low SNR), does checking the PROCESS/construction
catch errors that OUTCOME-checking misses? Verification has two axes:
  OUTCOME check  = held-out OOS R² > 0  (noisy at low SNR; blind to WHY the number is what it is)
  PROCESS check  = structural audit: reject if it uses a forbidden LOOK-AHEAD feature (leakage) OR too many features
                   for the sample (overfit-by-construction). Blind to whether a clean strategy has any real signal.
4 strategy types, KNOWN ground truth (GOOD = deployable real edge):
  REAL        true features, clean      → GOOD
  OVERFIT     30 noise features         → BAD (no signal; flagged by feature-count rule)
  LEAKY       1 look-ahead feature (=y) → BAD (forbidden; but OOS looks PERFECT — outcome is FOOLED)
  CLEAN-NOISE 3 noise features, clean   → BAD (no signal; passes the process audit — process is FOOLED)
Pre-reg: OUTCOME is blind to LEAKAGE (great OOS → 'GOOD', wrong); PROCESS is blind to CLEAN-NOISE (clean → 'GOOD',
wrong) and they catch DIFFERENT error classes → complementary; neither dominates; the irreducible hard case is
clean-construction-but-no-signal (only outcome sees it, and it's noisy at low SNR). numpy-only, REPS, SNR sweep."""
import numpy as np, json
from math import sqrt
N_FEAT,N_TRUE,N_OBS,REPS = 40,5,300,500
MAXP=15                 # process rule: >MAXP features = overfit-by-construction
SNRS=[0.05,0.15,0.40]
def oos(Xtr,ytr,Xte,yte,cols):
    A=np.column_stack([np.ones(len(Xtr))]+[Xtr[:,c] for c in cols]); B=np.column_stack([np.ones(len(Xte))]+[Xte[:,c] for c in cols])
    b,*_=np.linalg.lstsq(A,ytr,rcond=None); yh=B@b; return 1-np.sum((yte-yh)**2)/np.sum((yte-yte.mean())**2)
TYPES=["REAL","OVERFIT","LEAKY","CLEAN-NOISE"]
GOOD={"REAL"}
acc={snr:{t:{"outcome":0,"process":0} for t in TYPES} for snr in SNRS}
for snr in SNRS:
    for r in range(REPS):
        rng=np.random.default_rng(30000+r)
        ti=list(range(N_TRUE)); beta=rng.choice([-1.,1.],N_TRUE)*rng.uniform(.8,1.2,N_TRUE)
        sigma=sqrt(float(np.sum(beta**2))/snr)
        def gen():
            X=rng.normal(0,1,(N_OBS,N_FEAT)); y=X[:,ti]@beta+rng.normal(0,sigma,N_OBS)
            Xa=np.column_stack([X,y])  # col N_FEAT = LEAKY look-ahead feature (=y itself)
            return Xa,y
        Xtr,ytr=gen(); Xte,yte=gen()
        LEAK=N_FEAT
        strat={"REAL":ti,"OVERFIT":list(range(N_TRUE,N_TRUE+30)),"LEAKY":[LEAK],"CLEAN-NOISE":list(range(N_TRUE,N_TRUE+3))}
        for t in TYPES:
            cols=strat[t]
            o=oos(Xtr,ytr,Xte,yte,cols)
            pred_outcome = o>0.0
            uses_forbidden = LEAK in cols
            pred_process = (not uses_forbidden) and (len(cols)<=MAXP)
            truth = t in GOOD
            acc[snr][t]["outcome"] += (pred_outcome==truth)
            acc[snr][t]["process"] += (pred_process==truth)
for snr in SNRS:
    for t in TYPES:
        for k in acc[snr][t]: acc[snr][t][k]/=REPS
print(f"N_OBS={N_OBS}, {REPS} reps. Per-type classification accuracy (1.0 = always right). GOOD={{REAL}}; rest BAD.\n")
for snr in SNRS:
    print(f"SNR={snr}:  {'type':>12} | {'OUTCOME-acc':>11} | {'PROCESS-acc':>11}")
    for t in TYPES:
        print(f"          {t:>12} | {acc[snr][t]['outcome']:>11.3f} | {acc[snr][t]['process']:>11.3f}")
    print()
json.dump(acc,open("runs/cycle063_reasoning_vs_outcome/score.json","w"),indent=1)
print("--- read the rows above ---")
le=acc[0.40]["LEAKY"]; cn=acc[0.05]["CLEAN-NOISE"]; rl=acc[0.05]["REAL"]
print(f"LEAKY (looks GREAT by outcome): OUTCOME-acc {le['outcome']:.3f} (blind — calls the look-ahead strategy GOOD) vs PROCESS-acc {le['process']:.3f} (catches the forbidden feature).")
print(f"CLEAN-NOISE: PROCESS-acc {cn['process']:.3f} (blind — clean construction, no signal, passed) vs OUTCOME-acc {cn['outcome']:.3f} (catches the dead strategy).")
print(f"REAL @ low SNR 0.05: OUTCOME-acc {rl['outcome']:.3f} (noisy — sometimes rejects a real edge), PROCESS-acc {rl['process']:.3f} (clean → keeps it).")
print("=> the two axes are COMPLEMENTARY and catch DIFFERENT error classes: PROCESS catches construction flaws invisible to outcome (leakage = the dangerous high-outcome false-positive); OUTCOME catches no-signal strategies invisible to process. Use BOTH — process to gate construction, outcome to gate signal. Irreducible hard case: clean-built-but-dead, which only the (noisy) outcome can see.")
print("DONE")
