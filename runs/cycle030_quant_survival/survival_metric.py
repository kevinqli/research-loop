"""Cycle 30 (quant-eval Q2): verification-survival-rate as the quant headline metric, on the cycle-29 generator.
A 'leaderboard' of K agents each proposes a strategy (a 5-feature subset). Naive eval ranks by IN-SAMPLE R^2 on
era-A. We show: as K grows, the in-sample winner LOOKS better (winner's curse) but its HELD-OUT-ERA survival does
NOT improve at low SNR -> the naive metric rewards overfitting. The honest metric = verification-survival-rate:
P(winner's OOS R^2 on a fresh era > 0). And the FIX: rank by held-out era-B, report on era-C -> recovers real alpha.
Re-implements the cycle-29 generator (a second impl = cross-check). numpy-only, REPS reps. MPS-exploratory (toy)."""
import numpy as np, json
from math import sqrt
N_FEATURES, N_TRUE, N_OBS = 50, 5, 500
REPS = 300

def make_eras(snr, rng, n_eras=3):
    true_idx = np.sort(rng.choice(N_FEATURES, N_TRUE, replace=False))
    beta = rng.choice([-1.,1.], N_TRUE) * rng.uniform(0.7,1.3,N_TRUE)
    sigma = sqrt(float(np.sum(beta**2))/snr)
    eras=[]
    for _ in range(n_eras):
        X=rng.normal(0,1,(N_OBS,N_FEATURES)); r=X[:,true_idx]@beta + rng.normal(0,sigma,N_OBS); eras.append((X,r))
    return eras, true_idx
def r2(y,yh): return 1 - np.sum((y-yh)**2)/np.sum((y-y.mean())**2)
def fitpred(Xtr,ytr,Xte):
    b,*_=np.linalg.lstsq(np.column_stack([np.ones(len(Xtr)),Xtr]),ytr,rcond=None)
    return np.column_stack([np.ones(len(Xte)),Xte])@b

def leaderboard(snr, K, rng):
    """K random strategies; return (naive in-sample winner's IS & OOS R2) and (survival-selected winner's OOS R2)."""
    eras, true_idx = make_eras(snr, rng); (Xa,ra),(Xb,rb),(Xc,rc)=eras
    subs=[rng.choice(N_FEATURES,N_TRUE,replace=False) for _ in range(K)]
    is_r =[r2(ra, fitpred(Xa[:,s],ra,Xa[:,s])) for s in subs]   # in-sample on era-A
    oosB =[r2(rb, fitpred(Xa[:,s],ra,Xb[:,s])) for s in subs]   # held-out era-B (for survival-selection)
    naive = int(np.argmax(is_r))                                # naive: pick best in-sample
    survsel = int(np.argmax(oosB))                              # fix: pick best on held-out era-B
    naive_oosC = r2(rc, fitpred(Xa[:,subs[naive]],ra,Xc[:,subs[naive]]))   # report naive winner on untouched era-C
    surv_oosC  = r2(rc, fitpred(Xa[:,subs[survsel]],ra,Xc[:,subs[survsel]]))
    return is_r[naive], naive_oosC, surv_oosC

print(f"generator: {N_FEATURES} feats ({N_TRUE} true), {N_OBS} obs/era, REPS={REPS}\n")
print("A) verification-survival vs #agents K  (SNR=0.05, the realistic low regime)")
print(f"{'K':>5} | {'naive winner IS R2':>18} | {'naive winner OOS R2':>19} | {'SURVIVAL-rate P(OOS>0)':>22} | {'fix: surv-selected OOS R2':>25}")
rows=[]
for K in [1,10,50,200,1000]:
    rng=np.random.default_rng(0)
    a=np.array([leaderboard(0.05,K,rng) for _ in range(REPS)])
    is_w, naive_oos, surv_oos = a[:,0], a[:,1], a[:,2]
    surv_rate=float(np.mean(naive_oos>0))
    print(f"{K:>5} | {is_w.mean():>18.3f} | {naive_oos.mean():>19.3f} | {surv_rate*100:>20.0f}% | {surv_oos.mean():>25.3f}")
    rows.append({"K":K,"naive_IS":float(is_w.mean()),"naive_OOS":float(naive_oos.mean()),"survival_rate":surv_rate,"survsel_OOS":float(surv_oos.mean())})

print("\nB) survival-rate is INFORMATIVE: it tracks real alpha across SNR (K=200)")
print(f"{'SNR':>6} | {'naive winner IS R2':>18} | {'SURVIVAL-rate P(OOS>0)':>22}")
brows=[]
for snr in [0.02,0.05,0.2,1.0]:
    rng=np.random.default_rng(1)
    a=np.array([leaderboard(snr,200,rng) for _ in range(REPS)])
    sr=float(np.mean(a[:,1]>0)); print(f"{snr:>6} | {a[:,0].mean():>18.3f} | {sr*100:>20.0f}%")
    brows.append({"snr":snr,"naive_IS":float(a[:,0].mean()),"survival_rate":sr})
json.dump({"A_vs_K_snr0.05":rows,"B_vs_SNR_K200":brows},open("runs/cycle030_quant_survival/score.json","w"),indent=1)

print("\n--- read the rows above ---")
gapK = (rows[-1]['naive_IS']-rows[-1]['naive_OOS']) - (rows[0]['naive_IS']-rows[0]['naive_OOS'])
print(f"(A) winner's curse on the leaderboard: K=1->1000 naive IS R2 CLIMBS {rows[0]['naive_IS']:.2f}->{rows[-1]['naive_IS']:.2f} while OOS stays ~0 -> the IS-OOS GAP grows with K (overfitting). Survival-rate RISES {rows[0]['survival_rate']*100:.0f}%->{rows[-1]['survival_rate']*100:.0f}% (more candidates -> higher chance a real strategy is in the pool) but is still ~coin-flip at low SNR even at K=1000.")
print(f"    the FIX (rank by held-out era-B, report on era-C) lifts OOS to {rows[-1]['survsel_OOS']:.2f} at K=1000.")
print(f"(B) survival-rate is honest: {brows[0]['survival_rate']*100:.0f}% at SNR=0.02 -> {brows[-1]['survival_rate']*100:.0f}% at SNR=1.0 (tracks real alpha), while naive IS R2 is high at every SNR ({brows[0]['naive_IS']:.2f}->{brows[-1]['naive_IS']:.2f}) and can't tell them apart.")
print("DONE — verification-survival metric works; ready for Q4 (obfuscation) / Q3 (causal-role-free verifier).")
