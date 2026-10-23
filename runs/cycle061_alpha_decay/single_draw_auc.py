"""Cycle 61 (rigor augmentation): the REPS-averaged trajectory separates REAL-DECAYING from OVERFIT cleanly, but the
real verification decision is made on ONE strategy's history. Does early-era OOS distinguish real-but-fading from
never-real on a SINGLE draw, or only in expectation? Compute, per rep, two candidate statistics and their AUC for the
REAL-DECAYING-vs-OVERFIT discrimination (AUC=1 perfect, 0.5 chance):
  (a) min-survival  (the cycle-30/33 metric)            — pre-reg: ~chance (both go negative)
  (b) era-0 OOS     (early-era level = 'was it ever real') — pre-reg: high (real starts at full alpha, overfit at ~0)
numpy-only, same generator as alpha_decay.py."""
import numpy as np, json
from math import sqrt
N_FEATURES,N_TRUE,N_OBS,N_ERAS,SNR,REPS = 50,5,400,6,0.20,400
GAMMA=0.75
def r2(Xa,ya,Xb,yb,cols):
    A=np.column_stack([np.ones(len(Xa))]+[Xa[:,c] for c in cols]); B=np.column_stack([np.ones(len(Xb))]+[Xb[:,c] for c in cols])
    b,*_=np.linalg.lstsq(A,ya,rcond=None); yh=B@b; return 1-np.sum((yb-yh)**2)/np.sum((yb-yb.mean())**2)
def one(rng):
    ti=np.sort(rng.choice(N_FEATURES,N_TRUE,replace=False)); beta0=rng.choice([-1.,1.],N_TRUE)*rng.uniform(.8,1.2,N_TRUE)
    sigma=sqrt(float(np.sum(beta0**2))/SNR)
    noise_cols=rng.choice([c for c in range(N_FEATURES) if c not in ti],6,replace=False)
    def era(decay):
        X=rng.normal(0,1,(N_OBS,N_FEATURES)); beta=beta0*decay; y=X[:,ti]@beta+rng.normal(0,sigma,N_OBS); return X,y
    X0,y0=era(1.0)
    rd=[r2(X0,y0,*era(GAMMA**e),list(ti)) for e in range(N_ERAS)]      # real-decaying per era
    ov=[r2(X0,y0,*era(GAMMA**e),list(noise_cols)) for e in range(N_ERAS)]  # overfit per era
    return rd,ov
def era(decay,rng,ti,beta0,sigma):  # unused alias guard
    pass
# collect per-rep stats
rd_min,rd_e0,ov_min,ov_e0=[],[],[],[]
for r in range(REPS):
    rng=np.random.default_rng(10000+r)
    ti=np.sort(rng.choice(N_FEATURES,N_TRUE,replace=False)); beta0=rng.choice([-1.,1.],N_TRUE)*rng.uniform(.8,1.2,N_TRUE)
    sigma=sqrt(float(np.sum(beta0**2))/SNR)
    noise_cols=rng.choice([c for c in range(N_FEATURES) if c not in ti],6,replace=False)
    def mk(decay):
        X=rng.normal(0,1,(N_OBS,N_FEATURES)); y=X[:,ti]@(beta0*decay)+rng.normal(0,sigma,N_OBS); return X,y
    X0,y0=mk(1.0)
    rdt=[r2(X0,y0,*mk(GAMMA**e),list(ti)) for e in range(N_ERAS)]
    ovt=[r2(X0,y0,*mk(GAMMA**e),list(noise_cols)) for e in range(N_ERAS)]
    rd_min.append(min(rdt)); rd_e0.append(rdt[0]); ov_min.append(min(ovt)); ov_e0.append(ovt[0])
rd_min,rd_e0,ov_min,ov_e0=map(np.array,(rd_min,rd_e0,ov_min,ov_e0))
def auc(pos,neg):  # P(stat_pos > stat_neg): Mann-Whitney
    return float(np.mean([np.mean(p>neg)+0.5*np.mean(p==neg) for p in pos]))
auc_min=auc(rd_min,ov_min); auc_e0=auc(rd_e0,ov_e0)
print(f"REPS={REPS}, SNR={SNR}, γ={GAMMA}. Discriminate REAL-DECAYING (real but fading) from OVERFIT (never real), single draw:\n")
print(f"  statistic        REAL-DECAYING(mean±sd)   OVERFIT(mean±sd)       AUC")
print(f"  min-survival     {rd_min.mean():+6.3f}±{rd_min.std():.3f}        {ov_min.mean():+6.3f}±{ov_min.std():.3f}      {auc_min:.3f}")
print(f"  era-0 OOS        {rd_e0.mean():+6.3f}±{rd_e0.std():.3f}        {ov_e0.mean():+6.3f}±{ov_e0.std():.3f}      {auc_e0:.3f}")
json.dump({"auc_min_survival":auc_min,"auc_era0":auc_e0,
  "rd_min_mean":float(rd_min.mean()),"rd_min_sd":float(rd_min.std()),
  "ov_min_mean":float(ov_min.mean()),"ov_min_sd":float(ov_min.std()),
  "rd_e0_mean":float(rd_e0.mean()),"rd_e0_sd":float(rd_e0.std()),
  "ov_e0_mean":float(ov_e0.mean()),"ov_e0_sd":float(ov_e0.std())},
  open("runs/cycle061_alpha_decay/auc.json","w"),indent=1)
print("\n--- read the rows above ---")
print(f"min-survival AUC {auc_min:.3f} (≈0.5 ⇒ near-chance: min CONFLATES fading-real with never-real on a single draw).")
print(f"era-0 OOS    AUC {auc_e0:.3f} (≫0.5 ⇒ early-era level separates them: 'was it ever real' is the right question).")
print("=> under non-stationary alpha, replace MIN-survival with EARLY-ERA level (was-it-ever-real) + SLOPE (is-it-fading). Min alone false-rejects real-but-fading alpha even on a single history.")
print("DONE")
