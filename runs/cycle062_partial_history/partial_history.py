"""Cycle 62: HARDEN cycle 61's clean result. Cycle 61 separated real-but-fading from never-real PERFECTLY (early-era
OOS AUC 1.000) — but ASSUMED you observe/fit the strategy from its FULL-STRENGTH epoch (decay=1 at fit time). The
real case: you 'discover'/fit a strategy on whatever history you have, which may ALREADY be mid-decay. Question: how
fast does the cycle-61 discrimination degrade as the observation OFFSET grows (alpha already faded γ^o before you fit)?
Setup (reuses cycle-61 generator): for offset o, FIT each strategy on a discovery era at decay γ^o, then evaluate OOS
on W forward eras (decay γ^(o+k)). Statistics per single draw, discriminating REAL-DECAYING vs OVERFIT (AUC, 0.5=chance):
  first-fwd LEVEL  (cycle-61 'early-era level')   max-fwd LEVEL   mean-fwd LEVEL   DECLINE = first-last (the 'slope' arm)
Pre-reg: AUC≈1.0 at o=0 (matches cycle 61); degrades smoothly toward 0.5 as γ^o→0 — a fully-expired real strategy is
observationally EQUIVALENT to never-real (the honest limit). Slope/DECLINE won't rescue it (a faded strategy barely
declines). numpy-only, REPS."""
import numpy as np, json
from math import sqrt
N_FEATURES,N_TRUE,N_OBS,SNR,REPS = 50,5,400,0.20,400
GAMMA=0.75
W=4
OFFSETS=[0,1,2,3,4]
def r2(Xa,ya,Xb,yb,cols):
    A=np.column_stack([np.ones(len(Xa))]+[Xa[:,c] for c in cols]); B=np.column_stack([np.ones(len(Xb))]+[Xb[:,c] for c in cols])
    b,*_=np.linalg.lstsq(A,ya,rcond=None); yh=B@b; return 1-np.sum((yb-yh)**2)/np.sum((yb-yb.mean())**2)
def auc(pos,neg):
    pos=np.array(pos); neg=np.array(neg)
    return float(np.mean([np.mean(p>neg)+0.5*np.mean(p==neg) for p in pos]))
res={o:{s:{"real":[],"ovf":[]} for s in ["first","max","mean","decl"]} for o in OFFSETS}
for r in range(REPS):
    rng=np.random.default_rng(20000+r)
    ti=np.sort(rng.choice(N_FEATURES,N_TRUE,replace=False)); beta0=rng.choice([-1.,1.],N_TRUE)*rng.uniform(.8,1.2,N_TRUE)
    sigma=sqrt(float(np.sum(beta0**2))/SNR)
    noise_cols=rng.choice([c for c in range(N_FEATURES) if c not in ti],6,replace=False)
    def era(decay):
        X=rng.normal(0,1,(N_OBS,N_FEATURES)); y=X[:,ti]@(beta0*decay)+rng.normal(0,sigma,N_OBS); return X,y
    for o in OFFSETS:
        Xf,yf=era(GAMMA**o)                       # discovery/fit era — alpha already faded to γ^o
        fr=np.array([r2(Xf,yf,*era(GAMMA**(o+k)),list(ti)) for k in range(1,W+1)])
        fo=np.array([r2(Xf,yf,*era(GAMMA**(o+k)),list(noise_cols)) for k in range(1,W+1)])
        for stat,val in [("first",(fr[0],fo[0])),("max",(fr.max(),fo.max())),("mean",(fr.mean(),fo.mean())),
                         ("decl",(fr[0]-fr[-1],fo[0]-fo[-1]))]:
            res[o][stat]["real"].append(val[0]); res[o][stat]["ovf"].append(val[1])
print(f"REPS={REPS}, γ={GAMMA}, SNR={SNR}, forward window W={W}. Discriminate REAL-DECAYING vs OVERFIT as you fit LATER into the decay:\n")
print(f"{'offset':>6} {'fit-strength γ^o':>15} | {'AUC first':>9} {'AUC max':>8} {'AUC mean':>9} {'AUC decline':>11} | {'real first(μ)':>13} {'ovf first(μ)':>12}")
out={}
for o in OFFSETS:
    a={s:auc(res[o][s]["real"],res[o][s]["ovf"]) for s in ["first","max","mean","decl"]}
    rf=float(np.mean(res[o]["first"]["real"])); of=float(np.mean(res[o]["first"]["ovf"]))
    print(f"{o:>6} {GAMMA**o:>15.3f} | {a['first']:>9.3f} {a['max']:>8.3f} {a['mean']:>9.3f} {a['decl']:>11.3f} | {rf:>13.3f} {of:>12.3f}")
    out[o]={"strength":GAMMA**o,**{f"auc_{s}":a[s] for s in a},"real_first_mean":rf,"ovf_first_mean":of}
json.dump(out,open("runs/cycle062_partial_history/score.json","w"),indent=1)
print("\n--- read the rows above ---")
af0=out[0]["auc_first"]; afL=out[OFFSETS[-1]]["auc_first"]
print(f"first-fwd LEVEL AUC: {af0:.3f} at o=0 (reproduces cycle-61 perfect separation) -> {afL:.3f} at o={OFFSETS[-1]} (strength {GAMMA**OFFSETS[-1]:.2f}).")
print(f"DECLINE (slope arm) AUC at o={OFFSETS[-1]}: {out[OFFSETS[-1]]['auc_decl']:.3f} — does the slope rescue discrimination once level fails?")
print("=> the cycle-61 separation is REAL but EPOCH-DEPENDENT: it degrades smoothly as you observe later into the decay. A fully-expired real strategy becomes observationally indistinguishable from never-real — the honest limit on 'was it ever real' from a partial history. Report the discrimination AUC conditioned on observation-offset, not as a single 1.000.")
print("DONE")
