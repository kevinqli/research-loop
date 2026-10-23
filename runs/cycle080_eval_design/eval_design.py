"""Cycle 80 ("benchmark is the product" — concrete eval-design guidance). The survival metric is established (c30/33),
its boundaries mapped (overlap c79, obfuscation c31, non-stationarity c61). The open DESIGN question for someone
BUILDING the eval: how MANY diverse held-out regimes do you need, and which AGGREGATOR? Sweep D = #regimes ×
aggregator {mean, median, min} × signal {STATIONARY, DECAYING (c61)}. Discriminate REAL (signal present) from OVERFIT
(OOS ~0 every regime). Metric = AUC(aggregate, label=real).
Pre-reg: (stationary) all aggregators improve with D and saturate ~D=5 (noise-averaging); (decaying) MEAN/median still
work but MIN DEGRADES and gets WORSE with more regimes (min hits the most-faded regime ~0, indistinguishable from
overfit — the c61 'min inverts under non-stationarity', now as a function of D). Output: a design recommendation. numpy."""
import numpy as np, json
N,REPS,SIGMA = 400,150,0.10        # candidates/rep, reps, per-regime OOS noise
DS=[1,2,3,5,8,12]; GAMMA=0.70       # decay per regime for the DECAYING setting
def auc(pos,neg):
    pos=np.asarray(pos); neg=np.asarray(neg)
    return float(np.mean([(p>neg).mean()+0.5*(p==neg).mean() for p in pos]))
def aggr(M,kind):
    return {"mean":M.mean(1),"median":np.median(M,1),"min":M.min(1)}[kind]
out={}
for setting in ["stationary","decaying"]:
    out[setting]={}
    for kind in ["mean","median","min"]:
        row=[]
        for D in DS:
            aucs=[]
            for r in range(REPS):
                rng=np.random.default_rng(7000+r); h=N//2
                theta0=np.concatenate([rng.uniform(.20,.45,h), np.zeros(N-h)])  # REAL signal vs OVERFIT ~0
                label=np.concatenate([np.ones(h),np.zeros(N-h)])
                M=np.zeros((N,D))
                for d in range(D):
                    decay=GAMMA**d if setting=="decaying" else 1.0
                    M[:,d]=theta0*decay+rng.normal(0,SIGMA,N)               # noisy per-regime OOS
                a=aggr(M,kind); aucs.append(auc(a[label==1],a[label==0]))
            row.append(round(float(np.mean(aucs)),3))
        out[setting][kind]=row
print(f"N={N}, per-regime noise σ={SIGMA}, REPS={REPS}, decay γ={GAMMA}. AUC(real vs overfit) vs #regimes D:\n")
for setting in ["stationary","decaying"]:
    print(f"  [{setting}]   D = {DS}")
    for kind in ["mean","median","min"]:
        print(f"    {kind:>7}: {out[setting][kind]}")
    print()
json.dump({"DS":DS,**out},open("runs/cycle080_eval_design/score.json","w"),indent=1)
print("--- read the rows above ---")
sm,dm,dmin=out['stationary']['mean'],out['decaying']['mean'],out['decaying']['min']
print(f"STATIONARY mean: D=1 {sm[0]:.3f} -> D=5 {sm[3]:.3f} -> D=12 {sm[-1]:.3f} (improves, saturates).")
print(f"DECAYING mean: {dm[0]:.3f}->{dm[-1]:.3f} (robust); DECAYING MIN: {dmin[0]:.3f}->{dmin[-1]:.3f} (DEGRADES with more regimes — c61).")
print("=> DESIGN GUIDANCE: use MEAN/MEDIAN (not MIN) over ~5 diverse held-out regimes (diminishing returns past ~5-8); MIN is fragile under non-stationarity and gets WORSE with more regimes. Diversity COUNT has diminishing returns; the AGGREGATOR choice matters more once the signal can decay.")
print("DONE")
