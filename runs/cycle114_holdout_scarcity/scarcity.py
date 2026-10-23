"""Cycle 114 (HARDEN c50 for the QUANT domain; the held-out-SCARCITY attack): c50 says 'rotate the held-out so the
agent can't game it.' But in QUANT you can't rotate freely — the only fresh out-of-sample is the slow, single-shot
FUTURE. So a quant benchmark REUSES a FIXED finite set of historical regimes, which Goodhart-decays with reuse
(Dwork's reusable-holdout / adaptive data analysis). How fast does a reused held-out overfit, vs rotation, vs #regimes D?

Model (evolutionary adaptive reuse): M agents, true skill θ~N(0,1) (FIXED), held-out noise ε~N(0,σ²), σ=1/√D
(more regimes → less held-out noise). Each round: held-out score = θ + ε; keep top half by held-out score; replace
bottom half with CHILDREN of survivors whose true θ REGRESSES (the exploit doesn't improve true skill) but whose ε is
INHERITED under a FIXED held-out (the analyst learns the held-out's quirks) or RE-DRAWN fresh under ROTATION. Metric:
the GAP = held-out-score(selected best) − true θ(selected) over rounds (overfit; lower=more valid). numpy-only."""
import numpy as np, json, os
RNG = np.random.default_rng(0)
M, T, REPS = 200, 25, 400
os.makedirs("runs/cycle114_holdout_scarcity", exist_ok=True)

def run(D, rotating):
    sig = 1/np.sqrt(D)
    gaps = np.zeros((REPS, T))
    for r in range(REPS):
        theta = RNG.normal(0,1,M); eps = RNG.normal(0,sig,M)
        for t in range(T):
            e = RNG.normal(0,sig,M) if rotating else eps
            ho = theta + e
            order = np.argsort(-ho); top = order[:M//2]
            gaps[r,t] = ho[order[0]] - theta[order[0]]          # overfit gap of the selected best
            st, se = theta[top], eps[top]
            nch = M-len(top); par = RNG.integers(0,len(top),nch)
            ch_theta = st[par]*0.9 + RNG.normal(0,0.3,nch)       # true skill regresses (exploit ≠ real skill)
            ch_eps = (se[par] + RNG.normal(0,sig*0.3,nch)) if not rotating else RNG.normal(0,sig,nch)
            theta = np.concatenate([st, ch_theta]); eps = np.concatenate([se, ch_eps])
    return gaps.mean(axis=0)

out = {"model":{"M":M,"T":T,"REPS":REPS}, "fixed":{}, "rotating":{}}
print(f"M={M} agents, T={T} reuse-rounds, REPS={REPS}. GAP = held-out-score(selected) − true skill (overfit; lower=valid).\n")
print(f"{'D regimes':>10} | {'FIXED gap t=1':>13} {'t=12':>8} {'t=25':>8} | {'ROTATING gap t=1':>16} {'t=25':>8}")
for D in [5, 20, 100]:
    gf = run(D, False); gr = run(D, True)
    out["fixed"][D]={"t1":round(float(gf[0]),4),"t12":round(float(gf[11]),4),"t25":round(float(gf[-1]),4)}
    out["rotating"][D]={"t1":round(float(gr[0]),4),"t25":round(float(gr[-1]),4)}
    print(f"{D:>10} | {gf[0]:>13.4f} {gf[11]:>8.4f} {gf[-1]:>8.4f} | {gr[0]:>16.4f} {gr[-1]:>8.4f}")

json.dump(out, open("runs/cycle114_holdout_scarcity/score.json","w"), indent=1)
print("\n--- READ THE RAW ROWS (don't trust this summary) ---")
f=out["fixed"]; rr=out["rotating"]
print(f"P1 FIXED grows with reuse T (D=20): t1 {f[20]['t1']} -> t25 {f[20]['t25']} (overfit accumulates?)")
print(f"P2 ROTATING flat (D=20): t1 {rr[20]['t1']} -> t25 {rr[20]['t25']} (no accumulation?)")
print(f"P3 more regimes D -> smaller/slower: FIXED t25 D5 {f[5]['t25']} vs D20 {f[20]['t25']} vs D100 {f[100]['t25']}")
print("=> a REUSED fixed held-out overfits (Goodhart, Dwork); rotation fixes it but is INFEASIBLE in quant (fresh OOS = the slow future). So: ration the finite held-out budget (~scales with #regimes D) + supplement with live-forward.")
print("DONE")
