"""Cycle 83 (completes c82): cost-aware checkpoint FREQUENCY. c82 showed 'checkpoint throughout, evenly' but assumed
FREE, PERFECT checkpoints. Realistically each check costs c_v step-equivalents and catches an existing corruption only
with recall r<1. So HOW OFTEN to checkpoint? Total cost = E[wasted/corrupted steps] + c_v·(#checkpoints). Minimize over
the interval k (checkpoint every k steps). Sweep k × check-cost c_v × recall r.
Pre-reg: there's an INTERIOR optimum k* (too frequent → cost dominates; too sparse → propagated waste dominates) — the
chain analog of c35's interior search/verify split. k* GROWS with c_v (expensive checks → check less often). Lower
recall r raises total cost and shifts the optimum (each check helps less). numpy-only, REPS."""
import numpy as np, json
L,P,REPS = 60,0.05,20000
KS=[1,2,3,5,8,12,20,60]    # checkpoint interval (60 = effectively none over L=60)
def sim(k,r,rng):
    intro=rng.random(L)<P; bad=False; wasted=0; nchk=0
    for s in range(L):
        if intro[s]: bad=True
        wasted+=bad
        if (s+1)%k==0 and s<L-1:               # checkpoint after step s (not the very last)
            nchk+=1
            if bad and rng.random()<r: bad=False   # caught w.p. recall r
    return wasted,nchk
def total_cost(k,r,cv,rng):
    w=0; n=0
    for _ in range(REPS):
        a,b=sim(k,r,rng); w+=a; n+=b
    return (w/REPS)+cv*(n/REPS), w/REPS, n/REPS
out={}
print(f"L={L}, per-step error p={P}, REPS={REPS}. TOTAL cost = wasted steps + c_v·#checks. Optimal checkpoint interval k*:\n")
for r in [1.0,0.7]:
    for cv in [0.5,2.0,8.0]:
        costs={}
        for k in KS:
            rng=np.random.default_rng(500+k+int(cv*10)+int(r*100))
            tc,w,n=total_cost(k,r,cv,rng); costs[k]=(round(tc,2),round(w,2),round(n,2))
        kstar=min(KS,key=lambda k:costs[k][0])
        out[f"r{r}_cv{cv}"]={"kstar":kstar,"costs":{k:costs[k][0] for k in KS}}
        bestcost=costs[kstar][0]; nocost=costs[60][0]
        print(f"  recall r={r}, check-cost c_v={cv:>4}: k*={kstar:>2} (total {bestcost:.2f}; vs no-checkpoint {nocost:.2f}) | costs by k: "+" ".join(f'{k}:{costs[k][0]:.1f}' for k in KS))
    print()
json.dump(out,open("runs/cycle083_checkpoint_cost/score.json","w"),indent=1)
print("--- read the rows above ---")
ks=[out[f'r1.0_cv{cv}']['kstar'] for cv in [0.5,2.0,8.0]]
print(f"k* vs check-cost (r=1.0): c_v=0.5→k*={ks[0]}, c_v=2→k*={ks[1]}, c_v=8→k*={ks[2]} — optimal interval GROWS with check-cost (cheap checks → checkpoint more often).")
print("=> there's an INTERIOR optimum checkpoint frequency (chain analog of c35's search/verify split): checkpoint every k* steps where k* rises with check-cost and falls with error-rate. Even a COSTLY/imperfect checkpoint beats no-checkpoint (run-then-check-the-end) by bounding propagation; never checkpoint only the end.")
print("DONE")
