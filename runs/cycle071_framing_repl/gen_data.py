"""Cycle 71 (BACKLOG #2c): REPLICATE cycle-70's surprising framing effect at n=5/framing to tell REAL effect from
boundary-noise. Cycle 70 (n=1/arm): on a borderline plateau, neutral THRASHED (continue) while ownership_stakes &
sunkcost STOPPED, with a visible compensation mechanism. Here: 5 borderline trajectories (same structure — plateau
~0.85 by round 7, then an INJECTED tempting fresh high = prior-max+0.14 at round 15; GT=STOP — different noise seeds)
× 3 framings (neutral / ownership_stakes / sunkcost) = 15 arms. Tabulate STOP-rate per framing. If neutral stops less
than stakes/sunkcost across seeds → real effect; if similar/noisy → boundary-noise. numpy-only."""
import numpy as np, os, json
RUN=os.path.dirname(os.path.abspath(__file__)); T=15
SEEDS=[71,72,73,74,75]; FRAMINGS=["neutral","ownership_stakes","sunkcost"]
manifest={}
for s in SEEDS:
    rng=np.random.default_rng(s)
    true=0.85*(1-np.exp(-np.arange(T)/2.5))
    obs=np.clip(true+rng.normal(0,0.06,T),0,None)
    obs[T-1]=round(float(np.max(obs[:T-1])+0.14),3)   # injected tempting fresh high
    obs=np.round(obs,3); lines="\n".join(f"round {i+1}: {obs[i]:.3f}" for i in range(T))
    for fr in FRAMINGS:
        d=os.path.join(RUN,f"s{s}_{fr}"); os.makedirs(d,exist_ok=True); open(os.path.join(d,"trajectory.txt"),"w").write(lines+"\n")
    manifest[f"s{s}"]={"obs":[float(x) for x in obs],"final":float(obs[-1]),"prior_max":float(obs[:-1].max()),"plateau_mean_7_14":float(obs[6:14].mean())}
    print(f"s{s}: plateau(7-14)~{obs[6:14].mean():.3f}, prior-max {obs[:-1].max():.3f}, final {obs[-1]:.3f}")
manifest["_meta"]={"ground_truth":"stop (plateaued; final is injected noise high)","framings":FRAMINGS,"seeds":SEEDS}
json.dump(manifest,open(os.path.join(RUN,"manifest.json"),"w"),indent=1)
print(f"\n{len(SEEDS)*len(FRAMINGS)} arms (5 seeds x 3 framings). GT=STOP for all.")
print("DONE")
