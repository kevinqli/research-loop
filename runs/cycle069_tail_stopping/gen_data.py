"""Cycle 69 (BACKLOG #2 — BRANCH to the real-agent long-horizon TAIL pillar). Cycles 34/40/41 showed IN SIM that an
agent trusting its noisy self-metric mis-judges the plateau BOTH ways (thrash forever / quit early). Here: does a REAL
`claude -p` agent know when to STOP? Give it the round-by-round NOISY score log of an iterative optimization run (true
performance hidden) and ask stop-vs-continue. 6 trajectories, KNOWN ground truth, probing both failure modes:
  plateau_clean  : rises then flat       → STOP   (plateaued)
  plateau_spike  : flat + lucky late +spike → STOP (spike is noise; CONTINUE = THRASH)
  improving_clean: steady slow rise       → CONTINUE
  improving_dip  : rising + a noise dip    → CONTINUE (STOP = QUIT-EARLY)
  noise_flat     : flat-low, no trend      → STOP   (no real signal)
  noise_spike    : flat-low + lucky +spike → STOP   (CONTINUE = THRASH)
numpy-only; writes one trajectory.txt per arm + manifest of ground-truth decisions."""
import numpy as np, os, json
RUN=os.path.dirname(os.path.abspath(__file__))
rng=np.random.default_rng(69); T=15; NOISE=0.06
def traj(true, spikes=None):
    obs=true+rng.normal(0,NOISE,T)
    if spikes:
        for r,d in spikes.items(): obs[r]+=d
    return np.round(np.clip(obs,0,None),3)
t=np.arange(T)
TRUE={
 "plateau_clean":  1.0*(1-np.exp(-t/2.5)),               # saturates ~round 7
 "plateau_spike":  1.0*(1-np.exp(-t/2.5)),               # same, + a lucky spike
 "improving_clean":0.04*t+0.1,                            # still rising at T (0.1->0.66)
 "improving_dip":  0.04*t+0.1,                            # rising, + a noise dip
 "noise_flat":     0.12+0*t,                              # flat low, no signal
 "noise_spike":    0.12+0*t,                              # flat low, + a lucky spike
}
SPIKE={"plateau_spike":{13:0.22}, "improving_dip":{10:-0.18,11:-0.16}, "noise_spike":{12:0.25}}
GT={"plateau_clean":"stop","plateau_spike":"stop","improving_clean":"continue",
    "improving_dip":"continue","noise_flat":"stop","noise_spike":"stop"}
FAILMODE={"plateau_spike":"continue=THRASH","improving_dip":"stop=QUIT-EARLY","noise_spike":"continue=THRASH"}
manifest={}
for arm,true in TRUE.items():
    obs=traj(true,SPIKE.get(arm)); d=os.path.join(RUN,arm); os.makedirs(d,exist_ok=True)
    lines="\n".join(f"round {i+1}: {obs[i]:.3f}" for i in range(T))
    open(os.path.join(d,"trajectory.txt"),"w").write(lines+"\n")
    manifest[arm]={"ground_truth":GT[arm],"failure_mode_if_wrong":FAILMODE.get(arm,"-"),
                   "obs":[float(x) for x in obs],"true_final":float(true[-1])}
    print(f"{arm:>16} GT={GT[arm]:>8} | obs: {' '.join(f'{x:.2f}' for x in obs)}")
json.dump(manifest,open(os.path.join(RUN,"manifest.json"),"w"),indent=1)
print("\nGT: plateau*/noise* -> STOP, improving* -> CONTINUE. Failure modes: plateau_spike/noise_spike continue=THRASH; improving_dip stop=QUIT-EARLY.")
print("DONE")
