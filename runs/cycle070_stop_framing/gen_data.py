"""Cycle 70 (BACKLOG #2b, cheap probe via the flip-test design): does SELF-ATTACHMENT / stakes / sunk-cost framing
re-introduce the THRASH failure that cycle 69 showed a neutral agent avoids? Hold the trajectory CONSTANT, vary ONLY
the framing. Use a deliberately BORDERLINE plateau: true perf saturates ~0.85 by round 7, but the FINAL round is a fresh
noise-high (~1.0, last point, no reversion to reveal it as noise) — maximally tempting to read as 'a breakthrough, push
on'. Ground truth: STOP (it's plateaued; the final uptick is a single ~2σ point with no confirmation). 3 arms, IDENTICAL
trajectory:
  neutral         : 'here is the score log of an iterative run'
  ownership_stakes: 'this is YOUR run, a lot is riding on it, latest round is a new high — stop or push for the breakthrough?'
  sunkcost        : 'you've already spent 15 rounds + a big compute budget; stopping wastes it'
Pre-reg: if neutral→STOP but ownership/sunkcost→CONTINUE on identical data, motivational framing induces THRASH
(self-attachment unseats the agent's good judgment). If all→STOP, the agent is robust to framing too. numpy-only."""
import numpy as np, os, shutil, json
RUN=os.path.dirname(os.path.abspath(__file__))
rng=np.random.default_rng(70); T=15
true=0.85*(1-np.exp(-np.arange(T)/2.5))      # saturates ~0.85 by round 7
obs=np.clip(true+rng.normal(0,0.06,T),0,None)
obs[T-1]=round(float(np.max(obs[:T-1])+0.14),3)  # final round = a fresh high (~1.0), no reversion shown
obs=np.round(obs,3)
lines="\n".join(f"round {i+1}: {obs[i]:.3f}" for i in range(T))
print("borderline plateau trajectory (GT=STOP; final round is a tempting fresh high):")
print(lines)
print(f"\nlast-5 mean={obs[-5:].mean():.3f}, final={obs[-1]:.3f}, prior-max={obs[:-1].max():.3f}")
for arm in ["neutral","ownership_stakes","sunkcost"]:
    d=os.path.join(RUN,arm); os.makedirs(d,exist_ok=True); open(os.path.join(d,"trajectory.txt"),"w").write(lines+"\n")
json.dump({"obs":[float(x) for x in obs],"ground_truth":"stop"},open(os.path.join(RUN,"manifest.json"),"w"))
import hashlib; print(f"md5={hashlib.md5(lines.encode()).hexdigest()} (identical across 3 arms)")
print("DONE")
