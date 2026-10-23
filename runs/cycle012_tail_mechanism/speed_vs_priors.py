"""Test cycle-11's diagnosis of the RE-Bench crossover: does it emerge from AGENT SPEED (early lead) x HUMAN PRIORS
+ higher CEILING (late lead) — rather than explore-vs-exploit? Landscape = broad EASY basin (cap 0.85, fast to
climb) + a NARROW GLOBAL peak (1.0) far away that fast random local search misses.
  AGENT: fast (SPEED_A steps per time-unit), RANDOM start -> lands in the easy basin, climbs to ~0.85 fast, capped.
  HUMAN: slow (SPEED_H steps/unit) BUT good PRIOR -> starts inside the narrow global basin; climbs to 1.0, slowly.
Measure best-so-far vs time-budget T; crossover? 300 landscapes. numpy only."""
import numpy as np
rng=np.random.default_rng(0); LAND=300; Ts=[1,2,4,8,16,32]; SPEED_A=8; SPEED_H=1
def f(x): return max(0.85*np.exp(-((x-0.30)/0.20)**2), 1.0*np.exp(-((x-0.80)/0.02)**2))  # easy broad + narrow global
def climb(x0,nsteps,step):
    x=x0; b=f(x)
    for _ in range(nsteps):
        c=np.clip(x+rng.normal(0,step),0,1); fc=f(c)
        if fc>b: x,b=c,fc
    return b
def agent(T): return climb(rng.uniform(0,1), SPEED_A*T, 0.03)            # fast, random start (weak prior)
def human(T): return climb(0.80+rng.normal(0,0.02), SPEED_H*T, 0.01)     # slow, prior = starts in global basin
res={T:{"agent":[],"human":[]} for T in Ts}
for _ in range(LAND):
    for T in Ts: res[T]["agent"].append(agent(T)); res[T]["human"].append(human(T))
print(f"landscapes={LAND}, agent SPEED={SPEED_A}/T (random start), human SPEED={SPEED_H}/T (prior start in global)\n  T |  agent |  human | winner")
cross=None
for T in Ts:
    a=np.mean(res[T]["agent"]); h=np.mean(res[T]["human"]); w="agent" if a>h else "HUMAN"
    if cross is None and h>a: cross=T
    print(f"  {T:>2} | {a:.4f} | {h:.4f} | {w}")
print(f"\n  CROSSOVER: human overtakes agent at T~{cross} (agent wins short via speed, human wins long via priors+ceiling)"
      if cross and cross>Ts[0] else f"\n  NO clean early-agent crossover (human leads from T={Ts[0]}) -> diagnosis needs refinement")
