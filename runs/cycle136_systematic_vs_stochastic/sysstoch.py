"""Cycle 136 (HARDEN c135 + SYNTHESIS, ties c135×c118×c45×c133 — BACKLOG #20): the stochastic-vs-SYSTEMATIC taxonomy of
long-horizon failure. c135 showed decompose-verify-RETRY converts the exponential to linear — but it assumed every per-step
error is STOCHASTIC (a retry eventually succeeds). The genuinely-hard case is a SYSTEMATIC error / a "wall": a step the
agent conceptually can't do, which fails IDENTICALLY every attempt and in every sample. Retry can't cross a wall; a verifier
FLAGS it but can't CROSS it; consensus (c133) AGREES on it (shared blind spot, false consensus). What CAN cross a wall?
Only better GENERATION (raise per-step capability → lower the wall rate, c118).

Model: L steps. Each step is a WALL (systematic, fails forever) w.p. a, else STOCHASTIC with per-attempt failure b.
  DECOMPOSE-VERIFY-RETRY success = (1-a)^L  (succeed iff no walls; stochastic retried away — depends ONLY on a, not b).
  RAW SAMPLING (N samples) coverage = (1-a)^L · [1 - (1-(1-b)^L)^N]  (no walls AND ≥1 sample gets all stochastic right).
PRE-REG: P1 decompose=(1-a)^L (b-independent). P2 both CAPPED at the wall ceiling (1-a)^L; decompose removes only the
stochastic exponential (1-b)^L (a constant multiplicative edge over raw, in a). P3 as a rises both collapse → decompose's
edge becomes moot; the ONLY lever for systematic errors is lowering a (generation, c118), not retry/verify/consensus."""
import numpy as np, json, os
RNG = np.random.default_rng(0)
os.makedirs("runs/cycle136_systematic_vs_stochastic", exist_ok=True)
L, B, N = 50, 0.10, 30

def decompose(a, L=L): return (1-a)**L
def raw(a, L=L, b=B, N=N):
    no_wall = (1-a)**L; stoch_cov = 1 - (1 - (1-b)**L)**N
    return no_wall * stoch_cov

print(f"L={L} steps, stochastic per-attempt failure b={B}, raw budget N={N} samples.")
print(f"stochastic single-sample all-success (1-b)^L = {(1-B)**L:.4f}; raw covers it at N={N}: {1-(1-(1-B)**L)**N:.3f}\n")
print("=== success vs SYSTEMATIC (wall) rate a ===")
print(f"{'a (wall rate)':>13} | {'wall ceiling (1-a)^L':>20} | {'DECOMPOSE-verify':>16} | {'RAW (N samples)':>15} | {'decompose/raw':>13}")
rows=[]
for a in [0.0, 0.005, 0.01, 0.02, 0.05, 0.1]:
    d=decompose(a); r=raw(a); ratio=d/r if r>0 else float('inf')
    rows.append({"a":a,"ceiling":round((1-a)**L,4),"decompose":round(d,4),"raw":round(r,4),"ratio":(round(ratio,2) if np.isfinite(ratio) else None)})
    print(f"{a:>13} | {round((1-a)**L,4):>20} | {round(d,4):>16} | {round(r,4):>15} | {round(ratio,2):>13}")

print("\n=== MC confirmation (a=0.02): build trajectories, count decompose-success vs raw-coverage ===")
a=0.02; reps=20000
walls = RNG.random((reps, L)) < a                       # which steps are walls (fixed per task/rep)
no_wall_traj = ~walls.any(axis=1)                       # decompose succeeds iff no walls (stochastic retried away)
dec_mc = no_wall_traj.mean()
# raw: N independent samples; a sample succeeds iff (no walls) AND all non-wall steps pass their stochastic roll
samp = RNG.random((reps, N, L)) >= B                    # stochastic pass per (rep,sample,step)
ok_steps = samp & (~walls[:,None,:])                    # a step is ok in a sample iff passes AND not a wall
samp_ok = ok_steps.all(axis=2)                          # sample fully correct?
raw_mc = (samp_ok.any(axis=1) & no_wall_traj).mean() if True else 0
# (samp_ok already requires non-wall steps to pass; wall steps -> ok_steps False -> sample fails, so any-sample success already implies no-wall)
raw_mc = samp_ok.any(axis=1).mean()
print(f"a=0.02: decompose MC={dec_mc:.4f} (analytic {(1-a)**L:.4f}); raw MC={raw_mc:.4f} (analytic {raw(a):.4f})")

print("\n=== P3: the only lever for SYSTEMATIC errors is lowering a (generation, c118) — wall ceiling (1-a)^L vs a, L=50 ===")
for a in [0.10, 0.05, 0.02, 0.01, 0.005]:
    print(f"  halve the wall rate a {a}->{a/2}: ceiling (1-a)^L {(1-a)**L:.4f} -> {(1-a/2)**L:.4f}  (better GENERATION lifts the ceiling; retry/verify/consensus can't)")

json.dump({"L":L,"B":B,"N":N,"vs_a":rows,"mc":{"a":0.02,"dec":round(float(dec_mc),4),"raw":round(float(raw_mc),4)}}, open("runs/cycle136_systematic_vs_stochastic/score.json","w"), indent=1)
print("\n--- READ THE RAW ROWS ---")
print("decompose=(1-a)^L: " + " ".join(f"a{r['a']}:{r['decompose']}" for r in rows))
print("raw (N=30):       " + " ".join(f"a{r['a']}:{r['raw']}" for r in rows))
print("decompose/raw:    " + " ".join(f"a{r['a']}:{r['ratio']}" for r in rows) + "  (~constant multiplicative edge = the stochastic exponential decompose removes)")
print("=> TAXONOMY: STOCHASTIC errors (b) -> decompose-verify-RETRY (c135) removes the exponential; SYSTEMATIC errors / walls (a) -> retry, sampling, verification AND consensus (c133) all FAIL (you flag/agree-on a wall but can't cross it); the ONLY lever is better GENERATION (lower a, c118). Decompose-verify is necessary-but-insufficient: it fixes the stochastic exponential, but the systematic (1-a)^L is the irreducible tail.")
print("DONE")
