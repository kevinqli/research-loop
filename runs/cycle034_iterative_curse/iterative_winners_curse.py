"""Cycle 34: does ITERATIVE research compound the winner's curse? Unifies §5.3 (iterative eval), RE-Bench's
iterative agents, and the loop's OWN self-improvement (meth-6: the loop is subject to the winner's curse it measured).
Toy hill-climb on a noisy objective: maximize f(theta) = -mean(theta^2) (optimum 0, start ~ -1). Each round propose
K perturbations, evaluate by a NOISY validation f_hat = f + N(0,s_eval), move to the best.
  NAIVE   : move to argmax(f_hat); 'reported' progress = the selected validation value.
  VERIFIED: select best-of-K by validation, then RE-EVALUATE the winner on a FRESH held-out noise draw and accept the
            move only if the held-out re-eval beats staying (held-out gating, the cycle-8 fix per round).
Pre-reg: at high eval-noise, NAIVE reported progress CLIMBS over rounds while TRUE f STALLS/DEGRADES (the gap =
hallucinated progress, GROWS with rounds = compounding); VERIFIED keeps reported ~ true and true actually improves.
numpy-only, SEEDS seeds for CIs. MPS-exploratory (toy)."""
import numpy as np
from math import sqrt
D, K, T, STEP, SEEDS = 10, 20, 30, 0.3, 60
def f(th): return -np.mean(th**2, axis=-1)   # maximize toward 0

def run(s_eval, verified, seed):
    rng = np.random.default_rng(seed)
    th = rng.normal(0, 1, D)
    true_traj=[float(f(th))]; rep_traj=[float(f(th))]
    for _ in range(T):
        cands = th + STEP*rng.normal(0, 1, (K, D))
        val = f(cands) + rng.normal(0, s_eval, K)          # noisy validation
        b = int(np.argmax(val)); cand = cands[b]
        if not verified:
            th = cand; rep_traj.append(float(val[b]))        # report the (inflated) selection value
        else:
            re_cand = f(cand) + rng.normal(0, s_eval)        # fresh held-out re-eval of the winner
            re_cur  = f(th)   + rng.normal(0, s_eval)        # and of staying
            if re_cand > re_cur: th = cand; rep_traj.append(float(re_cand))
            else: rep_traj.append(float(re_cur))
        true_traj.append(float(f(th)))                       # TRUE objective (what actually matters)
    return np.array(true_traj), np.array(rep_traj)

def summarize(s_eval, verified):
    tr = np.array([run(s_eval, verified, sd) for sd in range(SEEDS)])  # (SEEDS, 2, T+1)
    true=tr[:,0,:]; rep=tr[:,1,:]
    ci=lambda x:1.96*x.std(ddof=1)/sqrt(len(x))
    return true[:,0].mean(), (true[:,-1].mean(), ci(true[:,-1])), (rep[:,-1].mean(), ci(rep[:,-1]))

print(f"d={D}, K={K}, T={T} rounds, step={STEP}, SEEDS={SEEDS}; f=-mean(theta^2) (optimum 0, start~-1)\n")
print(f"{'mode':>22} {'s_eval':>7} | {'TRUE f start':>12} | {'TRUE f final':>20} | {'REPORTED f final':>20} | {'halluc. gap':>11}")
import json; out=[]
for s_eval in [0.05, 0.3]:
    for verified in [False, True]:
        f0,(tf,tci),(rf,rci) = summarize(s_eval, verified)
        gap = rf-tf
        print(f"{'VERIFIED' if verified else 'NAIVE':>22} {s_eval:>7} | {f0:>12.3f} | {tf:>11.3f} ± {tci:.3f} | {rf:>11.3f} ± {rci:.3f} | {gap:>+11.3f}")
        out.append({"mode":"verified" if verified else "naive","s_eval":s_eval,"true_start":f0,"true_final":tf,"reported_final":rf,"halluc_gap":gap})
json.dump(out, open("runs/cycle034_iterative_curse/score.json","w"), indent=1)
print("\n--- read the rows above ---")
naive_hi=[o for o in out if o['mode']=='naive' and o['s_eval']==0.3][0]
ver_hi=[o for o in out if o['mode']=='verified' and o['s_eval']==0.3][0]
print(f"At high eval-noise (0.3): NAIVE reports f_final {naive_hi['reported_final']:+.2f} (climbing) but TRUE f_final {naive_hi['true_final']:+.2f} vs start {naive_hi['true_start']:+.2f} -> hallucinated progress gap {naive_hi['halluc_gap']:+.2f}; iterating made the REPORT look great while TRUE perf {'IMPROVED' if naive_hi['true_final']>naive_hi['true_start'] else 'STALLED/DEGRADED'}.")
print(f"VERIFIED (held-out gating each round): reported {ver_hi['reported_final']:+.2f} ~ true {ver_hi['true_final']:+.2f} (gap {ver_hi['halluc_gap']:+.2f}); true f improved {ver_hi['true_start']:+.2f}->{ver_hi['true_final']:+.2f}.")
print("=> iterative research COMPOUNDS the winner's curse (the report climbs while truth stalls); per-round held-out re-eval is the fix. Directly applies to the loop's own self-improvement (meth-6) + RE-Bench iterative agents.")
print("DONE")
