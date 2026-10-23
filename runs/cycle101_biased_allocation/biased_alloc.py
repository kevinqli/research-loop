"""Cycle 101 (HARDEN c97): does ADAPTIVE (successive-halving) verification allocation survive BIASED early estimates?
c97 showed SH > uniform under UNBIASED noise. But SH eliminates candidates on FEW early pulls — if early evals are
systematically biased (cold-start: a slow-starting-but-truly-best candidate looks bad early; or a cheap early-eval
that underrates "deep" candidates), SH may drop the true-best in round 0. Uniform gives every candidate the full
budget, so if its per-arm pulls exceed the warmup length it sees past the bias.

Model: K candidates, true theta~N(0,1). A candidate's first W pulls are WARMUP-biased: observed true-component =
theta*(1 - dscale) during warmup, theta after. dscale sweeps the cold-start severity: 0=none (c97), 0.5=compressed
(rank preserved), 1=early signal zeroed, 1.5=early signal INVERTED (best looks worst). Methods: UNIFORM (B/K pulls
each, mean), NAIVE-SH (c97, cumulative-mean halving), WARMUP-AWARE-SH (spend K*W warming everyone past the bias-
discarded, then clean SH on the rest). Metric: regret (theta_oracle - theta_selected) + P(correct). numpy-only, REPS."""
import numpy as np, json, os
RNG = np.random.default_rng(0)
REPS, K, SIG, B, W = 20000, 32, 1.0, 320, 4         # B=10/arm so uniform (10) sees past warmup (W=4)
R = int(np.ceil(np.log2(K)))
os.makedirs("runs/cycle101_biased_allocation", exist_ok=True)

def uniform(theta, dscale, budget=B):
    n = budget//K; wp = min(n, W)
    tmean = theta*(wp*(1-dscale)+(n-wp))/n                       # true-component of the n-pull mean
    obs = tmean + RNG.normal(0, SIG/np.sqrt(n), theta.shape)
    return obs.argmax(1)

def sh(theta, dscale, budget=B):
    reps = theta.shape[0]
    cur = np.tile(np.arange(K), (reps,1))
    psum = np.zeros((reps,K)); nsum = np.zeros((reps,K)); pcnt = np.zeros((reps,K))
    while cur.shape[1] > 1:
        m = cur.shape[1]; n_r = max(budget//(R*m), 1)
        idx = cur; th = np.take_along_axis(theta, idx, 1)
        pc_old = np.take_along_axis(pcnt, idx, 1); pc_new = pc_old + n_r
        wnew = np.clip(np.minimum(pc_new, W) - pc_old, 0, None)  # warmup pulls among the new n_r
        true_add = th*(wnew*(1-dscale) + (n_r - wnew))
        noise_add = RNG.normal(0, SIG*np.sqrt(n_r), (reps, m))   # sum of n_r iid noises
        ps_old = np.take_along_axis(psum, idx, 1); ns_old = np.take_along_axis(nsum, idx, 1)
        np.put_along_axis(psum, idx, ps_old+true_add, 1)
        np.put_along_axis(nsum, idx, ns_old+noise_add, 1)
        np.put_along_axis(pcnt, idx, pc_new, 1)
        means = (ps_old+true_add + ns_old+noise_add)/pc_new      # cumulative mean
        keep = (m+1)//2
        top = np.argsort(-means, 1)[:, :keep]
        cur = np.take_along_axis(cur, top, 1)
    return cur[:, 0]

def warmup_aware(theta, dscale, budget=B):
    return sh(theta, 0.0, budget=max(budget - K*W, K*2))         # warm everyone (discard), clean SH on the rest

def run(dscale):
    theta = RNG.normal(0, 1, (REPS, K)); r = np.arange(REPS); oracle = theta.max(1)
    out = {}
    for name, fn in [("uniform", uniform), ("naive_SH", sh), ("warmup_aware_SH", warmup_aware)]:
        sel = fn(theta, dscale)
        out[name] = {"regret": round(float((oracle-theta[r,sel]).mean()),4),
                     "p_correct": round(float((sel==theta.argmax(1)).mean()),3)}
    return out

res = {"model": {"REPS":REPS,"K":K,"SIG":SIG,"B":B,"W":W,"per_arm_uniform":B//K}, "sweep": {}}
print(f"K={K}, sig={SIG}, B={B} ({B//K}/arm), warmup W={W} pulls, REPS={REPS}.  dscale: 0=no bias, 1=early signal zeroed, 1.5=inverted.")
print(f"{'dscale':>7} | {'UNIFORM reg':>11} {'pc':>5} | {'NAIVE-SH reg':>12} {'pc':>5} | {'WARMUP-SH reg':>13} {'pc':>5} | {'SH beats unif?':>14}")
for d in [0.0, 0.5, 1.0, 1.5]:
    o = run(d); res["sweep"][d] = o
    win = "YES" if o["naive_SH"]["regret"] < o["uniform"]["regret"] else "NO (SH worse)"
    print(f"{d:>7} | {o['uniform']['regret']:>11.4f} {o['uniform']['p_correct']:>5} | {o['naive_SH']['regret']:>12.4f} {o['naive_SH']['p_correct']:>5} | {o['warmup_aware_SH']['regret']:>13.4f} {o['warmup_aware_SH']['p_correct']:>5} | {win:>14}")

json.dump(res, open("runs/cycle101_biased_allocation/score.json","w"), indent=1)
print("\n--- READ THE RAW ROWS (don't trust this summary) ---")
s = res["sweep"]
print(f"P1 dscale=0 (c97): naive-SH reg {s[0.0]['naive_SH']['regret']} vs uniform {s[0.0]['uniform']['regret']} (SH should win)")
print(f"P3 dscale=1.5 (inverted): naive-SH reg {s[1.5]['naive_SH']['regret']} vs uniform {s[1.5]['uniform']['regret']} (SH should LOSE)")
print(f"P4 fix dscale=1.5: warmup-aware-SH reg {s[1.5]['warmup_aware_SH']['regret']} (should beat naive-SH {s[1.5]['naive_SH']['regret']})")
print("DONE")
