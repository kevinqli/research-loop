"""Cycle 22 (pillar 2 scaffolds × the winner's curse): does a BRANCHING (best-of-K tree-search) scaffold beat a
LINEAR (single-trajectory) one — and how does the winner's curse (cycle 7) distort the comparison & the cost trade?
Model: each candidate solution has true perf theta ~ N(mu, s_true); single-holdout validation v = theta + N(0,s_val).
  LINEAR  (K=1): 1 candidate. reported = its v; true = its theta.
  BRANCH  (K):   K candidates, select argmax(v). reported = max v (winner's-curse inflated); true = theta[selected];
                 held-out RE-EVAL = theta[selected] + fresh N(0,s_val) (unbiased).
Branching costs ~K x the tokens. numpy-only, 4000 reps, across-rep CIs. MPS-exploratory (simulation of the mechanism).
Pre-reg: (1) TRUE gain of branch over linear GROWS with K but with diminishing returns; (2) REPORTED gain OVERSTATES
the true gain (winner's curse) -> comparing scaffolds by self-reported score over-credits branching; (3) held-out
re-eval recovers the honest gain; (4) true-gain-PER-TOKEN peaks at small K (K x cost eats the diminishing gain)."""
import numpy as np, json
from math import sqrt
rng = np.random.default_rng(0)
REPS = 4000; MU, S_TRUE, S_VAL = 0.80, 0.05, 0.05
Ks = [1, 2, 4, 8, 16, 32]
ci = lambda x: 1.96*np.asarray(x).std(ddof=1)/sqrt(len(x))

# linear baseline (K=1): true = theta, reported = theta + val noise
theta_lin = rng.normal(MU, S_TRUE, REPS)
reported_lin = theta_lin + rng.normal(0, S_VAL, REPS)
true_lin_mean = theta_lin.mean()
reported_lin_mean = reported_lin.mean()

print(f"REPS={REPS}, true~N({MU},{S_TRUE}), val noise={S_VAL}")
print(f"LINEAR (K=1): true={true_lin_mean:.4f}, reported={reported_lin_mean:.4f}\n")
print(f"{'K':>3} | {'true_perf':>16} | {'true_gain':>10} | {'reported_gain':>13} | {'inflation':>10} | {'reeval_gain':>11} | {'true_gain/token':>15}")
rows=[]
for K in Ks:
    th = rng.normal(MU, S_TRUE, (REPS, K))
    v = th + rng.normal(0, S_VAL, (REPS, K))
    sel = np.argmax(v, axis=1)
    true_b = th[np.arange(REPS), sel]
    reported_b = v[np.arange(REPS), sel]
    reeval_b = true_b + rng.normal(0, S_VAL, REPS)        # held-out re-eval (unbiased)
    true_gain = true_b.mean() - true_lin_mean             # genuine improvement from branching
    reported_gain = reported_b.mean() - reported_lin_mean # apparent improvement (what you'd see by self-report)
    reeval_gain = reeval_b.mean() - reported_lin_mean     # honest improvement after held-out re-eval
    inflation = reported_gain - true_gain                 # winner's-curse over-credit
    per_token = true_gain / K                             # true gain per unit compute (~K x tokens)
    print(f"{K:>3} | {true_b.mean():.4f} ± {ci(true_b):.4f} | {true_gain:+.4f}   | {reported_gain:+.4f}      | {inflation:+.4f}   | {reeval_gain:+.4f}    | {per_token:+.6f}")
    rows.append({"K":K,"true_perf":float(true_b.mean()),"true_gain":float(true_gain),
                 "reported_gain":float(reported_gain),"inflation":float(inflation),
                 "reeval_gain":float(reeval_gain),"true_gain_per_token":float(per_token)})
json.dump({"linear":{"true":float(true_lin_mean),"reported":float(reported_lin_mean)},"branch":rows},
          open("runs/cycle022_scaffold/score.json","w"), indent=1)

print("\n--- read the rows above, not this line ---")
r=rows  # K index: 0->K1,1->K2,...
print(f"(1) true gain grows w/ K but diminishes: K2 {r[1]['true_gain']:+.4f} -> K8 {r[3]['true_gain']:+.4f} -> K32 {r[5]['true_gain']:+.4f}")
print(f"(2) reported OVERSTATES: at K32 reported_gain {r[5]['reported_gain']:+.4f} vs true_gain {r[5]['true_gain']:+.4f} (inflation {r[5]['inflation']:+.4f})")
print(f"(3) held-out reeval_gain {r[5]['reeval_gain']:+.4f} ~ true_gain {r[5]['true_gain']:+.4f} (recovers honest gain)")
best=max(r, key=lambda x:x['true_gain_per_token']); print(f"(4) true-gain/token peaks at K={best['K']} ({best['true_gain_per_token']:+.6f})")
print("wrote runs/cycle022_scaffold/score.json ; DONE")
