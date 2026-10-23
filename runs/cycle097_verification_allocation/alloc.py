"""Cycle 97 (NEW mechanism, verification pillar): HOW to ALLOCATE a fixed verification budget across K candidates.
The thesis established THAT you re-eval (winner's curse, c7) and that independence matters (c45); c8 gave a crude
2-stage rule (re-eval top-3) and c35 the search/verify SPLIT. The unmined question: given a fixed budget B of re-evals
and K candidates with noisy eval, how to ALLOCATE re-evals to identify the true-best with least regret — UNIFORM
(B/K each) vs ADAPTIVE (successive-halving best-arm-identification: spend more on the contenders)? Connects the
winner's-curse-debiasing to the best-arm-identification literature (Karnin et al. 2013 successive halving).

Model: K candidates, true quality theta_i ~ N(0, sig_theta^2); each re-eval = theta_i + N(0, sig_obs^2). Budget B
total re-evals. Compare P(select true-best) and REGRET (theta_oracle - theta_selected). numpy-only, REPS reps."""
import numpy as np, json, os
RNG = np.random.default_rng(0)
REPS, SIG_THETA = 20000, 1.0
os.makedirs("runs/cycle097_verification_allocation", exist_ok=True)

def uniform(theta, B, sig):
    reps, K = theta.shape; n = max(B//K, 1)
    means = theta + RNG.normal(0, sig, (reps, K, n)).mean(axis=2)
    return means.argmax(axis=1), n*K

def succ_halving(theta, B, sig):
    reps, K = theta.shape; R = max(int(np.ceil(np.log2(K))), 1)
    cur = np.tile(np.arange(K), (reps, 1)); used = 0
    while cur.shape[1] > 1:
        m = cur.shape[1]; n_r = max(B//(m*R), 1)
        th = np.take_along_axis(theta, cur, axis=1)
        means = th + RNG.normal(0, sig, (reps, m, n_r)).mean(axis=2)
        used += n_r*m
        keep = (m+1)//2
        top = np.argsort(-means, axis=1)[:, :keep]
        cur = np.take_along_axis(cur, top, axis=1)
    return cur[:, 0], used

def evaluate(K, B, sig):
    theta = RNG.normal(0, SIG_THETA, (REPS, K))
    oracle = theta.max(axis=1); r = np.arange(REPS)
    su, bu = uniform(theta, B, sig); ss, bs = succ_halving(theta, B, sig)
    return {
        "uniform": {"p_correct": round(float((su==theta.argmax(1)).mean()),3),
                    "regret": round(float((oracle-theta[r,su]).mean()),4), "budget_used": int(bu)},
        "sh":      {"p_correct": round(float((ss==theta.argmax(1)).mean()),3),
                    "regret": round(float((oracle-theta[r,ss]).mean()),4), "budget_used": int(bs)},
    }

out = {"model": {"REPS":REPS,"SIG_THETA":SIG_THETA}, "A_vs_K": {}, "B_vs_budget": {}}

# ===== PART A: vs K at fixed budget-per-arm (B=6K, sig_obs=2 = noisy regime) =====
sig0 = 2.0; perarm = 6
print(f"=== A. Uniform vs Successive-Halving allocation vs #candidates K  (B=6K re-evals, sig_obs={sig0}) ===")
print(f"{'K':>4} {'B':>5} | {'UNIF p_corr':>11} {'regret':>7} | {'SH p_corr':>10} {'regret':>7} | {'regret cut':>10}")
for K in [4, 8, 16, 32, 64]:
    B = perarm*K; e = evaluate(K, B, sig0); out["A_vs_K"][K] = e
    rc = (e["uniform"]["regret"]-e["sh"]["regret"])
    print(f"{K:>4} {B:>5} | {e['uniform']['p_correct']:>11} {e['uniform']['regret']:>7} | {e['sh']['p_correct']:>10} {e['sh']['regret']:>7} | {rc:>+10.4f}  (used U={e['uniform']['budget_used']} SH={e['sh']['budget_used']})")

# ===== PART B: vs budget tightness at fixed K=32, sig_obs=2 =====
K0 = 32
print(f"\n=== B. vs budget (K={K0} candidates, sig_obs={sig0}) — when does adaptive stop mattering? ===")
print(f"{'B/K':>5} {'B':>6} | {'UNIF p_corr':>11} {'regret':>7} | {'SH p_corr':>10} {'regret':>7} | {'regret cut':>10}")
for perarm in [2, 4, 8, 16, 32]:
    B = perarm*K0; e = evaluate(K0, B, sig0); out["B_vs_budget"][perarm] = e
    rc = (e["uniform"]["regret"]-e["sh"]["regret"])
    print(f"{perarm:>5} {B:>6} | {e['uniform']['p_correct']:>11} {e['uniform']['regret']:>7} | {e['sh']['p_correct']:>10} {e['sh']['regret']:>7} | {rc:>+10.4f}")

json.dump(out, open("runs/cycle097_verification_allocation/score.json","w"), indent=1)
print("\n--- READ THE RAW ROWS ABOVE (don't trust this summary) ---")
A = out["A_vs_K"]
print(f"P1/P2 regret-cut (UNIF-SH) vs K: " + " ".join(f"K{K}:{A[K]['uniform']['regret']-A[K]['sh']['regret']:+.3f}" for K in [4,8,16,32,64]) + "  (adaptive better? grows with K?)")
Bd = out["B_vs_budget"]
print(f"P4 regret-cut vs budget (K=32): " + " ".join(f"{pa}/arm:{Bd[pa]['uniform']['regret']-Bd[pa]['sh']['regret']:+.3f}" for pa in [2,4,8,16,32]) + "  (shrinks as budget grows?)")
print("DONE")
