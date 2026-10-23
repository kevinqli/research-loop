"""Cycle 180 (#31 — the SELECT-then-ENSEMBLE hybrid / bias-variance frontier; NEW desk sim, diversified MODE off the
c178/c179 real-agent cycles; unifies c175's two endpoints into a continuum). c175: pure-SELECT (best-of-K, m=1) keeps
the tail but is winner's-cursed + high-variance; pure-ENSEMBLE (average all K, m=K) is honest + low-variance but
regresses to the mean. The hybrid: SELECT the top-m by the verifier, then AVERAGE those m. m=1 = pure select, m=K =
pure ensemble. = SHRINKAGE forecast combination (c176 grounding): interpolate between best-pick and equal-weight.

MODEL (identical to c175's positive verifier): K candidates, true q ~ N(MU, SIGMA_Q); verifier obs = q + N(0, sigma_v).
Rank by obs, take the top m: OUTPUT = mean of their TRUE q; REPORTED = mean of their obs. curse = reported - true.
Built-in CORRECTNESS CHECKS: m=1 must reproduce c175 SELECT (sigma_v=0.3 true~2.790, 1.0~2.326); m=K must reproduce
c175 ENSEMBLE (true~MU=1.0, curse~0, std~SIGMA_Q/sqrt(K)=0.224).

PRE-REG: P1 output-true DECREASES with m (m=1 max-tail -> m=K=MU), but a SMALL m>1 retains MOST of the tail (top-3 close
to top-1). P2 output-std DECREASES with m (averaging cuts variance toward sigma_q/sqrt(m)). P3 the CURSE DECREASES with
m (the top-m average obs is less inflated than the single max; m=K -> ~0). P4 under a honest-quality objective
(true - |curse|), m* is INTERIOR (>1) and GROWS with sigma_v (worse verifier -> average more = shrink toward equal-weight,
the c176 shrinkage prior).
"""
import numpy as np, json, os
os.makedirs("runs/cycle180_select_then_ensemble", exist_ok=True)
rng = np.random.default_rng(0)
MU, SIGMA_Q, K, REPS = 1.0, 1.0, 20, 40000
M_VALUES = [1, 2, 3, 5, 10, 20]
SIGMA_VS = [0.3, 1.0, 2.0]

def run(sigma_v):
    q = rng.normal(MU, SIGMA_Q, (REPS, K))
    obs = q + rng.normal(0, sigma_v, (REPS, K))
    order = np.argsort(-obs, axis=1)                 # descending by obs; rank once, reuse for every m
    q_ranked = np.take_along_axis(q, order, axis=1)
    obs_ranked = np.take_along_axis(obs, order, axis=1)
    out = {}
    for m in M_VALUES:
        out_true = q_ranked[:, :m].mean(axis=1)      # true quality of the top-m average
        out_rep = obs_ranked[:, :m].mean(axis=1)     # reported (verifier) score of the top-m average
        out[m] = {"true_mean": float(out_true.mean()), "true_std": float(out_true.std()),
                  "reported_mean": float(out_rep.mean()), "curse": float(out_rep.mean() - out_true.mean())}
    return out

results = {}
print(f"=== SELECT-then-ENSEMBLE hybrid (K={K}, mu={MU}, sigma_q={SIGMA_Q}, REPS={REPS}) ===")
for sv in SIGMA_VS:
    r = run(sv); results[str(sv)] = r
    m1, mK = r[1], r[K]
    tail1 = m1["true_mean"] - MU                       # the m=1 excess-over-mean (the full tail)
    print(f"\n--- sigma_v={sv} (m=1 SELECT true={m1['true_mean']:.3f} curse={m1['curse']:.3f} std={m1['true_std']:.3f} | "
          f"m={K} ENSEMBLE true={mK['true_mean']:.3f} curse={mK['curse']:.3f} std={mK['true_std']:.3f}) ---")
    print(f"{'m':>3} {'true':>7} {'std':>6} {'curse':>7} {'tail-kept%':>10} {'curse-left%':>11} {'honest=true-|curse|':>19}")
    best_m, best_score = None, -1e9
    for m in M_VALUES:
        d = r[m]
        tail_kept = 100 * (d["true_mean"] - MU) / tail1 if tail1 > 1e-9 else 0.0
        curse_left = 100 * d["curse"] / m1["curse"] if m1["curse"] > 1e-9 else 0.0
        honest = d["true_mean"] - abs(d["curse"])      # illustrative honest-quality objective (penalize over-statement)
        d["tail_kept_pct"] = round(tail_kept, 1); d["curse_left_pct"] = round(curse_left, 1); d["honest_quality"] = round(honest, 3)
        if honest > best_score: best_score, best_m = honest, m
        print(f"{m:>3} {d['true_mean']:>7.3f} {d['true_std']:>6.3f} {d['curse']:>7.3f} {tail_kept:>10.1f} {curse_left:>11.1f} {honest:>19.3f}")
    results[str(sv)]["m_star_honest"] = best_m
    print(f"  -> m* (max true-|curse|) = {best_m}")

json.dump({"MU": MU, "SIGMA_Q": SIGMA_Q, "K": K, "REPS": REPS, "M_VALUES": M_VALUES, "SIGMA_VS": SIGMA_VS, "results": results},
          open("runs/cycle180_select_then_ensemble/score.json", "w"), indent=1)
print("\nm*(sigma_v):", {sv: results[str(sv)]["m_star_honest"] for sv in SIGMA_VS})
print("PRE-REG: P1 true decreases w/ m (small m keeps most tail); P2 std decreases; P3 curse decreases; "
      "P4 m*(honest) interior + grows with sigma_v (shrink toward equal-weight, c176).")
print("DONE")
