"""Cycle 175 (NEW — COMBINE vs SELECT: a generation-side output strategy; diversified off the checkpoint/real-agent
streak). A parallel-agent lab with K candidate outputs can SELECT the best (best-of-K by a noisy verifier, c7) or
COMBINE them (ensemble/average). Select keeps the TAIL but is WINNER'S-CURSED (c7: the selected is inflated, high
variance); ensemble DODGES the curse (no selection) + is honest + low-variance, but REGRESSES TO THE MEAN (loses the tail).

MODEL: K candidates, true q ~ N(mu=1, sigma_q); observed = q + N(0, sigma_v) (verifier noise; corr can be flipped to
ANTI for an adversarial/bad verifier). SELECT: output = true q of argmax-observed; reported = max-observed (the curse).
ENSEMBLE: output = mean of the K true q; reported = mean-observed (honest). Metrics: E[true of output], std(true),
and reported-minus-true (the curse/over-statement).

PRE-REG: P1 SELECT >= ENSEMBLE in expected TRUE quality with a positively-correlated verifier (select gets >= mu;
ensemble = mu); gap grows with sigma_q + verifier quality. P2 ENSEMBLE has LOWER VARIANCE (mu +- sigma_q/sqrt(K)) and an
HONEST report (reported ~ true); SELECT's report is INFLATED (the curse). P3 ENSEMBLE BEATS select when the verifier is
ANTI-correlated/adversarial (select picks a lucky-bad candidate < mu) or near-random; + dodges the curse entirely.
"""
import numpy as np, json, os
os.makedirs("runs/cycle175_ensemble_vs_select", exist_ok=True)
rng = np.random.default_rng(0)
MU, SIGMA_Q, K, REPS = 1.0, 1.0, 20, 40000

def run(sigma_v, corr_sign=+1):
    # K candidates per rep; true quality q ~ N(MU, SIGMA_Q). Verifier score is on the quality scale (mean MU):
    # positive verifier tracks q (obs = q + noise); anti-correlated verifier REFLECTS q around MU
    # (obs = 2*MU - q + noise) -> identical marginal (mean MU, same variance) but anti-correlated, so the
    # curse stays measurable on the same scale in BOTH regimes. Vectorized over REPS for speed.
    q = rng.normal(MU, SIGMA_Q, (REPS, K))
    obs = (q if corr_sign > 0 else 2 * MU - q) + rng.normal(0, sigma_v, (REPS, K))
    i = obs.argmax(axis=1); rows = np.arange(REPS)
    st, sr = q[rows, i], obs[rows, i]            # SELECT: true / reported = best-by-observed
    et, er = q.mean(axis=1), obs.mean(axis=1)    # ENSEMBLE: true / reported = the average
    return {"select_true_mean": float(st.mean()), "select_true_std": float(st.std()),
            "select_reported_mean": float(sr.mean()), "select_curse": float(sr.mean() - st.mean()),
            "ensemble_true_mean": float(et.mean()), "ensemble_true_std": float(et.std()),
            "ensemble_reported_mean": float(er.mean()), "ensemble_curse": float(er.mean() - et.mean())}

out = {"MU": MU, "SIGMA_Q": SIGMA_Q, "K": K, "REPS": REPS, "positive_verifier": {}, "anti_verifier": {}}
print(f"=== COMBINE vs SELECT (K={K}, mu={MU}, sigma_q={SIGMA_Q}, REPS={REPS}) ===")
print("POSITIVE (normal) verifier — vary noise sigma_v:")
print(f"{'sig_v':>6} {'SELECT true':>12} {'(std)':>7} {'curse':>7} | {'ENS true':>9} {'(std)':>7} {'curse':>7} {'select>ens?':>11}")
for sv in [0.3, 0.6, 1.0, 2.0, 4.0]:
    r = run(sv, +1); out["positive_verifier"][sv] = r
    print(f"{sv:>6} {r['select_true_mean']:>12.3f} {r['select_true_std']:>7.3f} {r['select_curse']:>7.3f} | "
          f"{r['ensemble_true_mean']:>9.3f} {r['ensemble_true_std']:>7.3f} {r['ensemble_curse']:>7.3f} "
          f"{'YES' if r['select_true_mean']>r['ensemble_true_mean'] else 'NO':>11}")
print("ANTI-correlated (adversarial/broken) verifier:")
r = run(1.0, -1); out["anti_verifier"]["sigma_v_1.0"] = r
print(f"  SELECT true {r['select_true_mean']:.3f} (curse {r['select_curse']:.3f}) vs ENSEMBLE true {r['ensemble_true_mean']:.3f} "
      f"-> ensemble {'BEATS' if r['ensemble_true_mean']>r['select_true_mean'] else 'loses to'} select")
json.dump(out, open("runs/cycle175_ensemble_vs_select/score.json", "w"), indent=1)
print("PRE-REG: P1 select>=ens (positive verifier); P2 ens lower-std + honest (curse~0) vs select inflated; P3 ens BEATS select under anti-corr verifier.")
print("DONE")
