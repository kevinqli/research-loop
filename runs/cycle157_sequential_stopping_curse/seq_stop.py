"""Cycle 157 (NEW mechanism — the SEQUENTIAL-STOPPING winner's curse; verification pillar). c7 is the BATCH curse:
draw K candidates, ship the max-observed (inflated by the max-order-statistic E[max_K]). But a REAL agent searches
SEQUENTIALLY: generate candidates one at a time and STOP at the first that looks "good enough" (observed > threshold
τ), then ship it. The shipped result is winner's-cursed by TRUNCATION — you stopped BECAUSE it looked good (an up-
fluctuation), so observed > true. Distinct selection geometry from c7 (threshold-truncation, not max-of-K).

MODEL: candidates iid — true quality q ~ N(0,1), observed = q + N(0, σ_obs²). Generate sequentially; STOP at the
first observed > τ; ship it. Measure observed_stopped, true_stopped, bias = observed−true, search cost N, and a
HELD-OUT re-eval (a fresh independent observation of the stopped candidate's true q).

ANALYTIC: for the stopped candidate, observed > τ. E[true|stopped] = E[observed|stopped]/(1+σ²) and
bias = E[observed|stopped]·σ²/(1+σ²) — so observed is inflated by a constant FACTOR (1+σ²) over true (the bias/true
ratio = σ², constant in τ), while the ABSOLUTE bias grows with τ (E[observed|stopped] grows with τ).

PRE-REG:
  P1: the stopped candidate is INFLATED (observed > true); ABSOLUTE bias GROWS with τ.
  P2: higher τ → higher TRUE quality (searched longer, genuinely better) BUT more inflation — a tradeoff
      (raise the bar → better result, WORSE calibration); the inflation FACTOR observed/true ≈ (1+σ²) constant in τ.
  P3: a HELD-OUT re-eval of the stopped candidate ≈ true (unbiased) — SAME fix as c7 (c7/c45), new setting.
  P4: search cost (N until stop) GROWS with τ (explodes as τ approaches the achievable ceiling).
"""
import numpy as np, json, os
os.makedirs("runs/cycle157_sequential_stopping_curse", exist_ok=True)
rng = np.random.default_rng(0)

SIGMA_OBS = 0.6
TAU_VALUES = [0.5, 1.0, 1.5, 2.0, 2.5]
N_MAX = 20000
REPS = 20000

def run_tau(tau, reps):
    obs_s, true_s, held_s, costs, timeouts = [], [], [], [], 0
    for _ in range(reps):
        n = 0; stopped = False
        while n < N_MAX:
            n += 1
            q = rng.standard_normal()
            o = q + SIGMA_OBS * rng.standard_normal()
            if o > tau:
                obs_s.append(o); true_s.append(q)
                held_s.append(q + SIGMA_OBS * rng.standard_normal())   # held-out re-eval (fresh independent obs)
                costs.append(n); stopped = True; break
        if not stopped:
            timeouts += 1
    o = np.array(obs_s); t = np.array(true_s); h = np.array(held_s); c = np.array(costs)
    def ci(a): return float(1.96 * a.std(ddof=1) / np.sqrt(len(a)))
    return {"observed": [float(o.mean()), ci(o)], "true": [float(t.mean()), ci(t)],
            "bias_obs_minus_true": float(o.mean() - t.mean()), "held_out": [float(h.mean()), ci(h)],
            "held_minus_true": float(h.mean() - t.mean()), "inflation_factor_obs_over_true": float(o.mean() / t.mean()),
            "mean_search_cost": float(c.mean()), "timeout_frac": timeouts / reps}

out = {"SIGMA_OBS": SIGMA_OBS, "TAU_values": TAU_VALUES, "REPS": REPS, "predicted_inflation_factor_1plus_sig2": 1 + SIGMA_OBS**2, "by_tau": {}}
print(f"=== SEQUENTIAL-STOPPING winner's curse (sigma_obs={SIGMA_OBS}, REPS={REPS}) ===")
print(f"predicted inflation factor observed/true = 1+sigma^2 = {1+SIGMA_OBS**2:.3f} (constant in tau)")
print(f"{'tau':>5} {'observed':>16} {'true':>16} {'bias':>9} {'held-out':>16} {'infl.factor':>11} {'search_cost':>12}")
for tau in TAU_VALUES:
    r = run_tau(tau, REPS)
    out["by_tau"][tau] = r
    print(f"{tau:>5} {r['observed'][0]:>8.3f}+-{r['observed'][1]:<6.3f} {r['true'][0]:>8.3f}+-{r['true'][1]:<6.3f}"
          f" {r['bias_obs_minus_true']:>9.3f} {r['held_out'][0]:>8.3f}+-{r['held_out'][1]:<6.3f}"
          f" {r['inflation_factor_obs_over_true']:>11.3f} {r['mean_search_cost']:>12.1f}")
json.dump(out, open("runs/cycle157_sequential_stopping_curse/score.json", "w"), indent=1)
print("PRE-REG: P1 bias grows w/ tau; P2 true grows w/ tau, infl.factor~1+sig2 const; P3 held-out~true; P4 search cost grows w/ tau.")
print("DONE")
