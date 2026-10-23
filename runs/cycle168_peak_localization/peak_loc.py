"""Cycle 168 (#17d — HARDEN c148: how much FRESH data to LOCATE the design-overfitting peak R*? benchmark-design
pillar; diversified off the over-verification thread + GROUND mode). c148: adaptive benchmark design has an OVERFITTING
PEAK in true discrimination at R* rounds; a FRESH held-out re-eval RECOVERS the true curve and locates R*. But the fresh
held-out is itself FINITE/NOISY (A_fresh agents -> noise sigma_f = sigma0/sqrt(A_fresh)). So the builder picks
R_hat = argmax of the NOISY fresh curve, which mis-locates R*. How much does that cost?

MODEL: true curve T(R) = peak - kappa*(R-R*)^2 (a parabola; kappa = curvature = peak SHARPNESS; flat peak = small
kappa). The fresh re-eval observes T(R) + N(0, sigma_f^2) at each R; R_hat = argmax over R. Measure localization error
|R_hat - R*| and REGRET = T(R*) - T(R_hat) = kappa*(R_hat-R*)^2 (true-quality lost by stopping at the mis-located peak).

PRE-REG: P1 localization error |R_hat-R*| GROWS with sigma_f (less fresh data -> worse peak ID). P2 a FLATTER peak
(smaller kappa) -> LARGER localization error (the argmax is unstable when the curve is flat near R*). P3 (the actionable
nuance): but the REGRET is SMALL near a flat peak (mis-locating where the curve is flat costs little) -> you can tolerate
a noisy fresh re-eval and still get NEAR-optimal true quality even if you can't pin R*; the fresh-data budget you need
depends on the peak SHAPE (sharp peak -> mis-location is costly, need more fresh data; flat peak -> cheap, need less).
"""
import numpy as np, json, os
os.makedirs("runs/cycle168_peak_localization", exist_ok=True)
rng = np.random.default_rng(0)

R_GRID = np.arange(0, 201)      # candidate #rounds R
RSTAR = 100                     # true peak location
PEAK = 10.0
SIGMA_F = [0.25, 0.5, 1.0, 2.0, 4.0]    # fresh-re-eval noise (sigma0/sqrt(A_fresh); larger = less fresh data)
KAPPA = {"sharp": 0.01, "medium": 0.003, "flat": 0.001}   # peak curvature
REPS = 4000

def run(kappa, sigma_f, reps):
    T = PEAK - kappa * (R_GRID - RSTAR) ** 2          # true curve (parabola)
    loc_err, regret = [], []
    for _ in range(reps):
        obs = T + rng.normal(0, sigma_f, len(R_GRID))  # noisy fresh re-eval
        rhat = R_GRID[int(np.argmax(obs))]
        loc_err.append(abs(rhat - RSTAR))
        regret.append(T[RSTAR] - T[int(np.argmax(obs))])   # true-quality lost
    return float(np.mean(loc_err)), float(np.mean(regret))

out = {"R_GRID": [int(R_GRID[0]), int(R_GRID[-1])], "RSTAR": RSTAR, "PEAK": PEAK, "SIGMA_F": SIGMA_F,
       "KAPPA": KAPPA, "REPS": REPS, "by_kappa": {}}
print(f"=== #17d: locating the design-overfitting peak R*={RSTAR} from a NOISY fresh re-eval (REPS={REPS}) ===")
print(f"peak height={PEAK}; full-range regret if you picked R=0 or 200: kappa*100^2 = sharp {KAPPA['sharp']*1e4:.0f} / medium {KAPPA['medium']*1e4:.0f} / flat {KAPPA['flat']*1e4:.0f}")
for kname, kappa in KAPPA.items():
    print(f"\n--- {kname} peak (kappa={kappa}) ---")
    print(f"{'sigma_f':>8} {'|Rhat-R*|':>11} {'regret(true-qual)':>18}")
    out["by_kappa"][kname] = {}
    for sf in SIGMA_F:
        le, rg = run(kappa, sf, REPS)
        out["by_kappa"][kname][sf] = {"loc_err": round(le,2), "regret": round(rg,4)}
        print(f"{sf:>8} {le:>11.2f} {rg:>18.4f}")
json.dump(out, open("runs/cycle168_peak_localization/score.json", "w"), indent=1)
print("\nPRE-REG: P1 loc-err grows w/ sigma_f; P2 flatter peak -> larger loc-err; P3 regret SMALL near a flat peak (mis-location cheap).")
print("DONE")
