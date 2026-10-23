"""Cycle 155 (NEW mechanism — verification LATENCY; tail/checkpoint pillar). c82 ("checkpoint THROUGHOUT, evenly")
and c92 (long-horizon leaderboard) both assumed INSTANT checks. Real verification takes TIME: a check started at step
t completes at t+D (latency D) — meanwhile the agent keeps generating, building MORE work on an undetected error.
QUESTION: how does latency D change the optimal check-interval m*, and the cost?

MODEL (c82-faithful propagation): the agent generates T steps; state is CLEAN or CORRUPT-since-e. While CLEAN, each
step introduces an error w.p. p -> state becomes corrupt-since-this-step (everything downstream is corrupt, c82). A
check fires every m steps; a check fired at t COMPLETES at t+D and, on completion, catches any uncaught error at
position <= t and rolls back to it (re-doing the corrupted span). WASTED WORK = steps spent in the corrupt state (all
redone). COST = wasted_steps + n_checks * c_check.

ANALYTIC PRE-REG: cost(m,D) ~ p*T*(detection_delay) + (T/m)*c, where detection_delay = (time to next check) + D ~
m/2 + D. The D term is ADDITIVE and m-INDEPENDENT -> dCost/dm has NO D term -> m* = sqrt(2c/p) INDEPENDENT of D;
and the latency adds an IRREDUCIBLE tax ~ p*T*D that checking-more-often CANNOT reduce.
  P1: optimal check-interval m*(D) is ~CONSTANT in D (latency does NOT change how often to check).
  P2: total cost at m* rises ~LINEARLY in D (the latency tax is irreducible).
  P3: the levers for the latency tax are LOWER p (better generation, c118) or LOWER D (faster verifier), NOT smaller m
      (checking faster cuts the m/2 detection term but NOT the D floor -> diminishing/zero return vs latency).
  Control: D=0 recovers the c82 instant-check regime (m* = sqrt(2c/p)).
"""
import numpy as np, json, os
os.makedirs("runs/cycle155_verification_latency", exist_ok=True)
rng = np.random.default_rng(0)

T = 4000          # steps per run
P = 0.02          # per-step error rate (while clean)
C_CHECK = 20.0    # cost of one check (in step-equivalents)
M_VALUES = [5, 10, 20, 30, 50, 80, 120, 200, 300]
D_VALUES = [0, 10, 25, 50, 100]
REPS = 400

def wasted_steps(m, D, reps):
    """MC: count steps spent in the corrupt state (= redone work) over T, plus n_checks."""
    totals = []
    n_checks = (T // m)
    check_fire = np.arange(m, T + 1, m)              # checks fire at m, 2m, ...
    for _ in range(reps):
        # earliest error position in the run-up: simulate clean->corrupt transitions + catches
        corrupt_since = None          # None=clean, else the step index the corruption began
        corrupt_steps = 0
        errs = rng.random(T) < P      # candidate error at each step (only matters while clean)
        # precompute, for each check, its completion time t+D
        ci = 0
        for s in range(T):
            # does a check COMPLETE at s? (a check fired at f completes at f+D)
            # catch the earliest uncaught error <= f (the check's coverage = up to its fire time f)
            while ci < len(check_fire) and check_fire[ci] + D == s:
                f = check_fire[ci]
                if corrupt_since is not None and corrupt_since <= f:
                    corrupt_since = None   # caught + rolled back
                ci += 1
            if corrupt_since is None:
                if errs[s]:
                    corrupt_since = s
            else:
                corrupt_steps += 1         # this step's work is built on an uncaught error -> wasted
        totals.append(corrupt_steps + n_checks * C_CHECK)
    a = np.array(totals); return float(a.mean()), float(1.96 * a.std(ddof=1) / np.sqrt(reps)), n_checks

out = {"T": T, "P": P, "C_CHECK": C_CHECK, "M_values": M_VALUES, "D_values": D_VALUES, "REPS": REPS,
       "analytic_mstar_sqrt_2c_over_p": float(np.sqrt(2 * C_CHECK / P)), "by_D": {}}
print(f"=== verification LATENCY (T={T}, p={P}, c_check={C_CHECK}, REPS={REPS}) ===")
print(f"analytic m* = sqrt(2c/p) = {np.sqrt(2*C_CHECK/P):.1f} (instant-check optimum, D-independent prediction)")
print(f"{'D(latency)':>11} {'m*(argmin)':>11} {'cost@m*':>14} {'cost@m=5':>12} {'cost@m=200':>12}")
for D in D_VALUES:
    costs = {}
    for m in M_VALUES:
        mean, ci, nc = wasted_steps(m, D, REPS)
        costs[m] = (mean, ci)
    mstar = min(M_VALUES, key=lambda m: costs[m][0])
    out["by_D"][D] = {"mstar": mstar, "cost_at_mstar": costs[mstar][0],
                      "costs": {m: {"mean": costs[m][0], "ci": costs[m][1]} for m in M_VALUES}}
    print(f"{D:>11} {mstar:>11} {costs[mstar][0]:>9.1f}+-{costs[mstar][1]:<4.1f} {costs[5][0]:>12.1f} {costs[200][0]:>12.1f}")
# latency tax: cost@m* vs D should be ~linear; slope ~ p*T
import numpy as _np
Ds = _np.array(D_VALUES, dtype=float)
costs_at_mstar = _np.array([out["by_D"][D]["cost_at_mstar"] for D in D_VALUES])
slope = float(_np.polyfit(Ds, costs_at_mstar, 1)[0])
out["latency_tax_slope_dcost_dD"] = slope
out["predicted_slope_pT"] = float(P * T)
mstars = [out["by_D"][D]["mstar"] for D in D_VALUES]
out["mstar_range"] = [min(mstars), max(mstars)]
json.dump(out, open("runs/cycle155_verification_latency/score.json", "w"), indent=1)
print("---")
print(f"m*(D) across D = {mstars}  (PRE-REG P1: ~constant in D)")
print(f"latency tax: d(cost@m*)/dD = {slope:.2f}  vs predicted p*T = {P*T:.1f}  (PRE-REG P2: linear, slope~p*T)")
print("DONE")
