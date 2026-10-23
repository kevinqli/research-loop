"""Cycle 163 (#23 — HARDEN c155's verification-latency rule under (a) PARTIAL corruption and (b) VARIABLE latency).
c155 (assumed TOTAL downstream corruption + FIXED latency D) found: optimal check-interval m*~max(c82-optimal, ~2D);
latency = a modest irreducible tax fixed by a FASTER verifier, not more checks ("don't check faster than your latency").
Two stress tests:
  (a) PARTIAL corruption: an uncaught error doesn't FULLY corrupt downstream — each corrupt step costs gamma<1 (it
      degrades, not destroys). Predict: as gamma drops, m* RISES (errors cost less -> check less) and the latency tax
      SHRINKS ~gamma; the "don't check faster than D" rule HOLDS (even more so).
  (b) VARIABLE latency: D ~ a distribution (same MEAN D-bar), each check independent. With overlapping checks an error
      is caught at the MIN completion-time over covering checks -> a later FAST check can catch it before an earlier
      SLOW one. Predict: high D-variance gives LOWER detection delay + LOWER tax than FIXED D-bar (variance HELPS via
      the min), and m* may be SMALLER (more checks = more chances for a fast one).

Model (c155-faithful): T steps; while CLEAN, error w.p. p -> corrupt-since-e (propagation). A check fires every m;
fired at t it draws D_t and completes at t+D_t, catching any uncaught error <= t (rolls back). WASTED = gamma * (corrupt
steps). COST = wasted + n_checks*c.
"""
import numpy as np, json, os
os.makedirs("runs/cycle163_latency_harden", exist_ok=True)
rng = np.random.default_rng(0)

T, P, C_CHECK, REPS = 4000, 0.02, 20.0, 300
M_VALUES = [10, 20, 30, 50, 80, 120, 200, 300]

def run(m, D_mode, D_mean, gamma, reps):
    """D_mode: 'fixed' | 'uniform'(0..2*mean) | 'exp'(mean). Returns mean total cost over reps."""
    fire = np.arange(m, T + 1, m)
    totals = []
    for _ in range(reps):
        errs = rng.random(T) < P
        # pre-draw each check's latency
        if D_mode == "fixed":   Ds = np.full(len(fire), D_mean)
        elif D_mode == "uniform": Ds = rng.uniform(0, 2 * D_mean, len(fire))
        elif D_mode == "exp":     Ds = rng.exponential(D_mean, len(fire))
        complete = fire + Ds                      # completion time of each check
        # event-driven: process completions in time order; a completion at ct (covering up to its fire time f)
        # catches the earliest uncaught error <= f.
        order = np.argsort(complete)
        corrupt_since = None; corrupt_steps = 0; s = 0
        # walk steps 1..T, interleaving check-completions
        comp_times = complete[order]; comp_fire = fire[order]; ci = 0
        for step in range(T):
            while ci < len(comp_times) and comp_times[ci] <= step:
                if corrupt_since is not None and corrupt_since <= comp_fire[ci]:
                    corrupt_since = None
                ci += 1
            if corrupt_since is None:
                if errs[step]: corrupt_since = step
            else:
                corrupt_steps += 1
        totals.append(gamma * corrupt_steps + len(fire) * C_CHECK)
    return float(np.mean(totals))

def find_mstar(D_mode, D_mean, gamma, reps):
    costs = {m: run(m, D_mode, D_mean, gamma, reps) for m in M_VALUES}
    mstar = min(M_VALUES, key=lambda m: costs[m])
    return mstar, costs[mstar], costs

out = {"T": T, "P": P, "C_CHECK": C_CHECK, "REPS": REPS, "M_VALUES": M_VALUES}

# (a) PARTIAL corruption: fixed D=50, vary gamma
print("=== (a) PARTIAL corruption (fixed D=50), vary gamma ===")
print(f"{'gamma':>7} {'m*':>5} {'cost@m*':>10}")
out["partial"] = {}
for g in [1.0, 0.5, 0.2]:
    ms, cm, _ = find_mstar("fixed", 50, g, REPS)
    out["partial"][g] = {"mstar": ms, "cost": cm}
    print(f"{g:>7} {ms:>5} {cm:>10.1f}")

# (b) VARIABLE latency: gamma=1, D_mean=50, compare fixed vs uniform vs exp
print("=== (b) VARIABLE latency (gamma=1, D_mean=50): fixed vs uniform[0,100] vs exp(50) ===")
print(f"{'D_mode':>10} {'m*':>5} {'cost@m*':>10} {'cost@m=20':>10} {'cost@m=200':>10}")
out["variable"] = {}
for mode in ["fixed", "uniform", "exp"]:
    ms, cm, costs = find_mstar(mode, 50, 1.0, REPS)
    out["variable"][mode] = {"mstar": ms, "cost": cm, "cost_m20": costs[20], "cost_m200": costs[200]}
    print(f"{mode:>10} {ms:>5} {cm:>10.1f} {costs[20]:>10.1f} {costs[200]:>10.1f}")

json.dump(out, open("runs/cycle163_latency_harden/score.json", "w"), indent=1)
print("PRE-REG: (a) gamma down -> m* up + tax ~gamma (rule HOLDS); (b) variable D (same mean) -> LOWER cost than fixed (variance helps via min-completion), m* maybe smaller.")
print("DONE")
