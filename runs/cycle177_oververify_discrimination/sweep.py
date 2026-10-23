"""Cycle 177 (#27a HARDEN — does the over-verification stringency optimum tau* track the verifier's DISCRIMINATION?
Hardens c164/c165/c167 on the DETECTION-THRESHOLD axis (B) of the c170 two-axis map; diversified off the c175/c176
generation/ensemble cycles). c164 fixed the verifier noise SIGMA=0.5 (good-step score~N(1,sigma), bad~N(0,sigma);
separation=1, so discrimination d'=1/sigma) and found an inverted-U tau* in stringency (flag if score<tau). #27a: sweep
the discrimination (vary sigma) -> does tau* move? = SDT's "a better detector can run a stricter criterion".

MODEL (identical to c164, re-derived + vectorized): L-step task, genuine per-step error rate A. tau = flag threshold.
recall r=P(N(0,sigma)<tau)=catch a bad step; false-pos f=P(N(1,sigma)<tau)=flag a good step. Retry budget B=BUDGET_FRAC*L.
A flag (bad-caught OR good-false-alarm) consumes 1 retry IF budget>0. Trajectory CORRECT iff every bad step is caught
AND, processing in order, had budget when reached. Vectorized exact equivalent: correct iff (all bad steps caught) AND
(max over bad steps of the cumulative flag-count up-to-that-step <= B) -- because flags consume budget in order, the k-th
flag gets the k-th unit, so a bad step is fixable iff its flag-rank<=B; overcounting only begins after budget hits 0, by
which point any later bad step already fails (needs budget>0), so the condition is exact.

PRE-REG: P1 tau* INCREASES (verifier can be STRICTER) as discrimination rises (sigma DOWN) -- a tighter good-step
distribution makes raising tau buy recall at lower false-positive (budget-burn) cost. P2 the c164 horizon effect
(tau* DECREASES with L, lenient on long tasks) PERSISTS at each sigma. P3 the VALUE of tuning tau (P_at_tau* minus
P at a fixed naive tau=0.5) is LARGER for a WORSE verifier (sharp tradeoff) and small for a great one (wide plateau).
"""
import numpy as np, json, os
from math import erf, sqrt
os.makedirs("runs/cycle177_oververify_discrimination", exist_ok=True)

A = 0.05
BUDGET_FRAC = 0.15
REPS = 8000

def norm_cdf(x, mu, s): return 0.5 * (1 + erf((x - mu) / (s * sqrt(2))))

def p_correct(L, tau, sigma, reps, rng):
    r = norm_cdf(tau, 0.0, sigma); f = norm_cdf(tau, 1.0, sigma)
    B = max(1, int(BUDGET_FRAC * L))
    is_bad = rng.random((reps, L)) < A
    caught = rng.random((reps, L)) < r              # for bad steps: caught w.p. r
    ff     = rng.random((reps, L)) < f              # for good steps: false-flag w.p. f
    fail_missed = (is_bad & ~caught).any(axis=1)    # any genuine error uncaught -> propagates
    flag = (is_bad & caught) | (~is_bad & ff)       # events that consume a retry (in order)
    cum = np.cumsum(flag, axis=1)                   # flag-rank up to & incl each position
    max_bad_rank = np.where(is_bad, cum, 0).max(axis=1)   # 0 if no bad steps
    budget_fail = max_bad_rank > B                  # a bad step's flag-rank exceeds budget -> can't fix
    correct = ~fail_missed & ~budget_fail
    return float(correct.mean()), round(r, 3), round(f, 3), B

# ---- Part A: correctness check — reproduce c164 at sigma=0.5 (targets: tau*=0.25, P=0.764/0.566/0.211 @ L=10/30/100) ----
C164_TAU = [-0.5, 0.0, 0.25, 0.5, 0.75, 1.0, 1.5]
print("=== Part A: reproduce c164 (sigma=0.5) — correctness check ===")
rng = np.random.default_rng(0)
repro = {}
for L in [10, 30, 100]:
    row = {t: p_correct(L, t, 0.5, REPS, rng)[0] for t in C164_TAU}
    ts = max(C164_TAU, key=lambda t: row[t])
    repro[L] = {"tau_star": ts, "P_at_taustar": round(row[ts], 3), "P_at_0.25": round(row[0.25], 3)}
    print(f"  L={L:>3}: tau*={ts} P*={row[ts]:.3f} | P(tau=0.25)={row[0.25]:.3f}  (c164: tau*=0.25, P={ {10:0.764,30:0.566,100:0.211}[L] })")

# ---- Part B: sweep verifier discrimination (sigma); finer tau grid; find tau*(sigma,L) ----
SIGMAS = [0.25, 0.35, 0.5, 0.75, 1.0]          # d' = 1/sigma = 4.0, 2.86, 2.0, 1.33, 1.0
TAU_GRID = [round(x, 3) for x in np.arange(-0.5, 1.501, 0.125)]
L_VALUES = [10, 30, 100]
NAIVE_TAU = 0.5                                 # a fixed "balanced" stringency (midpoint of the two means)
out = {"A": A, "BUDGET_FRAC": BUDGET_FRAC, "REPS": REPS, "SIGMAS": SIGMAS, "TAU_GRID": TAU_GRID,
       "L_VALUES": L_VALUES, "NAIVE_TAU": NAIVE_TAU, "partA_repro": repro, "sweep": {}}
print("\n=== Part B: tau*(sigma, L) — does a better verifier run stricter? ===")
print(f"{'sigma':>6} {'dprime':>6} " + " ".join(f"L={L}" for L in L_VALUES) + "    (tau* per L; tuning-value vs tau=0.5 in [])")
for sigma in SIGMAS:
    rng = np.random.default_rng(0)
    out["sweep"][sigma] = {}
    cells = []
    for L in L_VALUES:
        row = {t: p_correct(L, t, sigma, REPS, rng)[0] for t in TAU_GRID}
        ts = max(TAU_GRID, key=lambda t: row[t])
        pmax = row[ts]; pnaive = row[NAIVE_TAU]
        # plateau: tau-range within 0.5pct of pmax (honest about argmax noise on a flat top)
        near = [t for t in TAU_GRID if row[t] >= pmax - 0.005]
        out["sweep"][sigma][L] = {"tau_star": ts, "P_at_taustar": round(pmax, 3),
                                  "P_at_naive0.5": round(pnaive, 3), "tuning_value": round(pmax - pnaive, 3),
                                  "plateau_lo": min(near), "plateau_hi": max(near)}
        cells.append(f"t*={ts:>5} [{pmax-pnaive:+.3f}]")
    print(f"{sigma:>6} {1.0/sigma:>6.2f} " + "  ".join(cells))
json.dump(out, open("runs/cycle177_oververify_discrimination/score.json", "w"), indent=1)

print("\n--- tau*(sigma) at L=30 (the clean mid-horizon) ---")
for sigma in SIGMAS:
    s = out["sweep"][sigma][30]
    print(f"  sigma={sigma} (d'={1.0/sigma:.2f}): tau*={s['tau_star']} (plateau {s['plateau_lo']}..{s['plateau_hi']}), "
          f"P*={s['P_at_taustar']}, tuning-value={s['tuning_value']}")
print("PRE-REG: P1 tau* rises as sigma falls (better verifier->stricter); P2 tau* falls with L at each sigma; "
      "P3 tuning-value larger for worse verifier.")
print("DONE")
