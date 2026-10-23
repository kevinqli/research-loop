"""Cycle 147 v2 — FIX the Thresholdout (P3): v1's threshold had over-large noise (Laplace scale THRESH/2) and NO
BUDGET, so ~7-15pct of holdout signs leaked -> it cut the overfit CONSTANT ~5x but kept the sqrt(R) EXPONENT
(slope 0.428). The CANONICAL Thresholdout (Dwork-Feldman-Hardt-Pitassi-Roth / "reusable holdout", Science 2015) has
(a) SMALL threshold noise and (b) a hard BUDGET B on the number of times it may return the holdout value; once the
budget is spent it stops consulting the holdout. The budget CAPS the number of leaked signs at B regardless of R ->
the aggregate leak -> (1/sqrt R)*B*E|c_h| -> 0, leaving only the ~N(0,1/A) floor (FLAT in R). This is c115's defense
(Ladder/reusable-holdout): you CAN reuse a calibration set safely IF you protect it.

Compares (same null construction as v1, true signal = 0):
  NON-ADAPTIVE (c129 best-of-R)   | ADAPTIVE (Dwork aggregate)   | THRESHOLDOUT-BUDGETED (c115 defense)
  + FRESH-HOLDOUT re-eval of the adaptive aggregate (c128/c7 fix) — should be ~0 at every R.

PRE-REG (P3 only; P1/P2 already confirmed in v1): the BUDGETED Thresholdout's overfit is ~FLAT in R (log-log slope
~0, not 0.5) and bounded near the no-leak floor ~1/sqrt(A)=0.045; it does NOT inherit the adaptive sqrt(R) growth.
"""
import numpy as np, json, os
os.makedirs("runs/cycle147_adaptive_holdout_reuse", exist_ok=True)
rng = np.random.default_rng(0)

A, M, D, REPS = 500, 500, 400, 600
R_VALUES = [2, 4, 8, 16, 32, 64, 128, 256, 400]
THRESH = 4.0 / np.sqrt(A)        # ~4 holdout-SE of c_h (SD 1/sqrt(A)=0.045)
NOISE = THRESH / 8.0             # SMALL threshold noise (v1 used THRESH/2 -> too leaky)
BUDGET = 40                      # max holdout consultations (canonical: ~the protection budget)

def corr(X, y, n): return X.T @ y / n

res = {r: {"nonadapt": [], "adapt": [], "thresholdout": [], "adapt_fresh": []} for r in R_VALUES}
leak_counts = {r: [] for r in R_VALUES}

for rep in range(REPS):
    y_h = rng.choice([-1.0, 1.0], size=A); y_t = rng.choice([-1.0, 1.0], size=M); y_f = rng.choice([-1.0, 1.0], size=A)
    X_h = rng.choice([-1.0, 1.0], size=(A, D)); X_t = rng.choice([-1.0, 1.0], size=(M, D)); X_f = rng.choice([-1.0, 1.0], size=(A, D))
    c_h, c_t, c_f = corr(X_h, y_h, A), corr(X_t, y_t, M), corr(X_f, y_f, A)
    for R in R_VALUES:
        ch, ct, cf = c_h[:R], c_t[:R], c_f[:R]
        res[R]["nonadapt"].append(np.max(ch))
        res[R]["adapt"].append(np.sum(np.sign(ch) * ch) / np.sqrt(R))
        res[R]["adapt_fresh"].append(np.sum(np.sign(ch) * cf) / np.sqrt(R))
        # budgeted Thresholdout: walk knobs in order; return c_h (leak sign) only if |c_h-c_t|>THRESH+noise AND budget left
        returned = ct.copy(); budget = BUDGET; leaks = 0
        noise = rng.laplace(0, NOISE, size=R)
        for j in range(R):
            if budget > 0 and abs(ch[j] - ct[j]) > THRESH + noise[j]:
                returned[j] = ch[j]; budget -= 1; leaks += 1
            # else keep ct[j]  (do NOT consult the holdout sign)
        res[R]["thresholdout"].append(np.sum(np.sign(returned) * ch) / np.sqrt(R))
        leak_counts[R].append(leaks)

def stat(v):
    a = np.array(v); return float(a.mean()), float(1.96 * a.std(ddof=1) / np.sqrt(len(a)))

summary = {"A": A, "M": M, "D": D, "REPS": REPS, "THRESH": float(THRESH), "NOISE": float(NOISE),
           "BUDGET": BUDGET, "R_values": R_VALUES, "by_R": {}}
for R in R_VALUES:
    summary["by_R"][R] = {k: {"mean": stat(res[R][k])[0], "ci95": stat(res[R][k])[1]} for k in res[R]}
    summary["by_R"][R]["mean_leaks"] = float(np.mean(leak_counts[R]))
logR = np.log(np.array(R_VALUES, dtype=float))
def slope(key):
    ys = np.log(np.abs(np.array([summary["by_R"][R][key]["mean"] for R in R_VALUES])))
    return float(np.polyfit(logR, ys, 1)[0])
summary["loglog_slope_thresholdout_budgeted"] = slope("thresholdout")
summary["loglog_slope_adapt"] = slope("adapt")
json.dump(summary, open("runs/cycle147_adaptive_holdout_reuse/score_v2.json", "w"), indent=1)

print("=== v2: BUDGETED Thresholdout (c115 defense) vs adaptive/non-adaptive (true signal = 0) ===")
print(f"A={A} M={M} D={D} REPS={REPS} THRESH={THRESH:.4f} NOISE={NOISE:.4f} BUDGET={BUDGET}")
print(f"{'R':>5} {'nonadapt':>16} {'adaptive':>16} {'thresh-budget':>16} {'adapt_fresh':>14} {'leaks':>7}")
for R in R_VALUES:
    b = summary["by_R"][R]
    print(f"{R:>5} {b['nonadapt']['mean']:>8.4f}+-{b['nonadapt']['ci95']:<6.4f}"
          f" {b['adapt']['mean']:>8.4f}+-{b['adapt']['ci95']:<6.4f}"
          f" {b['thresholdout']['mean']:>8.4f}+-{b['thresholdout']['ci95']:<6.4f}"
          f" {b['adapt_fresh']['mean']:>6.4f}+-{b['adapt_fresh']['ci95']:<5.4f}"
          f" {b['mean_leaks']:>7.2f}")
print("---")
print(f"loglog_slope_adapt:                {summary['loglog_slope_adapt']:.3f}   (sqrt(R), polynomial)")
print(f"loglog_slope_thresholdout_budgeted:{summary['loglog_slope_thresholdout_budgeted']:.3f}   (PRE-REG: ~0, FLAT/bounded)")
print(f"thresh-budget floor ~1/sqrt(A)={1/np.sqrt(A):.4f}")
print("DONE")
