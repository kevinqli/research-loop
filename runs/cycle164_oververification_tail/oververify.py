"""Cycle 164 (NEW mechanism — OVER-verification has its OWN exponential tail on long horizons; the MIRROR of c135;
COMPOSE c135 x c93 x c86; tail + verification pillars). c135: per-step ERRORS compound (s^L) -> decompose-verify-retry
fixes it. The unasked mirror: per-step VERIFICATION FALSE-POSITIVES also compound. A too-strict per-step verifier
flags/redoes GOOD steps; on a long horizon those false alarms BURN the retry budget that genuine errors need -> so the
optimal per-step stringency should DECREASE with horizon (be LENIENT on long tasks). c93 did recall (false-NEGATIVES),
c86 stringency single-step; the LONG-HORIZON compounding of false-POSITIVES is the new part.

MODEL: L-step task, genuine per-step error rate a. Per-step verifier stringency tau: good-step score ~N(1,sigma),
bad-step score ~N(0,sigma), FLAG if score<tau. So recall r=P(N(0,sigma)<tau) (catch a bad step) rises with tau, AND
false-positive f=P(N(1,sigma)<tau) (flag a good step) rises with tau. Retry budget B (scales with L). Process steps in
order: a FLAG (genuine-caught OR false-alarm) consumes 1 retry IF budget>0 (the step is redone). A genuine error: caught
& budget>0 -> fixed; caught & budget=0 -> PROPAGATES (wrong); missed -> PROPAGATES (wrong). A false alarm: budget>0 ->
wastes a retry (redo, still good); budget=0 -> shipped as-is (good, harmless). So over-verification BURNS the budget on
false alarms, leaving none for real errors. Metric: P(trajectory correct) vs tau, for several L. Find tau*(L).

PRE-REG: P1 over-verification (high tau -> high f) on long L burns the budget (false-flags ~L*f) -> P(correct) drops.
P2 the optimal stringency tau*(L) DECREASES with horizon L (long tasks -> LENIENT verifier). P3 beyond some horizon,
STRICT per-step verification is net-NEGATIVE vs a lenient one.
"""
import numpy as np, json, os
from math import erf, sqrt
os.makedirs("runs/cycle164_oververification_tail", exist_ok=True)
rng = np.random.default_rng(0)

A = 0.05               # genuine per-step error rate
SIGMA = 0.5            # verifier noise (good~N(1,sig), bad~N(0,sig))
BUDGET_FRAC = 0.15     # retry budget B = BUDGET_FRAC * L
TAU_VALUES = [-0.5, 0.0, 0.25, 0.5, 0.75, 1.0, 1.5]   # stringency: flag if score<tau (higher=stricter)
L_VALUES = [10, 30, 100, 300]
REPS = 4000

def norm_cdf(x, mu, s): return 0.5 * (1 + erf((x - mu) / (s * sqrt(2))))

def trajectory_correct(L, tau, reps):
    r = norm_cdf(tau, 0.0, SIGMA)    # recall on bad steps (score<tau)
    f = norm_cdf(tau, 1.0, SIGMA)    # false-positive on good steps
    B = max(1, int(BUDGET_FRAC * L))
    ok = 0
    for _ in range(reps):
        budget = B; correct = True
        is_bad = rng.random(L) < A
        u = rng.random(L)
        for i in range(L):
            if is_bad[i]:
                caught = u[i] < r
                if caught and budget > 0:
                    budget -= 1            # redo, fixed
                else:
                    correct = False; break  # missed OR no budget -> propagates
            else:
                false_flag = u[i] < f
                if false_flag and budget > 0:
                    budget -= 1            # wasted retry on a good step
                # else: shipped as-is (good) — harmless but no budget consumed if 0
        if correct: ok += 1
    return ok / reps, r, f, B

out = {"A": A, "SIGMA": SIGMA, "BUDGET_FRAC": BUDGET_FRAC, "TAU_VALUES": TAU_VALUES, "L_VALUES": L_VALUES, "REPS": REPS, "by_L": {}}
print(f"=== OVER-verification tail (a={A}, sigma={SIGMA}, budget={BUDGET_FRAC}*L, REPS={REPS}) ===")
print(f"{'L':>5} " + " ".join(f"tau={t:>4}" for t in TAU_VALUES) + "   tau*  (r,f at tau*)")
for L in L_VALUES:
    row = {}
    for t in TAU_VALUES:
        p, r, f, B = trajectory_correct(L, t, REPS)
        row[t] = {"P_correct": round(p,3), "recall": round(r,3), "false_pos": round(f,3)}
    taustar = max(TAU_VALUES, key=lambda t: row[t]["P_correct"])
    out["by_L"][L] = {"row": row, "tau_star": taustar, "P_at_taustar": row[taustar]["P_correct"],
                      "r_at_taustar": row[taustar]["recall"], "f_at_taustar": row[taustar]["false_pos"]}
    cells = " ".join(f"{row[t]['P_correct']:>8.3f}" for t in TAU_VALUES)
    print(f"{L:>5} {cells}   tau*={taustar}  (r={row[taustar]['recall']:.2f},f={row[taustar]['false_pos']:.2f})")
json.dump(out, open("runs/cycle164_oververification_tail/score.json", "w"), indent=1)
print("--- tau*(L) ---", [out["by_L"][L]["tau_star"] for L in L_VALUES])
print("PRE-REG: P1 high-tau P(correct) drops on long L (budget burned); P2 tau*(L) DECREASES with L (lenient on long tasks); P3 strict net-negative at long L.")
print("DONE")
