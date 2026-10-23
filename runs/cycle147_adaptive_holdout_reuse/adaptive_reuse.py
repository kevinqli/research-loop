"""Cycle 147 (#17b — the ADAPTIVE benchmark-builder curse; COMPOSE c129 x c115 x c50; diversified OFF the
real-agent/tail/framing thread per the portfolio note). c129 showed REUSING a held-out calibration set across R
benchmark-design rounds re-curses it — but ONLY via the NON-ADAPTIVE mechanism (draw R iid candidate designs, keep
the best-on-held-out) -> the gap grows like the order statistic over R rounds = ~sqrt(ln R) (SLOW, sub-polynomial).
The STRONGER, realistic effect is ADAPTIVE (Dwork et al. 2015, "Preserving Statistical Validity in Adaptive Data
Analysis"): the builder uses the held-out's FEEDBACK to CONSTRUCT the next design -> overfit grows ~sqrt(R/A),
POLYNOMIALLY faster. c129 thus UNDERSTATES the danger of an adaptively-reused held-out.

CANONICAL, EXACTLY-SIMULABLE construction (Dwork Fig-1 feature-aggregation, mapped to benchmark design):
  - A held-out of A "calibration agents"; a label y in {+-1} (pure NOISE - the design has NO real signal, so true
    score = 0 by construction; ANY held-out advantage above 0 IS overfit, no confound).
  - d candidate "design knobs", each a +-1 vector INDEPENDENT of y (no signal). The builder touches the holdout R
    times (R <= d).
  - NON-ADAPTIVE (c129 pick-best): score R knobs on the holdout, REPORT the single best holdout corr = max_j c_j
    ~ sqrt(2 ln R / A).  [the c129 order-statistic]
  - ADAPTIVE (Dwork): use the holdout to learn each knob's sign, AGGREGATE all R by their holdout-sign -> report the
    aggregate's holdout corr = (1/sqrt(R)) sum |c_j| ~ sqrt(2R/(pi A)).  [the effect c129 didn't model]
  - THRESHOLDOUT (c115/reusable-holdout DEFENSE): the builder also has a TRAINING set (m samples); Thresholdout
    returns the holdout corr ONLY when it differs from the training corr by > a noise threshold, else returns the
    training corr -> the builder ends up aggregating by the TRAINING sign -> holdout overfit ~ 0 (validity restored).
  - POSITIVE CONTROL (c128/c7 held-out re-eval): re-score the believed-best/aggregate design on a FRESH holdout ->
    advantage collapses to ~0 (confirms the inflation is overfit, not signal; the independent re-eval is the fix).

PRE-REG (fix before peeking):
  P1: NON-ADAPTIVE overfit grows like sqrt(ln R) -> log-log slope d(log overfit)/d(log R) -> ~0 (sub-polynomial).
  P2: ADAPTIVE overfit grows like sqrt(R) -> log-log slope ~ +0.5 (POLYNOMIAL); adaptive >> non-adaptive, ratio
      ~ sqrt(R/ln R) -> grows without bound in R. => c129 (non-adaptive) UNDERSTATES adaptive-reuse danger.
  P3: THRESHOLDOUT keeps adaptive overfit ~flat near the training-set floor (defense works); FRESH-holdout re-eval
      of every arm -> ~0 (the independent re-eval, c128, is unconditionally robust).
  Predicted exponents: non-adaptive ~0 (really 0.5*loglog), adaptive ~0.5.
"""
import numpy as np, json, os
os.makedirs("runs/cycle147_adaptive_holdout_reuse", exist_ok=True)
rng = np.random.default_rng(0)

A = 500          # held-out calibration-set size
M = 500          # training-set size (for the Thresholdout defense)
D = 400          # number of candidate design knobs (features), >= max R
REPS = 600
R_VALUES = [2, 4, 8, 16, 32, 64, 128, 256, 400]
THRESH = 4.0 / np.sqrt(A)   # Thresholdout margin ~ a few holdout-SEs

def corr(features, y, n):
    # features: (n, k) in +-1 ; y: (n,) in +-1 ; returns (k,) per-feature corr = (1/n) sum x*y
    return features.T @ y / n

res = {r: {"nonadapt": [], "adapt": [], "thresholdout": [],
           "adapt_fresh": [], "nonadapt_fresh": []} for r in R_VALUES}

for rep in range(REPS):
    # one experiment: holdout (A), training (M), fresh holdout (A), all with PURE-NOISE features (no signal)
    y_h = rng.choice([-1.0, 1.0], size=A)
    y_t = rng.choice([-1.0, 1.0], size=M)
    y_f = rng.choice([-1.0, 1.0], size=A)            # fresh independent holdout
    X_h = rng.choice([-1.0, 1.0], size=(A, D))        # knobs independent of every label (true signal = 0)
    X_t = rng.choice([-1.0, 1.0], size=(M, D))
    X_f = rng.choice([-1.0, 1.0], size=(A, D))
    c_h = corr(X_h, y_h, A)      # holdout corr per knob
    c_t = corr(X_t, y_t, M)      # training corr per knob
    c_f = corr(X_f, y_f, A)      # fresh-holdout corr per knob

    for R in R_VALUES:
        ch, ct, cf = c_h[:R], c_t[:R], c_f[:R]
        # NON-ADAPTIVE (c129): report the single best holdout corr over R candidate knobs
        res[R]["nonadapt"].append(np.max(ch))
        # its fresh-holdout re-eval (positive control): the SAME knob, scored on the fresh holdout
        res[R]["nonadapt_fresh"].append(cf[np.argmax(ch)])
        # ADAPTIVE (Dwork): aggregate all R knobs by their HOLDOUT sign -> aggregate holdout corr
        agg_h = np.sum(np.sign(ch) * ch) / np.sqrt(R)            # = (1/sqrt R) sum |c_h|
        res[R]["adapt"].append(agg_h)
        # the SAME aggregate (built with holdout signs) re-scored on the FRESH holdout (positive control)
        agg_fresh = np.sum(np.sign(ch) * cf) / np.sqrt(R)
        res[R]["adapt_fresh"].append(agg_fresh)
        # THRESHOLDOUT (c115 defense): return holdout corr only if |c_h - c_t| > THRESH (+Laplace noise), else c_t;
        # builder aggregates by the sign of the RETURNED value
        noise = rng.laplace(0, THRESH / 2.0, size=R)
        returned = np.where(np.abs(ch - ct) > THRESH + noise, ch, ct)
        agg_thr = np.sum(np.sign(returned) * ch) / np.sqrt(R)     # holdout corr of the thresholdout-built aggregate
        res[R]["thresholdout"].append(agg_thr)

def stat(v):
    a = np.array(v); m = a.mean(); se = a.std(ddof=1) / np.sqrt(len(a))
    return float(m), float(1.96 * se)

summary = {"A": A, "M": M, "D": D, "REPS": REPS, "THRESH": float(THRESH), "R_values": R_VALUES, "by_R": {}}
for R in R_VALUES:
    summary["by_R"][R] = {k: {"mean": stat(res[R][k])[0], "ci95": stat(res[R][k])[1]} for k in res[R]}

# log-log scaling exponents (slope of log(mean overfit) vs log R) for non-adaptive vs adaptive
logR = np.log(np.array(R_VALUES, dtype=float))
def slope(key):
    ys = np.log(np.array([summary["by_R"][R][key]["mean"] for R in R_VALUES]))
    return float(np.polyfit(logR, ys, 1)[0])
summary["loglog_slope_nonadapt"] = slope("nonadapt")
summary["loglog_slope_adapt"] = slope("adapt")
summary["loglog_slope_thresholdout"] = slope("thresholdout")
# ratio adaptive/nonadaptive at the largest R
Rmax = R_VALUES[-1]
summary["ratio_adapt_over_nonadapt_at_Rmax"] = (summary["by_R"][Rmax]["adapt"]["mean"]
                                                / summary["by_R"][Rmax]["nonadapt"]["mean"])
# theory references
summary["theory"] = {"nonadapt_sqrt_2lnR_over_A_at_Rmax": float(np.sqrt(2*np.log(Rmax)/A)),
                     "adapt_sqrt_2Rmax_over_piA": float(np.sqrt(2*Rmax/(np.pi*A)))}

json.dump(summary, open("runs/cycle147_adaptive_holdout_reuse/score.json", "w"), indent=1)

print("=== ADAPTIVE vs NON-ADAPTIVE held-out reuse (true signal = 0; any advantage is overfit) ===")
print(f"A(holdout)={A} M(train)={M} D(knobs)={D} REPS={REPS} THRESH={THRESH:.4f}")
print(f"{'R':>5} {'nonadapt':>18} {'adaptive':>18} {'thresholdout':>18} {'adapt_fresh':>14} {'nonad_fresh':>14}")
for R in R_VALUES:
    b = summary["by_R"][R]
    print(f"{R:>5} {b['nonadapt']['mean']:>8.4f}+-{b['nonadapt']['ci95']:<7.4f}"
          f" {b['adapt']['mean']:>8.4f}+-{b['adapt']['ci95']:<7.4f}"
          f" {b['thresholdout']['mean']:>8.4f}+-{b['thresholdout']['ci95']:<7.4f}"
          f" {b['adapt_fresh']['mean']:>6.4f}+-{b['adapt_fresh']['ci95']:<5.4f}"
          f" {b['nonadapt_fresh']['mean']:>6.4f}+-{b['nonadapt_fresh']['ci95']:<5.4f}")
print("---")
print(f"loglog_slope_nonadapt:    {summary['loglog_slope_nonadapt']:.3f}   (PRE-REG P1: ~0, sub-polynomial / sqrt(lnR))")
print(f"loglog_slope_adapt:       {summary['loglog_slope_adapt']:.3f}   (PRE-REG P2: ~0.5, polynomial / sqrt(R))")
print(f"loglog_slope_thresholdout:{summary['loglog_slope_thresholdout']:.3f}   (PRE-REG P3: ~0, defense holds)")
print(f"ratio adapt/nonadapt @R={Rmax}: {summary['ratio_adapt_over_nonadapt_at_Rmax']:.2f}")
print(f"theory nonadapt sqrt(2lnR/A)@Rmax={summary['theory']['nonadapt_sqrt_2lnR_over_A_at_Rmax']:.4f} ; "
      f"adapt sqrt(2R/piA)@Rmax={summary['theory']['adapt_sqrt_2Rmax_over_piA']:.4f}")
print("DONE")
