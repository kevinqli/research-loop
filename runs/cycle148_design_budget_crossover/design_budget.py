"""Cycle 148 (#17c — the SIGNAL-vs-OVERFIT crossover; HARDEN c147; benchmark-design pillar). c147 bounded the
PURE-NULL adaptive overfit (~sqrt(R/A)) — but real calibration sets have SIGNAL: some design tweaks genuinely
improve discrimination, some hurt. The ACTIONABLE question for a benchmark builder: how many adaptive
design iterations R can I afford on ONE calibration set of A agents before chasing calibration NOISE degrades the
TRUE discrimination? i.e. where is the classic overfitting PEAK, in the benchmark-DESIGN setting?

MODEL (the classic in-sample-vs-out-of-sample overfitting curve, mapped to adaptive benchmark design):
  - D candidate design tweaks; each has a TRUE effect on discrimination g_true,j ~ N(0, sigma_true^2) (symmetric:
    ~half help, ~half hurt — realistic for design tweaks). The builder cannot see g_true.
  - The CALIBRATION set (A agents) gives a NOISY estimate g_hat,j = g_true,j + N(0, sigma_cal^2), sigma_cal =
    sigma_obs / sqrt(A) (more calibration agents -> less noise, c147's A).
  - ADAPTIVE design = accept tweaks in DECREASING g_hat order (greedy: keep what looks best on calibration).
    After accepting the top-R by g_hat:  BELIEVED quality = sum of g_hat (in-sample, inflated);
    TRUE quality = sum of g_true (out-of-sample, what generalizes).
  - R* = argmax_R TRUE(R) = the AFFORDABLE design budget (accept this many tweaks; beyond it you sweep in
    noise-lucky HARMFUL tweaks and TRUE discrimination DECLINES while BELIEVED keeps rising).
  - FRESH-HOLDOUT re-eval (c147/c128 fix): a 2nd independent noisy estimate g_hat2 (indep noise) of the selected
    set -> should track TRUE (unbiased, since the fresh noise is independent of the g_hat selection) -> lets the
    builder LOCATE R* empirically.

PRE-REG (fix before peeking):
  P1: BELIEVED(R) rises MONOTONICALLY in R (overfit accumulates; you keep summing the largest g_hat).
  P2: TRUE(R) rises to a PEAK at R* then DECLINES (the overfitting curve) — the builder eventually accepts
      harmful tweaks whose high g_hat was noise.
  P3: R* and the peak-TRUE both INCREASE with A: more calibration agents -> g_hat ranks tweaks closer to g_true
      order -> you add them in true-benefit order -> peak LATER + HIGHER. As A->inf, R* -> #{g_true>0} (=D/2) and
      peak -> the ORACLE (sum of all positive g_true). At small A, R* << D/2 and peak << oracle.
  P4: the BELIEVED-TRUE gap at R* = the c147 overfit; the FRESH re-eval recovers TRUE (so R* is findable on
      fresh data, not on the calibration set the builder tuned against).
"""
import numpy as np, json, os
os.makedirs("runs/cycle148_design_budget_crossover", exist_ok=True)
rng = np.random.default_rng(0)

D = 200                 # candidate design tweaks
SIGMA_TRUE = 1.0        # spread of TRUE design effects (signal)
SIGMA_OBS = 3.0         # per-agent calibration noise scale (sigma_cal = SIGMA_OBS/sqrt(A))
A_VALUES = [10, 30, 100, 300, 1000]
REPS = 3000

def summarize_runs(per_rep):
    a = np.array(per_rep, dtype=float); m = a.mean(); se = a.std(ddof=1) / np.sqrt(len(a))
    return float(m), float(1.96 * se)

results = {"D": D, "SIGMA_TRUE": SIGMA_TRUE, "SIGMA_OBS": SIGMA_OBS, "A_values": A_VALUES, "REPS": REPS, "by_A": {}}
# also keep mean curves for one A for the monotonic/peak check
mean_curves = {}

for A in A_VALUES:
    sigma_cal = SIGMA_OBS / np.sqrt(A)
    Rstar, peak_true, true_at_Rstar, believed_at_Rstar, gap_at_Rstar = [], [], [], [], []
    reeval_at_Rstar, oracle_peak, oracle_Rcount = [], [], []
    true_curve_acc = np.zeros(D); believed_curve_acc = np.zeros(D); reeval_curve_acc = np.zeros(D)
    for rep in range(REPS):
        g_true = rng.normal(0, SIGMA_TRUE, size=D)
        g_hat = g_true + rng.normal(0, sigma_cal, size=D)        # calibration estimate
        g_hat2 = g_true + rng.normal(0, sigma_cal, size=D)        # FRESH independent re-eval
        order = np.argsort(-g_hat)                                # accept tweaks best-calibration-first
        true_c = np.cumsum(g_true[order])                         # TRUE(R)
        believed_c = np.cumsum(g_hat[order])                      # BELIEVED(R)
        reeval_c = np.cumsum(g_hat2[order])                       # fresh re-eval(R)
        rs = int(np.argmax(true_c))                               # R* (0-indexed; +1 = count)
        Rstar.append(rs + 1)
        peak_true.append(true_c[rs]); true_at_Rstar.append(true_c[rs])
        believed_at_Rstar.append(believed_c[rs]); gap_at_Rstar.append(believed_c[rs] - true_c[rs])
        reeval_at_Rstar.append(reeval_c[rs])
        # oracle: accept exactly the positive-true tweaks
        oracle_peak.append(float(g_true[g_true > 0].sum())); oracle_Rcount.append(int((g_true > 0).sum()))
        true_curve_acc += true_c; believed_curve_acc += believed_c; reeval_curve_acc += reeval_c
    results["by_A"][A] = {
        "sigma_cal": float(sigma_cal), "snr_true_over_cal": float(SIGMA_TRUE / sigma_cal),
        "Rstar": summarize_runs(Rstar), "peak_true": summarize_runs(peak_true),
        "believed_at_Rstar": summarize_runs(believed_at_Rstar), "gap_at_Rstar": summarize_runs(gap_at_Rstar),
        "reeval_at_Rstar": summarize_runs(reeval_at_Rstar),
        "oracle_peak": summarize_runs(oracle_peak), "oracle_Rcount": summarize_runs(oracle_Rcount),
    }
    mean_curves[A] = {"true": (true_curve_acc / REPS).tolist(),
                      "believed": (believed_curve_acc / REPS).tolist(),
                      "reeval": (reeval_curve_acc / REPS).tolist()}

# monotonicity of believed + peak-then-decline of true, checked on the MEAN curves
checks = {}
for A in A_VALUES:
    tc = np.array(mean_curves[A]["true"]); bc = np.array(mean_curves[A]["believed"])
    checks[A] = {"believed_monotonic_incr": bool(np.all(np.diff(bc) > -1e-9)),
                 "true_peaks_then_declines": bool(np.argmax(tc) < D - 1 and tc[-1] < tc[int(np.argmax(tc))] - 1e-9),
                 "true_peak_R_meancurve": int(np.argmax(tc) + 1), "true_final_minus_peak": float(tc[-1] - tc.max())}
results["checks"] = checks
json.dump(results, open("runs/cycle148_design_budget_crossover/score.json", "w"), indent=1)

print("=== #17c: adaptive design BUDGET (signal-vs-overfit crossover) — overfitting curve in benchmark design ===")
print(f"D={D} sigma_true={SIGMA_TRUE} sigma_obs={SIGMA_OBS} REPS={REPS}")
print(f"{'A':>6} {'snr':>6} {'R*(budget)':>16} {'peak_true':>16} {'believed@R*':>16} {'gap@R*':>14} {'reeval@R*':>14} {'oracle':>14}")
for A in A_VALUES:
    b = results["by_A"][A]
    print(f"{A:>6} {b['snr_true_over_cal']:>6.2f}"
          f" {b['Rstar'][0]:>7.1f}+-{b['Rstar'][1]:<6.1f}"
          f" {b['peak_true'][0]:>7.2f}+-{b['peak_true'][1]:<6.2f}"
          f" {b['believed_at_Rstar'][0]:>7.2f}+-{b['believed_at_Rstar'][1]:<6.2f}"
          f" {b['gap_at_Rstar'][0]:>6.2f}+-{b['gap_at_Rstar'][1]:<5.2f}"
          f" {b['reeval_at_Rstar'][0]:>6.2f}+-{b['reeval_at_Rstar'][1]:<5.2f}"
          f" {b['oracle_peak'][0]:>6.2f}+-{b['oracle_peak'][1]:<5.2f}")
print("--- checks (on mean curves) ---")
for A in A_VALUES:
    c = checks[A]
    print(f"A={A:>5}: believed_monotonic={c['believed_monotonic_incr']} true_peaks_then_declines={c['true_peaks_then_declines']} "
          f"peakR(meancurve)={c['true_peak_R_meancurve']} (final-peak)={c['true_final_minus_peak']:.2f}")
print("PRE-REG: P1 believed monotonic; P2 true peaks then declines; P3 R* & peak grow with A (-> oracle); "
      "P4 reeval@R* ~ true@R* (fresh re-eval recovers true).")
print("DONE")
