"""Cycle 148 v2 — ADVERSARIAL robustness check of the design-budget finding under a SPARSE/ASYMMETRIC signal
(constitution sec5: don't trust the first result; replicate under a perturbation). v1 used symmetric g_true~N(0,1)
-> R*=D/2 (accept-if-positive). Is that an ARTIFACT of symmetry? Real design tweaks are SPARSE-positive: MOST are
neutral/slightly-harmful, a FEW genuinely help. Re-run with g_true ~ mixture: (1-p) * N(-0.1, 0.1) [neutral/harmful]
+ p * N(+1.0, 0.3) [the few real improvements], p=0.15.

Re-tests: P2 (true peaks then declines), P3 (does R* now move with A? peak-height grow with A?), P4 (fresh re-eval
recovers true), and the c147 gap (shrinks with A). PRE-REG: with sparse signal, R* should be SMALL (~ the count of
real positives, weighted by how well A resolves them); at LOW A more false-positives are swept in -> R* may be LARGER
and peak LOWER (noise inflates the apparent count of good tweaks); fresh re-eval still recovers true; gap still
shrinks with A. The DURABLE story (peak exists; A buys realized-quality + honesty; judge on fresh data) should hold.
"""
import numpy as np, json, os
os.makedirs("runs/cycle148_design_budget_crossover", exist_ok=True)
rng = np.random.default_rng(1)

D = 200; SIGMA_OBS = 3.0; A_VALUES = [10, 30, 100, 300, 1000]; REPS = 3000
P_SIGNAL = 0.15      # fraction of tweaks that genuinely help

def draw_gtrue(n):
    is_sig = rng.random(n) < P_SIGNAL
    g = np.where(is_sig, rng.normal(1.0, 0.3, n), rng.normal(-0.1, 0.1, n))
    return g

def summ(v):
    a = np.array(v, dtype=float); return float(a.mean()), float(1.96 * a.std(ddof=1) / np.sqrt(len(a)))

results = {"D": D, "P_SIGNAL": P_SIGNAL, "SIGMA_OBS": SIGMA_OBS, "A_values": A_VALUES, "REPS": REPS, "by_A": {}}
mean_curves = {}
for A in A_VALUES:
    sc = SIGMA_OBS / np.sqrt(A)
    Rstar, peak_true, believed_at, gap_at, reeval_at, oracle_pk, oracle_n = [], [], [], [], [], [], []
    tacc = np.zeros(D)
    for rep in range(REPS):
        g_true = draw_gtrue(D)
        g_hat = g_true + rng.normal(0, sc, D)
        g_hat2 = g_true + rng.normal(0, sc, D)
        order = np.argsort(-g_hat)
        tc = np.cumsum(g_true[order]); bc = np.cumsum(g_hat[order]); rc = np.cumsum(g_hat2[order])
        rs = int(np.argmax(tc))
        Rstar.append(rs + 1); peak_true.append(tc[rs]); believed_at.append(bc[rs])
        gap_at.append(bc[rs] - tc[rs]); reeval_at.append(rc[rs])
        oracle_pk.append(float(g_true[g_true > 0].sum())); oracle_n.append(int((g_true > 0).sum()))
        tacc += tc
    results["by_A"][A] = {"sigma_cal": float(sc), "Rstar": summ(Rstar), "peak_true": summ(peak_true),
        "believed_at_Rstar": summ(believed_at), "gap_at_Rstar": summ(gap_at), "reeval_at_Rstar": summ(reeval_at),
        "oracle_peak": summ(oracle_pk), "oracle_Rcount": summ(oracle_n)}
    mean_curves[A] = (tacc / REPS)

results["checks"] = {A: {"true_peaks_then_declines": bool(np.argmax(mean_curves[A]) < D - 1
                         and mean_curves[A][-1] < mean_curves[A].max() - 1e-9),
                         "peakR_meancurve": int(np.argmax(mean_curves[A]) + 1)} for A in A_VALUES}
json.dump(results, open("runs/cycle148_design_budget_crossover/score_sparse.json", "w"), indent=1)

print("=== #17c v2: SPARSE-positive signal (p_signal=0.15; most tweaks neutral/harmful) ===")
print(f"D={D} P_SIGNAL={P_SIGNAL} sigma_obs={SIGMA_OBS} REPS={REPS}")
print(f"{'A':>6} {'sig_cal':>8} {'R*(budget)':>15} {'peak_true':>14} {'believed@R*':>14} {'gap@R*':>13} {'reeval@R*':>13} {'oracle(n)':>16}")
for A in A_VALUES:
    b = results["by_A"][A]
    print(f"{A:>6} {b['sigma_cal']:>8.3f}"
          f" {b['Rstar'][0]:>6.1f}+-{b['Rstar'][1]:<5.1f}"
          f" {b['peak_true'][0]:>6.2f}+-{b['peak_true'][1]:<5.2f}"
          f" {b['believed_at_Rstar'][0]:>6.2f}+-{b['believed_at_Rstar'][1]:<5.2f}"
          f" {b['gap_at_Rstar'][0]:>5.2f}+-{b['gap_at_Rstar'][1]:<5.2f}"
          f" {b['reeval_at_Rstar'][0]:>5.2f}+-{b['reeval_at_Rstar'][1]:<5.2f}"
          f" {b['oracle_peak'][0]:>5.2f}(n={b['oracle_Rcount'][0]:.0f})")
print("--- checks ---")
for A in A_VALUES:
    print(f"A={A:>5}: true_peaks_then_declines={results['checks'][A]['true_peaks_then_declines']} peakR={results['checks'][A]['peakR_meancurve']}")
print("DONE")
