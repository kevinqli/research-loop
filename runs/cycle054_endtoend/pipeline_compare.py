"""Cycle 54: does the RECOMMENDED design compose? End-to-end pipeline comparison (synthesizes cycles 7/45/48/50/51)
under a FIXED budget B. Produces the single headline ROI number for a parallel-agent lab.
  NAIVE      : K=B REDUNDANT candidates (corr ρ_hi, cheap) → select by IN-SAMPLE metric → REPORT the in-sample value.
  RECOMMENDED: DIVERSE candidates (corr ρ_lo, cost κ → fewer) → select by a SECRET held-out re-eval → REPORT on a
               SEPARATE secret set (split select/report, cycle 8).
Measure, for each: TRUE perf of the final pick (what you actually get) AND reporting gap (reported − true; honesty).
Pre-reg: RECOMMENDED has (a) higher TRUE perf (diversity → better candidate available + held-out → picks it) and
(b) ~0 reporting gap (honest), vs NAIVE's lower true perf + large inflated report. numpy-only, REPS reps."""
import numpy as np
from math import sqrt
rng = np.random.default_rng(0)
REPS, B, MU, S_TRUE, S = 5000, 100, 0.80, 0.05, 0.05
RHO_HI, RHO_LO, KAPPA = 0.90, 0.0, 2.0   # redundant vs diverse; diversity cost

def corr_theta(K, rho):
    g = rng.normal(0,1); z = rng.normal(0,1,K)
    return MU + S_TRUE*(sqrt(rho)*g + sqrt(1-rho)*z)

def run():
    naive_true=np.empty(REPS); naive_rep=np.empty(REPS); rec_true=np.empty(REPS); rec_rep=np.empty(REPS); oracle=np.empty(REPS)
    K_naive = B
    K_rec = max(2, int(round(B/(1+KAPPA*(1-RHO_LO)))))   # diverse candidates cost more → fewer (≈33)
    for r in range(REPS):
        # NAIVE: redundant candidates, select by in-sample, report in-sample
        th_n = corr_theta(K_naive, RHO_HI); val_n = th_n + rng.normal(0,S,K_naive)
        s = int(np.argmax(val_n)); naive_true[r]=th_n[s]; naive_rep[r]=val_n[s]
        # RECOMMENDED: diverse candidates, select by SECRET held-out, report on a SEPARATE secret set
        th_r = corr_theta(K_rec, RHO_LO)
        sel_set = th_r + rng.normal(0,S,K_rec)            # held-out-A used to SELECT (independent of any in-sample)
        s2 = int(np.argmax(sel_set)); rec_true[r]=th_r[s2]
        rec_rep[r] = th_r[s2] + rng.normal(0,S)          # held-out-B (separate secret) used to REPORT
        oracle[r] = th_r.max()                            # best truly-available in the diverse pool
    ci=lambda x:1.96*x.std(ddof=1)/sqrt(REPS)
    return (K_naive,K_rec, naive_true.mean(),ci(naive_true), naive_rep.mean(),
            rec_true.mean(),ci(rec_true), rec_rep.mean(), oracle.mean())

K_naive,K_rec, nt,nt_ci, nr, rt,rt_ci, rr, orac = run()
print(f"budget B={B}: NAIVE={K_naive} redundant (ρ={RHO_HI}); RECOMMENDED={K_rec} diverse (ρ={RHO_LO}, cost κ={KAPPA}); reps={REPS}\n")
print(f"{'pipeline':>14} | {'TRUE perf of pick':>18} | {'REPORTED':>9} | {'reporting gap':>13}")
print(f"{'NAIVE':>14} | {nt:.4f} ± {nt_ci:.4f} | {nr:>9.4f} | {nr-nt:>+13.4f}")
print(f"{'RECOMMENDED':>14} | {rt:.4f} ± {rt_ci:.4f} | {rr:>9.4f} | {rr-rt:>+13.4f}")
print(f"{'(oracle: best in diverse pool)':>30} {orac:.4f}")
import json; json.dump({"K_naive":K_naive,"K_rec":K_rec,"naive_true":nt,"naive_rep":nr,"rec_true":rt,"rec_rep":rr,
  "oracle":orac,"true_uplift":rt-nt,"naive_gap":nr-nt,"rec_gap":rr-rt},open("runs/cycle054_endtoend/score.json","w"),indent=1)
print("\n--- read the rows above ---")
print(f"TRUE-perf uplift: RECOMMENDED {rt:.4f} vs NAIVE {nt:.4f} = +{rt-nt:.4f} ({100*(rt-nt)/S_TRUE:.0f}% of the candidate σ; {100*(rt/nt-1):+.1f}% relative).")
print(f"REPORTING honesty: NAIVE over-reports by {nr-nt:+.4f} (hallucinated progress); RECOMMENDED gap {rr-rt:+.4f} (~0, honest).")
print("=> the recommended design (DIVERSE candidates + SECRET held-out select/report) composes: it delivers higher TRUE performance AND an honest report, vs naive's lower true perf + inflated report — even though it runs 3x FEWER candidates. Diversity + independent secret verification > raw parallelism + self-report.")
print("DONE")
