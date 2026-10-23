"""BENCHMARK_DEMO — the quant-agent-eval centerpiece, runnable in one file (built cycle 166).
"The benchmark IS the product." This demonstrates the loop's whole thesis on toy data end-to-end:
  getting a number is easy (every strategy looks great IN-SAMPLE) -> knowing it's real is hard
  (only an INDEPENDENT held-out + a MULTI-REGIME survival metric separate real alpha from lucky overfit).
Run:  python BENCHMARK_DEMO.py
It generates toy agent-strategies (half with true alpha, half pure overfit), shows the WINNER'S CURSE on the
in-sample leaderboard (c7/c28), and shows the FIXES (held-out re-eval c50; multi-regime survival c30) discriminate
real-from-overfit where in-sample cannot. Canonical consolidation of c7 / c28 / c30 / c50 / c54 into the actual eval.
"""
import numpy as np, json

SEED = 0
N_REAL, N_OVERFIT = 300, 300     # toy agent-strategies
N_ERAS = 6                        # disjoint market regimes/eras
NOISE = 0.7                       # per-era Sharpe noise
INSAMPLE_FILTER = 0.8            # "passed the backtest" threshold on era-0 (in-sample) Sharpe
rng = np.random.default_rng(SEED)

def auc(pos, neg):
    """P(a random real > a random overfit) on a metric — discrimination of real-vs-overfit."""
    pos, neg = np.asarray(pos), np.asarray(neg)
    return float(np.mean(rng.choice(pos, 20000) > rng.choice(neg, 20000)))

# --- 1. generate strategies: REAL have true forward alpha; OVERFIT have ZERO true alpha ---
true_alpha = np.concatenate([rng.uniform(0.3, 1.2, N_REAL), np.zeros(N_OVERFIT)])
is_real = np.concatenate([np.ones(N_REAL, bool), np.zeros(N_OVERFIT, bool)])
# per-era Sharpe = true alpha + iid noise (era 0 = in-sample/backtest; eras 1..5 = held-out future regimes)
era_sharpe = true_alpha[:, None] + rng.normal(0, NOISE, (N_REAL + N_OVERFIT, N_ERAS))

# --- 2. the realistic setup: you only SEE strategies that PASSED the backtest (era-0 Sharpe > filter) ---
# This selects ALL the genuinely-good (real) PLUS the LUCKY overfit (era-0 noise pushed them over). The classic trap.
passed = era_sharpe[:, 0] > INSAMPLE_FILTER
ksharpe, kreal, ktrue = era_sharpe[passed], is_real[passed], true_alpha[passed]
n_pass_real, n_pass_overfit = int(kreal.sum()), int((~kreal).sum())

# --- 3. three eval metrics on the PASSED pool ---
m_insample = ksharpe[:, 0]                 # rank by the backtest (era 0) — the naive leaderboard
m_heldout  = ksharpe[:, 1:].mean(axis=1)   # mean over held-out eras 1..5 (c50: report on data you didn't select on)
m_survival = ksharpe[:, 1:].min(axis=1)    # MIN across held-out eras (c30 multi-regime survival: was it real EVERY regime)

auc_insample = auc(m_insample[kreal], m_insample[~kreal])
auc_heldout  = auc(m_heldout[kreal],  m_heldout[~kreal])
auc_survival = auc(m_survival[kreal], m_survival[~kreal])

# --- 4. the WINNER'S CURSE on the in-sample leaderboard: who's at the top, and does it hold up? ---
top = np.argsort(-m_insample)[:10]         # top-10 by backtest
top_real_frac = float(kreal[top].mean())
top_insample_mean = float(m_insample[top].mean())
top_heldout_mean  = float(m_heldout[top].mean())   # the SAME strategies, scored on held-out -> the curse collapse

summary = {"n_passed_backtest": int(passed.sum()), "of_which_real": n_pass_real, "of_which_OVERFIT_lucky": n_pass_overfit,
  "AUC_real_vs_overfit": {"in_sample_leaderboard": round(auc_insample,3), "held_out_re_eval": round(auc_heldout,3),
                          "multi_regime_survival": round(auc_survival,3)},
  "winners_curse_top10_by_backtest": {"real_fraction": top_real_frac, "in_sample_Sharpe": round(top_insample_mean,2),
                                      "SAME_strategies_held_out_Sharpe": round(top_heldout_mean,2)}}
import os
if os.path.isdir("runs"):
    json.dump(summary, open("runs/benchmark_demo_score.json", "w"), indent=1)

print("=" * 74)
print("BENCHMARK DEMO — separating real alpha from lucky overfit (the quant-agent-eval)")
print("=" * 74)
print(f"Generated {N_REAL} real-alpha + {N_OVERFIT} pure-overfit toy strategies; {N_ERAS} disjoint eras.")
print(f"\nSTEP 1 - the backtest filter (era-0 Sharpe > {INSAMPLE_FILTER}) 'passes' {passed.sum()} strategies:")
print(f"   {n_pass_real} genuinely-good  +  {n_pass_overfit} LUCKY-OVERFIT (zero true alpha, but era-0 noise pushed them over).")
print(f"   -> the leaderboard is now polluted with lucky fakes that look strong in-sample. 'Getting a number is easy.'")
print(f"\nSTEP 2 - can each metric tell the {n_pass_real} real from the {n_pass_overfit} overfit?  (AUC; 0.5 = coin flip, 1.0 = perfect)")
print(f"   IN-SAMPLE leaderboard (rank by the backtest)        AUC = {auc_insample:.3f}   <- WEAK: selection inflates it (the curse)")
print(f"   HELD-OUT re-eval (mean of eras you didn't select on) AUC = {auc_heldout:.3f}   <- THE FIX: recovers the signal (c7/c50)")
print(f"   MULTI-REGIME SURVIVAL (MIN across held-out eras)     AUC = {auc_survival:.3f}   <- robustness metric (c30); see note")
print(f"   [note: here eras are iid so MIN ~ the mean; MIN's real value is catching alpha that DIES in some regime")
print(f"          (faded-real), which the mean hides - the c30/c61 multi-regime point, not exercised by this iid toy.]")
print(f"\nSTEP 3 - the winner's curse, concretely: the TOP 10 by backtest Sharpe")
print(f"   their in-sample Sharpe {top_insample_mean:.2f} COLLAPSES to {top_heldout_mean:.2f} held-out ({(1-top_heldout_mean/top_insample_mean)*100:.0f}% inflation from selection).")
print(f"\nTHESIS: in-sample ranking is WINNER'S-CURSED (AUC {auc_insample:.2f}, weak - selection inflates the reported score);")
print(f"   an INDEPENDENT HELD-OUT re-eval (AUC {auc_heldout:.2f}) is what recovers real-vs-overfit. Knowing it's real is the")
print(f"   hard part, and the eval that does it IS the product. (Run on your own agents: this is the quant-agent-eval skeleton.)")
print("=" * 74)
