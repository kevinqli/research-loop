"""Quant-eval prototype #1: the winner's curse IS backtest overfitting (no ground truth, low SNR).
Maps the loop's cycle-7/8 result into quant language: K candidate strategies, each with a TRUE annualized
Sharpe mu_i; an in-sample (IS) backtest gives a NOISY Sharpe SR_is = mu + N(0, s_is) where s_is ~ 1/sqrt(years)
is the finite-sample error; we SELECT argmax(SR_is) and ask how it does out-of-sample (OOS, a fresh era).
This is exactly Bailey & Lopez de Prado's "backtest overfitting" / Deflated-Sharpe selection bias.
Two regimes + the verification fix. numpy only, REPS sims, across-rep CIs. (Toy — directional, not a real backtest.)"""
import numpy as np
from math import sqrt
rng = np.random.default_rng(0)
REPS = 4000
S_IS = 0.5   # SE of an annualized Sharpe from a ~4yr backtest (~1/sqrt(4))
S_OOS = 0.5  # a separate ~4yr era
Ks = [1, 5, 10, 50, 100, 500, 1000]   # # of strategy VARIANTS tried (a quant tries hundreds)
ci = lambda x: 1.96*np.asarray(x).std(ddof=1)/sqrt(len(x))

def run(mu_draw, label):
    print(f"\n=== {label} ===")
    print(f"{'K':>5} | {'IS Sharpe(sel)':>16} | {'OOS Sharpe(sel)':>17} | {'overfit gap':>12} | {'true mu(sel)':>12} | {'survival%':>9}")
    rows=[]
    for K in Ks:
        is_sel=np.empty(REPS); oos_sel=np.empty(REPS); mu_sel=np.empty(REPS); surv=np.empty(REPS)
        for r in range(REPS):
            mu = mu_draw(K)                          # true Sharpes of the K variants
            sr_is = mu + rng.normal(0, S_IS, K)      # noisy in-sample backtest
            sr_oos = mu + rng.normal(0, S_OOS, K)    # fresh-era out-of-sample
            s = int(np.argmax(sr_is))                # pick the best backtest
            is_sel[r]=sr_is[s]; oos_sel[r]=sr_oos[s]; mu_sel[r]=mu[s]
            surv[r] = 1.0 if (sr_is[s] > 1.0 and sr_oos[s] > 0.5) else (np.nan if sr_is[s]<=1.0 else 0.0)
        gap = is_sel.mean()-oos_sel.mean()
        survival = np.nanmean(surv)*100 if not np.all(np.isnan(surv)) else float('nan')
        print(f"{K:>5} | {is_sel.mean():>6.3f} ± {ci(is_sel):.3f} | {oos_sel.mean():>7.3f} ± {ci(oos_sel):.3f}  | {gap:>10.3f}   | {mu_sel.mean():>10.3f}   | {survival:>7.0f}%")
        rows.append((K, is_sel.mean(), oos_sel.mean(), gap, mu_sel.mean(), survival))
    return rows

# Regime A: NO ALPHA AT ALL (every strategy is truly worthless, mu=0). Pure overfitting.
nullrows = run(lambda K: np.zeros(K), "REGIME A — NULL (every strategy mu=0; ANY positive IS Sharpe is pure overfitting)")
# Regime B: low SNR — 95% worthless, 5% real (mu~N(0.8,0.1)). Realistic alpha hunt.
def mix(K):
    mu=np.where(rng.random(K)<0.05, rng.normal(0.8,0.1,K), 0.0); return mu
sigrows = run(mix, "REGIME B — LOW-SNR SIGNAL (5% of strategies have real Sharpe ~0.8, 95% are noise)")

# The verification fix (cycle-8 ported): select on era-A, REPORT on a separate era-B (no double-dip)
print("\n=== VERIFICATION FIX (null regime, K=100): report on a SEPARATE era, not the selection era ===")
K=100; naive_is=np.empty(REPS); naive_oosSAME=np.empty(REPS); honest=np.empty(REPS)
for r in range(REPS):
    mu=np.zeros(K)
    eraA=mu+rng.normal(0,S_IS,K); eraB=mu+rng.normal(0,S_OOS,K); eraC=mu+rng.normal(0,S_OOS,K)
    s=int(np.argmax(eraA))
    naive_is[r]=eraA[s]                       # report the selection-era Sharpe (what charlatans show)
    naive_oosSAME[r]=max(eraA[s],eraB[s])     # select on best of A,B then report it -> still curse
    honest[r]=eraC[s]                         # select on A, REPORT on untouched era C
print(f"  naive (report selection-era Sharpe):     {naive_is.mean():+.3f} ± {ci(naive_is):.3f}   <- looks great, all overfitting")
print(f"  select-on-A, REPORT on held-out era C:   {honest.mean():+.3f} ± {ci(honest):.3f}   <- ~0, the truth (mu=0)")
print("\n--- read the rows above ---")
print(f"NULL regime: best-of-{Ks[-1]} backtests shows IS Sharpe {nullrows[-1][1]:.2f} with ZERO true alpha; OOS {nullrows[-1][2]:.2f}. That gap IS backtest overfitting, and it GROWS with K (variants tried).")
print(f"Deflated-Sharpe sanity: E[max of {Ks[-1]} N(0,{S_IS})] ~ {S_IS*np.mean(np.max(rng.normal(0,1,(2000,Ks[-1])),axis=1)):.2f} (matches the IS(sel) above).")
print("DONE")
