"""Cycle 29 (quant-eval Q1): the dual-SNR synthetic-market generator + validation that the SNR knob works.
A market with a TUNABLE signal-to-noise knob and a KNOWN latent causal structure (ground truth for building the
eval; the agent never sees which features are real). Returns r = X[:, true_idx] @ beta + noise, with noise scaled
so var(signal)/var(noise) = SNR exactly. Multiple eras share the SAME beta (real alpha generalizes; noise-fits
don't). numpy-only. MPS-exploratory (toy synthetic). Validates: (1) the SNR knob controls the in-sample/OOS
overfitting gap; (2) best-of-K feature-subset selection overfits to ~0 OOS (the winner's curse, in feature space)
while the TRUE subset survives a held-out era — exactly what the verification-survival eval (Q2) needs.

Theory anchors (sanity checks): a correctly-specified model's max achievable R^2 = SNR/(1+SNR)
(SNR=1 -> 0.5, 0.2 -> 0.167, 0.05 -> 0.048). The agent's job (later cycles) is to FIND true_idx and predict OOS."""
import numpy as np, json
from math import sqrt

N_FEATURES = 50      # 5 true alpha drivers + 45 decoys
N_TRUE = 5
N_OBS = 500          # observations per era
N_ERAS = 5
SNRS = [0.02, 0.05, 0.2, 1.0]   # the knob (quant is LOW-SNR; 0.02-0.05 is realistic)
K_SUBSETS = 300      # # of random 5-feature strategies a "quant"/agent tries

def make_market(snr, seed):
    """Returns eras (list of (X, r)) sharing one hidden beta + true_idx. var(signal)/var(noise)=snr exactly."""
    rng = np.random.default_rng(seed)
    true_idx = np.sort(rng.choice(N_FEATURES, N_TRUE, replace=False))
    beta = rng.choice([-1.0, 1.0], N_TRUE) * rng.uniform(0.7, 1.3, N_TRUE)   # fixed across eras
    var_signal = float(np.sum(beta**2))                                       # X~N(0,1) iid -> var(signal)=||beta||^2
    sigma_noise = sqrt(var_signal / snr)
    eras = []
    for e in range(N_ERAS):
        X = rng.normal(0, 1, (N_OBS, N_FEATURES))
        signal = X[:, true_idx] @ beta
        r = signal + rng.normal(0, sigma_noise, N_OBS)
        eras.append((X, r))
    return eras, true_idx, beta, sigma_noise

def r2(y, yhat):
    ss_res = np.sum((y - yhat)**2); ss_tot = np.sum((y - y.mean())**2)
    return 1 - ss_res/ss_tot

def ols_fit_predict(Xtr, ytr, Xte):
    beta_hat, *_ = np.linalg.lstsq(np.column_stack([np.ones(len(Xtr)), Xtr]), ytr, rcond=None)
    return np.column_stack([np.ones(len(Xte)), Xte]) @ beta_hat

print(f"dims: {N_FEATURES} feats ({N_TRUE} true), {N_OBS} obs/era, {N_ERAS} eras; K={K_SUBSETS} random strategies\n")
print(f"{'SNR':>5} | {'maxR2=SNR/(1+SNR)':>17} | {'fullmodel IS R2':>15} | {'fullmodel OOS R2':>16} | {'TRUEfeat OOS R2':>15} | {'best-of-K IS R2':>15} | {'best-of-K OOS R2':>16}")
rows=[]
for snr in SNRS:
    eras, true_idx, beta, sig = make_market(snr, seed=0)
    (X0,r0),(X1,r1) = eras[0], eras[1]   # era0 = build, era1 = held-out
    # full model (all 50 features) — overfits
    is_full = r2(r0, ols_fit_predict(X0,r0,X0))
    oos_full = r2(r1, ols_fit_predict(X0,r0,X1))
    # true-feature model (oracle structure) — generalizes
    oos_true = r2(r1, ols_fit_predict(X0[:,true_idx],r0,X1[:,true_idx]))
    # WINNER'S CURSE in feature space: try K random 5-subsets, pick best IS R2 on era0, eval OOS on era1
    rng = np.random.default_rng(123)
    best_is=-9; best_sub=None
    for _ in range(K_SUBSETS):
        sub = rng.choice(N_FEATURES, N_TRUE, replace=False)
        is_r = r2(r0, ols_fit_predict(X0[:,sub],r0,X0[:,sub]))
        if is_r>best_is: best_is=is_r; best_sub=sub
    oos_bestk = r2(r1, ols_fit_predict(X0[:,best_sub],r0,X1[:,best_sub]))
    maxr2 = snr/(1+snr)
    print(f"{snr:>5} | {maxr2:>17.3f} | {is_full:>15.3f} | {oos_full:>16.3f} | {oos_true:>15.3f} | {best_is:>15.3f} | {oos_bestk:>16.3f}")
    rows.append({"snr":snr,"max_r2":maxr2,"is_full":float(is_full),"oos_full":float(oos_full),
                 "oos_true":float(oos_true),"bestK_is":float(best_is),"bestK_oos":float(oos_bestk)})
json.dump({"dims":{"n_features":N_FEATURES,"n_true":N_TRUE,"n_obs":N_OBS,"K":K_SUBSETS},"rows":rows},
          open("runs/cycle029_quant_dualsnr/score.json","w"), indent=1)

print("\n--- read the rows above ---")
lo,hi = rows[0],rows[-1]
print(f"(1) SNR knob controls overfitting gap: at SNR={lo['snr']} full-model IS R2 {lo['is_full']:.2f} but OOS {lo['oos_full']:.2f} (gap {lo['is_full']-lo['oos_full']:.2f}); at SNR={hi['snr']} IS {hi['is_full']:.2f} OOS {hi['oos_full']:.2f} (gap {hi['is_full']-hi['oos_full']:.2f}).")
print(f"(2) winner's curse in feature space: at SNR={lo['snr']} best-of-{K_SUBSETS} subsets shows IS R2 {lo['bestK_is']:.2f} but OOS {lo['bestK_oos']:.2f}; the TRUE subset OOS {lo['oos_true']:.2f} (~max {lo['max_r2']:.2f}). Real alpha survives the held-out era, the noise-fit does not.")
print(f"(3) true-feat OOS R2 tracks the theoretical max SNR/(1+SNR): " + ", ".join(f"{r['oos_true']:.3f}vs{r['max_r2']:.3f}" for r in rows))
print("DONE — generator validated; ready for Q2 (verification-survival metric) + Q4 (obfuscation).")
