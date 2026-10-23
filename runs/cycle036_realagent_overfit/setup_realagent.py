"""Cycle 36 setup: REAL-agent test of backtest overfitting (closes the sim-only gap).
Generate era-A (train) + era-B (held-out, hidden) from the cycle-29 dual-SNR generator at a realistic low SNR.
The agent sees ONLY era_A.csv and is asked to find the best predictive model + report its EXPECTED out-of-sample R^2.
We then fit OLS on its selected features (era-A) and evaluate on era-B, comparing its self-report to the truth.
Tests: (1) does a real agent OVERFIT (pick noise features)? (2) is its expected-OOS self-report CALIBRATED or
inflated (the winner's curse with a real agent)? numpy-only. MPS-exploratory (toy)."""
import numpy as np, json, csv
from math import sqrt
N_FEATURES, N_TRUE, N_OBS, SNR, SEED = 50, 5, 300, 0.10, 7
D = "runs/cycle036_realagent_overfit/"
rng = np.random.default_rng(SEED)
true_idx = np.sort(rng.choice(N_FEATURES, N_TRUE, replace=False))
beta = rng.choice([-1.,1.], N_TRUE) * rng.uniform(0.8,1.2,N_TRUE)
sigma = sqrt(float(np.sum(beta**2))/SNR)
def era():
    X = rng.normal(0,1,(N_OBS,N_FEATURES)); y = X[:,true_idx]@beta + rng.normal(0,sigma,N_OBS); return X,y
Xa,ya = era(); Xb,yb = era()
# write era_A.csv for the agent (feat0..feat49 + target); era_B stays hidden
hdr = [f"feat{i}" for i in range(N_FEATURES)] + ["target"]
with open(D+"era_A.csv","w",newline="") as f:
    w=csv.writer(f); w.writerow(hdr)
    for i in range(N_OBS): w.writerow([f"{v:.5f}" for v in Xa[i]] + [f"{ya[i]:.5f}"])
# stash era_B + ground truth for scoring (NOT given to the agent)
np.savez(D+"hidden.npz", Xa=Xa, ya=ya, Xb=Xb, yb=yb, true_idx=true_idx, beta=beta)
json.dump({"true_idx": true_idx.tolist(), "snr": SNR, "max_r2_ceiling": SNR/(1+SNR),
           "n_features": N_FEATURES, "n_true": N_TRUE, "n_obs": N_OBS},
          open(D+"ground_truth.json","w"), indent=1)
prompt = """You are a quant researcher. The file ./era_A.csv has 50 features (feat0..feat49) and a column `target`
(asset returns). Find the BEST predictive model for `target`. You may run python and try multiple approaches.
Then write ONLY this JSON to ./answer.json:
{"selected_features": ["featX", ...],  // the <=10 features your best model uses
 "expected_oos_r2": <float>,           // your honest estimate of this model's R^2 on FUTURE UNSEEN data (a later era)
 "method": "<one line: how you chose, and how you estimated expected_oos_r2>"}
Be realistic: this is noisy financial data. Report the R^2 you'd actually expect out-of-sample, not in-sample."""
open(D+"PROMPT.txt","w").write(prompt)
print(f"wrote era_A.csv ({N_OBS}x{N_FEATURES}), hidden.npz, ground_truth.json, PROMPT.txt")
print(f"true features: {[f'feat{i}' for i in true_idx]}; SNR={SNR} -> OOS R^2 ceiling = {SNR/(1+SNR):.3f}")
print("DONE")
