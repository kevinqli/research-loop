"""Cycle 39 setup: is an agent's self-reported CONFIDENCE a usable SELECTION signal, or do you need an external
held-out evaluator? Tests the cycle-37 compression hypothesis decisively. 6 tasks spanning a WIDE difficulty range
(SNR 0.02..0.5 -> true OOS ceiling 0.02..0.33). One claude -p agent reports, per task, its best model + expected
OOS R^2 (its confidence). We score true OOS on hidden era-B and ask: does confidence RANK true performance
(Spearman), and does its range track the true range, or is it COMPRESSED? numpy-only."""
import numpy as np, json, csv
from math import sqrt
N_FEATURES, N_TRUE, N_OBS = 50, 5, 300
SNRS = [0.02, 0.05, 0.10, 0.20, 0.35, 0.50]   # wide difficulty spread
D = "runs/cycle039_confidence_signal/"
for i, snr in enumerate(SNRS, 1):
    rng = np.random.default_rng(100+i)
    true_idx = np.sort(rng.choice(N_FEATURES, N_TRUE, replace=False))
    beta = rng.choice([-1.,1.], N_TRUE)*rng.uniform(0.8,1.2,N_TRUE); sigma = sqrt(float(np.sum(beta**2))/snr)
    def era():
        X=rng.normal(0,1,(N_OBS,N_FEATURES)); y=X[:,true_idx]@beta+rng.normal(0,sigma,N_OBS); return X,y
    Xa,ya=era(); Xb,yb=era()
    hdr=[f"feat{j}" for j in range(N_FEATURES)]+["target"]
    with open(D+f"era_A_task{i}.csv","w",newline="") as f:
        w=csv.writer(f); w.writerow(hdr)
        for k in range(N_OBS): w.writerow([f"{v:.5f}" for v in Xa[k]]+[f"{ya[k]:.5f}"])
    json.dump({"snr":snr,"true_idx":true_idx.tolist(),"ceiling":snr/(1+snr),
               "Xb":Xb.tolist(),"yb":yb.tolist(),"Xa":Xa.tolist(),"ya":ya.tolist()},
              open(D+f"hidden_task{i}.json","w"))
prompt = """You are a quant researcher. There are 6 independent datasets ./era_A_task1.csv ... ./era_A_task6.csv,
each with 50 features (feat0..feat49) and a column `target`. The datasets vary in how much real signal they contain
(some are nearly pure noise, some have strong signal) — your job is to tell them apart by how well you can predict.
For EACH dataset, find the best model and write ONLY this JSON to ./answers.json:
{"task1": {"selected_features": ["featX",...(<=10)], "expected_oos_r2": <float>}, ... "task6": {...}}
Report the R^2 you'd actually expect OUT-OF-SAMPLE (a later era). Use the full plausible range — if a dataset is
almost pure noise say ~0; if it has strong signal say so. You may run python."""
open(D+"PROMPT.txt","w").write(prompt)
print(f"wrote 6 era_A CSVs (SNR {SNRS}) + hidden_task*.json + PROMPT.txt; ceilings {[round(s/(1+s),3) for s in SNRS]}")
print("DONE")
