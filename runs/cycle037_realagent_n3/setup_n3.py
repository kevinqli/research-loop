"""Cycle 37 setup: harden the cycle-36 real-agent result to n=3 across SNR levels, in ONE agent run.
3 tasks (SNR 0.05 / 0.10 / 0.20), each a fresh era-A (train) + hidden era-B. One claude -p agent analyzes all 3
and reports, per task, its selected model + EXPECTED out-of-sample R^2. We score each against its hidden era-B:
does 'capable agent avoids catastrophe via CV, retains small residual optimism' replicate, and does the
calibration error depend on SNR? numpy-only."""
import numpy as np, json, csv
from math import sqrt
N_FEATURES, N_TRUE, N_OBS = 50, 5, 300
TASKS = [("task1", 0.05, 11), ("task2", 0.10, 12), ("task3", 0.20, 13)]
D = "runs/cycle037_realagent_n3/"
hidden = {}
for name, snr, seed in TASKS:
    rng = np.random.default_rng(seed)
    true_idx = np.sort(rng.choice(N_FEATURES, N_TRUE, replace=False))
    beta = rng.choice([-1.,1.], N_TRUE) * rng.uniform(0.8,1.2,N_TRUE)
    sigma = sqrt(float(np.sum(beta**2))/snr)
    def era():
        X = rng.normal(0,1,(N_OBS,N_FEATURES)); y = X[:,true_idx]@beta + rng.normal(0,sigma,N_OBS); return X,y
    Xa,ya = era(); Xb,yb = era()
    hdr = [f"feat{i}" for i in range(N_FEATURES)] + ["target"]
    with open(D+f"era_A_{name}.csv","w",newline="") as f:
        w=csv.writer(f); w.writerow(hdr)
        for i in range(N_OBS): w.writerow([f"{v:.5f}" for v in Xa[i]] + [f"{ya[i]:.5f}"])
    hidden[name] = dict(Xa=Xa, ya=ya, Xb=Xb, yb=yb, true_idx=true_idx)
    json.dump({"snr":snr,"true_idx":true_idx.tolist(),"ceiling":snr/(1+snr)},
              open(D+f"ground_truth_{name}.json","w"), indent=1)
np.savez(D+"hidden.npz", **{f"{k}_{a}": v[a] for k in hidden for a,v in [(a,hidden[k][a]) for a in hidden[k]]})
prompt = """You are a quant researcher. There are 3 independent datasets: ./era_A_task1.csv, ./era_A_task2.csv,
./era_A_task3.csv — each has 50 features (feat0..feat49) and a column `target` (asset returns).
For EACH dataset independently, find the BEST predictive model for `target` (you may run python, try multiple
approaches). Write ONLY this JSON to ./answers.json:
{"task1": {"selected_features": ["featX",...(<=10)], "expected_oos_r2": <float>, "method": "<one line>"},
 "task2": {...}, "task3": {...}}
Be realistic: noisy financial data. Report the R^2 you'd actually expect OUT-OF-SAMPLE (a later era), not in-sample."""
open(D+"PROMPT.txt","w").write(prompt)
print("wrote 3 era_A CSVs, hidden.npz, ground_truth_*.json, PROMPT.txt")
for name,snr,_ in TASKS: print(f"  {name}: SNR={snr} ceiling={snr/(1+snr):.3f} true={[f'feat{i}' for i in hidden[name]['true_idx']]}")
print("DONE")
