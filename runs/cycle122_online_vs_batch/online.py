"""Cycle 122 (NEW; ties c7 + the secretary problem + c102/c117 commitment): ONLINE/irrevocable selection vs BATCH
under eval-noise. The verification fix for the winner's curse is RE-EVALUATION (c7/c8) — but you can only re-eval
candidates you KEPT. Irrevocable SEQUENTIAL selection (secretary-style: candidates arrive 1-by-1, accept/reject NOW)
forecloses re-eval → it should be far more FRAGILE to eval-noise than BATCH (keep all, re-eval the finalists).

Model: K candidates, true q~N(0,1); observed = q + N(0,σ). BATCH: observe all, re-eval top-3 on an independent
held-out, pick best (c8 recipe). SEQUENTIAL: random arrival order; observe the first round(K/e) as a learning phase,
then accept the FIRST whose observed value exceeds the learning-phase max (classic 1/e secretary rule); irrevocable.
ORACLE = max true q. Metric: true q of the selected. numpy-only, REPS reps."""
import numpy as np, json, os
RNG = np.random.default_rng(0)
REPS, K, M = 20000, 50, 3
LEARN = int(round(K/np.e))                      # 1/e learning phase ≈ 18
os.makedirs("runs/cycle122_online_vs_batch", exist_ok=True)

def trial(sig):
    q = RNG.normal(0,1,(REPS,K)); obs = q + RNG.normal(0,sig,(REPS,K)); r = np.arange(REPS)
    oracle = q.max(1)
    # BATCH: re-eval top-M on an independent held-out, pick best held-out
    topm = np.argsort(-obs,1)[:,:M]
    qh = np.take_along_axis(q,topm,1) + RNG.normal(0,sig,(REPS,M))
    bsel = np.take_along_axis(topm, qh.argmax(1)[:,None],1)[:,0]; batch = q[r,bsel]
    # SEQUENTIAL secretary (irrevocable): accept first obs > max(obs[:LEARN]); else last
    thr = obs[:,:LEARN].max(1)
    out = q[:,-1].copy()                         # default: forced to take the last
    taken = np.zeros(REPS, bool)
    for j in range(LEARN, K):
        pick = (~taken) & (obs[:,j] > thr)
        out = np.where(pick, q[:,j], out); taken = taken | pick
    return round(float(oracle.mean()),4), round(float(batch.mean()),4), round(float(out.mean()),4)

out = {"model":{"REPS":REPS,"K":K,"M":M,"LEARN":LEARN}, "results":{}}
print(f"K={K} candidates, batch re-eval top-{M}, secretary learning-phase={LEARN} (≈K/e). REPS={REPS}.")
print(f"true q of SELECTED (higher=better); ORACLE=max true q.\n")
print(f"{'σ (eval-noise)':>14} | {'ORACLE':>7} | {'BATCH(re-eval)':>14} | {'SEQUENTIAL(irrev.)':>18} | {'batch − seq':>11}")
for sig in [0.1, 0.5, 1.0, 2.0]:
    orc, b, s = trial(sig); out["results"][sig] = {"oracle":orc,"batch":b,"seq":s,"gap":round(b-s,4)}
    print(f"{sig:>14} | {orc:>7} | {b:>14} | {s:>18} | {b-s:>+11.4f}")

json.dump(out, open("runs/cycle122_online_vs_batch/score.json","w"), indent=1)
print("\n--- READ THE RAW ROWS (don't trust this summary) ---")
r = out["results"]
print(f"P1 batch−seq gap vs σ: " + " ".join(f"σ{s}:{r[s]['gap']:+.3f}" for s in [0.1,0.5,1.0,2.0]) + " (grows with noise → sequential fragile?)")
print(f"low-σ: batch {r[0.1]['batch']} ≈ seq {r[0.1]['seq']}? high-σ: batch {r[2.0]['batch']} vs seq {r[2.0]['seq']}")
print("=> irrevocable SEQUENTIAL commitment is FRAGILE to eval-noise (can't re-eval what it rejected); BATCH (keep + re-eval) is robust. Under noisy judgment, don't irrevocably commit — keep options open + re-eval (the c7 fix needs kept candidates).")
print("DONE")
