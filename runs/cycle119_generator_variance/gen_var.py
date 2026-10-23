"""Cycle 119 (generation side, follow-up to c118; composes c57+c48+c7): generator QUALITY (mean) vs DIVERSITY
(variance) as a function of FLEET SIZE K. Best-of-K keeps only the tail, so a high-VARIANCE ('wild') generator can
beat a high-MEAN reliable one once K is large enough (the best-of-many hits the wild tail; c57's E[max_K]). Also
measure each generator's winner's-curse GAP (how hard its candidates are to RANK under eval noise) to see which one
NEEDS verification more.

RELIABLE: true q ~ N(0.8, 0.3) (high mean, low spread). WILD: ~ N(0.2, 0.6) (lower mean, fat tail). observed = q +
N(0, 0.4). For each K: oracle = max true q (the ceiling); no_verify outcome = true q of argmax(observed); gap =
observed − true of the no-verify pick (the winner's-curse delusion). numpy-only, REPS reps."""
import numpy as np, json, os
RNG = np.random.default_rng(0)
REPS, SIG_OBS = 30000, 0.4
GENS = {"reliable":(0.8,0.3), "wild":(0.2,0.6)}
os.makedirs("runs/cycle119_generator_variance", exist_ok=True)

def measure(mu, sg, K):
    q = RNG.normal(mu, sg, (REPS,K)); obs = q + RNG.normal(0,SIG_OBS,(REPS,K)); r=np.arange(REPS)
    oracle = q.max(1); sel = obs.argmax(1)
    return float(oracle.mean()), float(q[r,sel].mean()), float((obs[r,sel]-q[r,sel]).mean())

out = {"model":{"REPS":REPS,"SIG_OBS":SIG_OBS,"GENS":GENS}, "K":{}}
Ks = [1,2,5,10,20,50,100]
print(f"RELIABLE N(0.8,0.3) vs WILD N(0.2,0.6); eval-noise σ={SIG_OBS}. ORACLE=best-of-K ceiling.\n")
print(f"{'K':>5} | {'REL oracle':>10} {'WILD oracle':>11} {'winner':>7} | {'REL gap':>8} {'WILD gap':>9}")
for K in Ks:
    ro,rnv,rg = measure(*GENS["reliable"], K); wo,wnv,wg = measure(*GENS["wild"], K)
    out["K"][K] = {"rel_oracle":round(ro,4),"wild_oracle":round(wo,4),"rel_gap":round(rg,4),"wild_gap":round(wg,4),
                   "rel_nv":round(rnv,4),"wild_nv":round(wnv,4)}
    win = "WILD" if wo>ro else "RELIABLE"
    print(f"{K:>5} | {ro:>10.4f} {wo:>11.4f} {win:>7} | {rg:>8.4f} {wg:>9.4f}")

json.dump(out, open("runs/cycle119_generator_variance/score.json","w"), indent=1)
print("\n--- READ THE RAW ROWS (don't trust this summary) ---")
k=out["K"]
flip = next((K for K in Ks if k[K]["wild_oracle"]>k[K]["rel_oracle"]), None)
print(f"P1 FLIP: reliable wins small K, wild wins large K. flip at K≈{flip} (rel oracle {k[Ks[0]]['rel_oracle']}→{k[Ks[-1]]['rel_oracle']}, wild {k[Ks[0]]['wild_oracle']}→{k[Ks[-1]]['wild_oracle']})")
print(f"P2 winner's-curse GAP (K=50): reliable {k[50]['rel_gap']} vs wild {k[50]['wild_gap']} (which generator's candidates are HARDER to rank → needs verification more?)")
print("=> generator variance ∝ fleet size: large fleet → WILD (swing for the fences, best-of-many hits the tail); small fleet → RELIABLE. And the harder-to-rank generator needs verification more.")
print("DONE")
