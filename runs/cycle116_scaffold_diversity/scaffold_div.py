"""Cycle 116 (NEW, scaffold pillar; diversify off the leaderboard thread): SCAFFOLD-SHAPE diversity. c94 found the
optimal breadth-vs-depth aspect ratio depends on the per-step error rate p (D*∝1/p). But what if p is UNKNOWN at
fleet-design time? Does a HETEROGENEOUS fleet (agents spanning different aspect ratios) beat a HOMOGENEOUS one (all
at the shape optimal for the EXPECTED p)? Extends diversify-don't-replicate (c48/51) from APPROACH-diversity to
STRUCTURAL/scaffold-shape diversity.

Model (reuses c94): each agent = a scaffold with breadth W, depth D=B/W; its 'reach' = best branch's true quality =
max over W of (approach a~N(0,1) + δ·useful refinement), useful=min(geom(p)-1, D) under the ACTUAL per-step error p.
Fleet of K agents; fleet outcome = max reach over the K agents. HOMOGENEOUS: all K at W*(E[p]); HETEROGENEOUS: K agents
log-spread across W. Compare under LOW p-uncertainty (p fixed) vs HIGH (p ~ log-uniform wide). numpy-only, REPS reps."""
import numpy as np, json, os
RNG = np.random.default_rng(0)
REPS, B, K, DELTA, SIG_A = 4000, 240, 12, 0.1, 1.0
WS_HET = [2,3,4,6,8,12,16,24,40,60,80,120]    # 12 agents, log-spread (divisors of 240)
W_HOM = 5                                      # ~optimal for p0=0.05 (D=48), per c94
os.makedirs("runs/cycle116_scaffold_diversity", exist_ok=True)

def agent_reach(W, p, reps):                   # p: scalar or (reps,)
    D = B//W
    pa = np.full((reps,W), p) if np.isscalar(p) else np.repeat(np.asarray(p)[:,None], W, axis=1)
    a = RNG.normal(0, SIG_A, (reps,W))
    useful = np.clip(RNG.geometric(pa)-1, 0, D)
    return (a + DELTA*useful).max(axis=1)

def fleet_outcome(hetero, p):
    reaches = [agent_reach(WS_HET[k] if hetero else W_HOM, p, REPS) for k in range(K)]
    return float(np.max(np.stack(reaches, axis=1), axis=1).mean())

out = {"model":{"REPS":REPS,"B":B,"K":K,"W_HOM":W_HOM}, "results":{}}
print(f"B={B}/agent, K={K} agents, δ={DELTA}. HOMOGENEOUS=all W={W_HOM} (opt for p0=0.05); HETEROGENEOUS=spread {WS_HET}.")
print(f"Fleet outcome = max reach over K agents (higher=better).\n")
print(f"{'p-uncertainty':>16} | {'HOMOGENEOUS':>12} | {'HETEROGENEOUS':>14} | {'hetero − homo':>13}")
for label, p in [("LOW (p=0.05)", 0.05),
                 ("MED (p~U[.03,.10])", np.exp(RNG.uniform(np.log(0.03), np.log(0.10), REPS))),
                 ("HIGH (p~U[.01,.30])", np.exp(RNG.uniform(np.log(0.01), np.log(0.30), REPS)))]:
    hom = fleet_outcome(False, p); het = fleet_outcome(True, p)
    out["results"][label] = {"homo":round(hom,4), "hetero":round(het,4), "diff":round(het-hom,4)}
    print(f"{label:>16} | {hom:>12.4f} | {het:>14.4f} | {het-hom:>+13.4f}")

json.dump(out, open("runs/cycle116_scaffold_diversity/score.json","w"), indent=1)
print("\n--- READ THE RAW ROWS (don't trust this summary) ---")
r = out["results"]
print(f"P2 LOW uncertainty: homo {r['LOW (p=0.05)']['homo']} vs hetero {r['LOW (p=0.05)']['hetero']} (homo wins/ties?)")
print(f"P1 HIGH uncertainty: hetero {r['HIGH (p~U[.01,.30])']['hetero']} vs homo {r['HIGH (p~U[.01,.30])']['homo']} (hetero wins?)")
print(f"P3 hetero−homo grows with uncertainty: LOW {r['LOW (p=0.05)']['diff']} -> MED {r['MED (p~U[.03,.10])']['diff']} -> HIGH {r['HIGH (p~U[.01,.30])']['diff']}")
print("DONE")
