"""Cycle 154 (#10a — does c94's D*~1/p survive HETEROGENEOUS branch error-rates? scaffold pillar, HARDEN). c94: a
scaffold's optimal aspect ratio (breadth W vs depth D) is governed by the per-step error rate p — go DEEP when
refinement is reliable (D*~1/p), WIDE when steps corrupt fast. c94 assumed a SINGLE p (all branches equally risky).
#10a: real fleets have HETEROGENEOUS branches — some approaches are riskier (high p_i) than others. With a FIXED total
depth budget B = sum d_i across W branches, should you allocate depth ADAPTIVELY (deeper on low-p reliable branches)
or UNIFORMLY (equal depth)? And does the c94 rule become a PER-BRANCH rule d_i* ~ 1/p_i?

MODEL (faithful to c94's refinement-vs-propagation): a branch refined to depth d under per-step error p has expected
quality Q(d,p) = d * (1-p)^d  [refinement gain d * the survival-without-corruption (1-p)^d = c94's error propagation].
=> each branch has an interior optimal depth d*(p) = argmax_d d*(1-p)^d = -1/ln(1-p) ~ 1/p (small p) = c94's D*~1/p.
The fleet refines W branches at depths {d_i} (sum = B), observes Q_i + N(0,sigma_obs) (selection noise), keeps the
argmax, realizes its TRUE Q. Outcome = E[realized true Q].

ARMS: UNIFORM (d_i = B/W) | ADAPTIVE (d_i ~ 1/p_i, normalized to B) | ORACLE (greedy water-fill on true E[best]).

PRE-REG:
  P1 (control): HOMOGENEOUS p (all equal) -> ADAPTIVE == UNIFORM (no heterogeneity to exploit).
  P2: HETEROGENEOUS p -> ADAPTIVE (deeper on reliable branches) BEATS UNIFORM; the gain GROWS with the spread of p
      (uniform over-refines risky branches where depth corrupts, under-refines reliable ones).
  P3: the per-branch optimal d_i* ~ 1/p_i (c94's rule per-branch) — so allocating ~1/p_i approaches the oracle.
"""
import numpy as np, json, os
os.makedirs("runs/cycle154_heterogeneous_depth", exist_ok=True)
rng = np.random.default_rng(0)

W = 8                     # branches
B = 80.0                  # total depth budget (sum of d_i)
SIGMA_OBS = 0.5           # selection noise
REPS = 20000

def quality(d, p):
    return d * (1.0 - p) ** d          # refinement d * survival (1-p)^d  (c94 propagation)

def d_star(p):
    return -1.0 / np.log(1.0 - p)       # argmax_d d*(1-p)^d  (~1/p for small p)

def oracle_alloc(ps, B):
    # greedy water-fill: add budget in small steps to the branch that most raises the BEST true Q
    d = np.zeros(len(ps)); step = 0.5
    n = int(round(B / step))
    for _ in range(n):
        cur_best = np.max(quality(d, ps)) if d.sum() > 0 else 0.0
        gains = [np.max(quality(d + np.eye(len(ps))[i]*step, ps)) - cur_best for i in range(len(ps))]
        d[int(np.argmax(gains))] += step
    return d

def fleet_outcome(ps, d, reps):
    Qtrue = quality(d, ps)                                  # (W,) true quality per branch
    obs = Qtrue[None, :] + rng.normal(0, SIGMA_OBS, size=(reps, len(ps)))
    sel = np.argmax(obs, axis=1)
    realized = Qtrue[sel]
    return float(realized.mean()), float(1.96 * realized.std(ddof=1) / np.sqrt(reps))

# heterogeneity sweep: p_i spread around mean 0.1 with increasing spread (log-uniform width)
configs = {
  "homogeneous (all p=0.10)":          np.full(W, 0.10),
  "mild (p in [0.06,0.16])":           None,  # filled below (seeded, fixed)
  "moderate (p in [0.03,0.30])":       None,
  "strong (p in [0.015,0.50])":        None,
}
# fixed deterministic heterogeneous p-vectors (log-spaced so the geomean ~0.10)
configs["mild (p in [0.06,0.16])"]     = np.geomspace(0.06, 0.16, W)
configs["moderate (p in [0.03,0.30])"] = np.geomspace(0.03, 0.30, W)
configs["strong (p in [0.015,0.50])"]  = np.geomspace(0.015, 0.50, W)

out = {"W": W, "B": B, "SIGMA_OBS": SIGMA_OBS, "REPS": REPS, "configs": {}}
print(f"=== #10a: heterogeneous-p depth allocation (W={W}, B={B}, sigma_obs={SIGMA_OBS}, REPS={REPS}) ===")
print(f"per-branch d*(p)~1/p check: p=0.10 -> d*={d_star(0.10):.2f} (1/p=10.0); p=0.30 -> d*={d_star(0.30):.2f} (1/p=3.33); p=0.015 -> d*={d_star(0.015):.1f} (1/p=66.7)")
print(f"{'config':>32} {'uniform':>16} {'adaptive(1/p)':>18} {'oracle':>16} {'adapt-unif':>12}")
for name, ps in configs.items():
    ps = np.array(ps, dtype=float)
    d_uniform = np.full(W, B / W)
    inv = 1.0 / ps; d_adaptive = inv / inv.sum() * B
    d_oracle = oracle_alloc(ps, B)
    u_m, u_c = fleet_outcome(ps, d_uniform, REPS)
    a_m, a_c = fleet_outcome(ps, d_adaptive, REPS)
    o_m, o_c = fleet_outcome(ps, d_oracle, REPS)
    out["configs"][name] = {"p": ps.tolist(), "uniform": [u_m, u_c], "adaptive": [a_m, a_c], "oracle": [o_m, o_c],
                            "adaptive_minus_uniform": a_m - u_m, "oracle_d": d_oracle.tolist(), "adaptive_d": d_adaptive.tolist()}
    print(f"{name:>32} {u_m:>8.3f}+-{u_c:<6.3f} {a_m:>9.3f}+-{a_c:<6.3f} {o_m:>8.3f}+-{o_c:<6.3f} {a_m-u_m:>+12.3f}")
json.dump(out, open("runs/cycle154_heterogeneous_depth/score.json", "w"), indent=1)
print("PRE-REG: P1 homogeneous -> adaptive==uniform; P2 heterogeneous -> adaptive>uniform, gain grows with spread; P3 d*~1/p per-branch.")
print("DONE")
