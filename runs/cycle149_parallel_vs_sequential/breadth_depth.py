"""Cycle 149 (#10-adjacent, SCAFFOLD pillar — diversified off benchmark-design per the c148 note). The parallel-agent
lab question: at FIXED total compute C, is it better to run N SHORT parallel agents and SELECT the best
(BREADTH — buys coverage of the ceiling distribution, but pays the winner's-curse SELECTION tax, c7/c90), or 1 LONG
agent (DEPTH — no selection tax, but bounded by ONE attempt's ceiling, c118)? UNIFIES c118 (generator/coverage) x c7
(selection curse) x c11-12 (depth/refinement) into a compute-allocation rule.

MODEL (fixed compute C; refinement timescale tau):
  - An attempt has a TRUE ceiling q ~ N(mu, sigma_ceil) (different approaches reach different best-case quality, c118).
  - After t steps it realizes q * (1 - exp(-t/tau)) of its ceiling (diminishing-returns refinement, c11-12).
  - BREADTH N: N attempts, each gets t=C/N steps -> realizes q_i*(1-exp(-(C/N)/tau)); OBSERVED score = realized +
    N(0, sigma_v) (noisy verifier); SELECT argmax observed; realized_true at the selected index (the c7 selection,
    so a noisy verifier => winner's-curse tax: you may pick a lucky-looking under-performer).
  - DEPTH = N=1 (one attempt, C steps -> fully refined toward its single ceiling).
  - N* = argmax_N E[realized_true] = the optimal degree of parallelism.

PRE-REG (fix before peeking):
  P1 (control): sigma_ceil=0 (identical ceilings) -> N*=1 (DEPTH). No coverage benefit to sampling many, and breadth
     just under-refines each -> pure depth wins.
  P2: sigma_ceil HIGH + sigma_v LOW (good verifier) -> N*>1 (PARALLELISM pays): big upside to finding a high-ceiling
     attempt AND you can actually select it.
  P3: the SELECTION TAX caps parallelism: raising sigma_v (noisier verifier) LOWERS N* (toward depth) -> there is an
     INTERIOR optimum; over-parallelizing (huge N, each barely refined, selected by noise) is bad (c7 winner's curse).
  P4 (the c118 tie): breadth's advantage over depth = coverage gain E[max_N ceilings]-E[ceiling] (~sigma_ceil*sqrt(2lnN)),
     REALIZED only if the verifier can select it; a noisy verifier converts coverage into the winner's curse.
"""
import numpy as np, json, os
os.makedirs("runs/cycle149_parallel_vs_sequential", exist_ok=True)
rng = np.random.default_rng(0)

MU = 1.0; C = 100.0; TAU = 10.0        # compute budget, refinement timescale (depth N=1 -> 1-e^-10 ~ fully refined)
N_VALUES = [1, 2, 4, 5, 10, 20, 50, 100]
SIGMA_CEIL_VALUES = [0.0, 0.1, 0.3, 0.6]
SIGMA_V_VALUES = [0.0, 0.2, 0.5]
REPS = 20000

def realized_for_N(N, sigma_ceil, sigma_v):
    t = C / N
    frac = 1.0 - np.exp(-t / TAU)                       # fraction of ceiling realized at this depth
    q = rng.normal(MU, sigma_ceil, size=(REPS, N))      # each attempt's true ceiling
    realized_true = q * frac                            # true realized quality per attempt
    observed = realized_true + rng.normal(0, sigma_v, size=(REPS, N))  # noisy verifier score
    sel = np.argmax(observed, axis=1)                   # select best-by-verifier (c7)
    picked = realized_true[np.arange(REPS), sel]        # the TRUE quality of the selected attempt
    return picked.mean(), picked.std(ddof=1) / np.sqrt(REPS), frac

results = {"MU": MU, "C": C, "TAU": TAU, "N_values": N_VALUES, "sigma_ceil_values": SIGMA_CEIL_VALUES,
           "sigma_v_values": SIGMA_V_VALUES, "REPS": REPS, "grid": {}}
for sc in SIGMA_CEIL_VALUES:
    for sv in SIGMA_V_VALUES:
        curve = {}
        for N in N_VALUES:
            m, se, frac = realized_for_N(N, sc, sv)
            curve[N] = {"mean": float(m), "se": float(se), "frac_ceiling": float(frac)}
        best_N = max(N_VALUES, key=lambda n: curve[n]["mean"])
        depth_val = curve[1]["mean"]
        results["grid"][f"sc{sc}_sv{sv}"] = {"sigma_ceil": sc, "sigma_v": sv, "curve": curve,
            "N_star": best_N, "best_mean": curve[best_N]["mean"], "depth_mean": depth_val,
            "breadth_advantage": curve[best_N]["mean"] - depth_val}
json.dump(results, open("runs/cycle149_parallel_vs_sequential/score.json", "w"), indent=1)

print("=== #10: BREADTH (N parallel, select best) vs DEPTH (1 long agent) at fixed compute C ===")
print(f"MU={MU} C={C} TAU={TAU} REPS={REPS}; depth(N=1) frac_ceiling={1-np.exp(-C/TAU):.3f}")
print(f"{'sigma_ceil':>10} {'sigma_v':>8} {'N*':>5} {'best_mean':>11} {'depth(N=1)':>11} {'breadth_adv':>12}")
for sc in SIGMA_CEIL_VALUES:
    for sv in SIGMA_V_VALUES:
        g = results["grid"][f"sc{sc}_sv{sv}"]
        print(f"{sc:>10} {sv:>8} {g['N_star']:>5} {g['best_mean']:>11.4f} {g['depth_mean']:>11.4f} {g['breadth_advantage']:>+12.4f}")
print("--- example curves (mean realized-true vs N) ---")
for key in ["sc0.0_sv0.0", "sc0.6_sv0.0", "sc0.6_sv0.5"]:
    g = results["grid"][key]
    row = " ".join(f"N{N}:{g['curve'][N]['mean']:.3f}" for N in N_VALUES)
    print(f"  {key} (N*={g['N_star']}): {row}")
print("PRE-REG: P1 sc=0 -> N*=1; P2 high sc + low sv -> N*>1; P3 raising sv lowers N* (selection tax caps parallelism); "
      "P4 breadth-adv = coverage gain realized only if verifier can select.")
print("DONE")
