"""Cycle 132 (real-agent, HARDEN c131 — BACKLOG #3b): c131 showed an agent KNOWS the winner's curse and discounts, but
over-shrinks to a GENERIC PRIOR blind to the observed (corr(est,obs)~0), crushing genuine winners; capability didn't help.
SHARP follow-up: is that a MISSING-INFORMATION failure (the agent had no σ/prior, so it used a vague domain prior) -- FIXABLE
by SUPPLYING the statistical structure (μ, σ_true, σ_obs, K) so it can compute the proper reliability-weighted shrinkage --
or an UN-OVERRIDABLE heuristic (still flat ~0.6 even with the structure)? Reuse c131's EXACT instances (same 10-candidate
tables) for a controlled within-instance comparison: c131 = no structure, c132 = structure supplied. Run Haiku+Opus.

Bayes per-candidate posterior mean (ignoring selection): (σ_obs²·μ + σ_true²·X)/(σ_true²+σ_obs²), μ=0.5 σ_true=0.5 σ_obs=0.6.
PRE-REG: P1 (FIXABLE) estimates TRACK observed (corr >> c131 ~0), ~approach the Bayes ref, high-true no longer crushed.
P2 (UN-OVERRIDABLE) still flat ~0.6 (corr~0). Sub-Q: does the agent ALSO selection-correct (lower the lucky-fake seed4)?"""
import json, os, math
MU, SIG_T, SIG_O = 0.5, 0.5, 0.6
GT = json.load(open("runs/cycle131_realagent_curse_discount/ground_truth.json"))
os.makedirs("runs/cycle132_curse_discount_with_structure", exist_ok=True)

def bayes_ref(X):  # per-candidate posterior mean (no selection correction)
    return (SIG_O**2 * MU + SIG_T**2 * X) / (SIG_T**2 + SIG_O**2)

PROMPT = ("You are estimating the true Sharpe ratio of a trading strategy. Here is the EXACT statistical setup:\n"
  "- There are 10 candidate strategies. Each strategy's TRUE Sharpe ratio was drawn independently from a Normal "
  "distribution with mean 0.5 and standard deviation 0.5.\n"
  "- Each strategy's BACKTESTED Sharpe = its true Sharpe + independent measurement noise ~ Normal(0, standard deviation 0.6).\n"
  "- You then selected the strategy with the HIGHEST backtested Sharpe out of the 10.\n\n"
  "The 10 backtested Sharpes are:\n{table}\n\nThe selected (highest) is Strategy {w}: backtest = {obs}.\n\n"
  "Using this statistical structure, give your single best estimate of the SELECTED strategy's TRUE Sharpe ratio. "
  "Show your calculation briefly, then end with exactly: 'ESTIMATE: <number>'")

meta = []
for g in GT:
    obs = g["obs"]; w = g["winner_idx"]
    table = "\n".join(f"  Strategy {i+1}: {obs[i]:.2f}" for i in range(len(obs)))
    base = f"runs/cycle132_curse_discount_with_structure/traj{g['seed']}"
    open(base+"_struct.txt","w").write(PROMPT.format(table=table, w=w+1, obs=g["obs_winner"]))
    meta.append({"seed":g["seed"], "obs_winner":g["obs_winner"], "true_winner":g["true_winner"],
                 "bayes_ref":round(bayes_ref(g["obs_winner"]),3), "c131_haiku":None, "c131_opus":None})
json.dump(meta, open("runs/cycle132_curse_discount_with_structure/ref.json","w"), indent=1)
print("Built 4 structure-supplied prompts (same instances as c131). Bayes per-candidate reference (tracks observed):")
for m in meta:
    print(f"  seed {m['seed']}: obs={m['obs_winner']} true={m['true_winner']} bayes_ref(no-selection)={m['bayes_ref']} "
          f"(c131 blind prior was ~0.6-1.0). A FIXABLE agent should TRACK observed (~bayes_ref or lower w/ selection); "
          f"an UN-OVERRIDABLE one stays flat ~0.6.")
print("=> compare est-with-structure (c132) vs c131's blind estimates on the SAME instances; does corr(est,obs) rise?")
print("DONE")
