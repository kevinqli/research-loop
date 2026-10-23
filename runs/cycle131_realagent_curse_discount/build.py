"""Cycle 131 (real-agent probe, high-α; the CENTERPIECE on a real agent — the KNOWING-DOING gap): c98 showed a frontier
agent KNOWS the winner's curse abstractly (states it unprompted when designing a benchmark). Does that knowledge TRANSFER
to BEHAVIOR when it just estimates a selected candidate's value (not cued to think about selection bias)? Pre-build K
candidates with KNOWN true values + noisy observed scores (meth-24); the max-observed 'winner' is inflated (observed >>
true = the curse gap). Ask the agent to pick the best and estimate its TRUE forward value. Does it DISCOUNT the observed
(applies the curse it knows) or report ~observed (knowing-doing gap)? Two framings: PLAIN vs a HINT naming the selection.

discount = (observed_winner - estimate) / (observed_winner - true_winner):  0 = no discount (cursed), 1 = full (oracle).
PRE-REG: P1 PLAIN ~0 discount (knowing-doing gap). P2 HINT discounts toward truth. ALT: PLAIN also discounts (no gap,
capable self-correct, tempers the moat). numpy-only build + emit prompts."""
import numpy as np, json, os
RNG = np.random.default_rng(0)
K, MU, SIG_TRUE, SIG_OBS = 10, 0.5, 0.5, 0.6
os.makedirs("runs/cycle131_realagent_curse_discount", exist_ok=True)

def make(seed):
    rng = np.random.default_rng(seed)
    true = rng.normal(MU, SIG_TRUE, K)
    obs = true + rng.normal(0, SIG_OBS, K)
    w = int(obs.argmax())
    return {"seed": seed, "true": [round(float(t),3) for t in true], "obs": [round(float(o),2) for o in obs],
            "winner_idx": w, "obs_winner": round(float(obs[w]),2), "true_winner": round(float(true[w]),3),
            "curse_gap": round(float(obs[w]-true[w]),3), "mean_true": round(float(true.mean()),3)}

trajs = [make(s) for s in [1,2,3,4]]
PLAIN = ("You are allocating capital across trading strategies. Below are {K} candidate strategies, each with its "
  "BACKTESTED (in-sample) Sharpe ratio:\n{table}\n\nYou will deploy exactly ONE strategy.\n"
  "(1) Which strategy do you deploy?\n(2) Give your single best NUMERICAL estimate of that strategy's TRUE going-forward "
  "(out-of-sample) Sharpe ratio.\nThink briefly, then end with exactly: 'ESTIMATE: <number>'")
HINT = ("You are allocating capital across trading strategies. Below are {K} candidate strategies, each with its "
  "BACKTESTED (in-sample) Sharpe ratio:\n{table}\n\nNote: each backtest Sharpe is a NOISY estimate of the strategy's true "
  "Sharpe, and you are about to pick the MAXIMUM of {K} such noisy numbers.\nYou will deploy exactly ONE strategy.\n"
  "(1) Which strategy do you deploy?\n(2) Give your single best NUMERICAL estimate of that strategy's TRUE going-forward "
  "(out-of-sample) Sharpe ratio.\nThink briefly, then end with exactly: 'ESTIMATE: <number>'")

meta = []
for t in trajs:
    table = "\n".join(f"  Strategy {i+1}: backtest Sharpe = {t['obs'][i]:.2f}" for i in range(K))
    base = f"runs/cycle131_realagent_curse_discount/traj{t['seed']}"
    open(base+"_plain.txt","w").write(PLAIN.format(K=K, table=table))
    open(base+"_hint.txt","w").write(HINT.format(K=K, table=table))
    meta.append(t)
json.dump(meta, open("runs/cycle131_realagent_curse_discount/ground_truth.json","w"), indent=1)
print("Built 4 trajectories x 2 framings (plain/hint). Ground truth (the winner's observed is INFLATED vs its true):")
for t in meta:
    print(f"  seed {t['seed']}: winner=Strategy{t['winner_idx']+1} obs={t['obs_winner']} true={t['true_winner']} "
          f"curse_gap={t['curse_gap']} (mean_true={t['mean_true']}). A cursed agent reports ~{t['obs_winner']}; oracle ~{t['true_winner']}; shrinkage in between.")
print("=> discount=(obs_w - estimate)/(obs_w - true_w): 0=cursed (no discount), 1=oracle. Test PLAIN (knowing-doing gap?) vs HINT.")
print("DONE")
