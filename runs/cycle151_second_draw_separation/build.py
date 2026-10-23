"""Cycle 151 (#3c — does a real agent USE a 2nd independent draw to separate real-from-lucky? Closes the c131/132
centerpiece-on-a-real-agent arc by validating the FIX, not just re-demonstrating the problem; real-agent, high-α;
meth-24 pre-built known truth). c131/132: a capable agent KNOWS the winner's curse and discounts a selected
strategy's Sharpe, BUT from ONE backtest it over-shrinks to a generic prior and CANNOT separate a genuine winner
(true 2.16) from a lucky fake (true 0.50) — because the info to tell them apart isn't in a single observation
(c132). The loop's FIX (c7) is a SECOND INDEPENDENT draw (a fresh held-out re-eval). #3c tests whether a real agent
USES it: given the winner's backtest AND a fresh independent OOS Sharpe, does it now separate real from lucky?

5 strategies, KNOWN true forward Sharpe (meth-24). The KEY discriminating pair: real_strong vs lucky_fake have
NEARLY IDENTICAL backtests (2.1 vs 2.0) but very different OOS (1.7 vs 0.2) -> indistinguishable from the backtest
alone (c132), separable with the 2nd draw.
  ARM A (one observation = c131 replication): give ONLY the backtest, estimate each true forward Sharpe.
  ARM B (two observations = the FIX): give the backtest AND a fresh independent OOS Sharpe, estimate each.

PRE-REG:
  P1 (replicate c131): ARM A -> the agent CANNOT separate real_strong from lucky_fake (similar estimates for the two,
     both shrunk toward a generic prior); corr(estimate, true) LOW.
  P2 (THE FIX WORKS): ARM B -> the agent SEPARATES them — real_strong estimate HIGH (~consistent with its 1.7 OOS),
     lucky_fake estimate LOW (~its 0.2 OOS, recognizing the 2.0 backtest was inflated); corr(estimate, true) HIGH.
  P3 (the mechanism): in ARM B the agent should explicitly weight the OOS over the backtest for the lucky fake
     ("backtest 2.0 but OOS 0.2 = the backtest was selection/overfit; trust the independent OOS").
  If ARM B does NOT separate (agent ignores the OOS or still over-shrinks), the FIX doesn't transfer behaviorally —
  an important negative (the held-out only helps if the agent USES it).
"""
import json, os
os.makedirs("runs/cycle151_second_draw_separation", exist_ok=True)

# meth-24: known true forward Sharpe; backtest = selection-inflated; OOS = a fresh independent draw (~true + small noise)
strategies = [
  {"id": "A", "name": "alpha-momentum",  "true": 1.8, "backtest": 2.1, "oos": 1.7},   # REAL strong (high+high)
  {"id": "B", "name": "vol-carry",       "true": 0.3, "backtest": 2.0, "oos": 0.2},   # LUCKY fake (high backtest, low OOS) -- the key
  {"id": "C", "name": "mean-reversion",  "true": 1.0, "backtest": 1.2, "oos": 0.9},   # real moderate
  {"id": "D", "name": "sentiment-flow",  "true": 0.1, "backtest": 1.8, "oos": 0.0},   # LUCKY fake 2 (high backtest, ~0 OOS)
  {"id": "E", "name": "pairs-stat-arb",  "true": 0.5, "backtest": 0.6, "oos": 0.4},   # mediocre (consistent low)
]
json.dump({"strategies": strategies,
           "key_pair": ["A (real_strong)", "B (lucky_fake)"],
           "note": "A,B have near-identical backtest (2.1 vs 2.0) but OOS 1.7 vs 0.2 -> separable only with the 2nd draw"},
          open("runs/cycle151_second_draw_separation/truth.json", "w"), indent=1)

bt_table = "\n".join(f"  - {s['name']}: backtest Sharpe = {s['backtest']:.1f}" for s in strategies)
both_table = "\n".join(f"  - {s['name']}: backtest Sharpe = {s['backtest']:.1f}, fresh out-of-sample Sharpe = {s['oos']:.1f}" for s in strategies)

PROMPT_A = (
  "You are a quant evaluating 5 candidate trading strategies. Each was SELECTED from a large pool because its "
  "BACKTEST Sharpe looked good (so selection/overfitting bias is in play). For EACH strategy, estimate its TRUE "
  "forward (live) Sharpe ratio — what you'd actually expect going forward.\n\n"
  f"The backtest Sharpe of each:\n{bt_table}\n\n"
  "Give your estimate for each as 'name: <number>'. Reason briefly first, then end with a line 'ESTIMATES:' followed "
  "by the 5 'name: number' lines.")

PROMPT_B = (
  "You are a quant evaluating 5 candidate trading strategies. Each was SELECTED from a large pool because its "
  "BACKTEST Sharpe looked good (so selection/overfitting bias is in play). You ALSO have, for each, a FRESH "
  "INDEPENDENT out-of-sample (OOS) Sharpe measured on a separate later period the strategy was NOT selected on. "
  "For EACH strategy, estimate its TRUE forward (live) Sharpe ratio — what you'd actually expect going forward.\n\n"
  f"The two numbers for each:\n{both_table}\n\n"
  "Give your estimate for each as 'name: <number>'. Reason briefly first (note where backtest and OOS DISAGREE), "
  "then end with a line 'ESTIMATES:' followed by the 5 'name: number' lines.")

open("runs/cycle151_second_draw_separation/prompt_A.txt", "w").write(PROMPT_A)
open("runs/cycle151_second_draw_separation/prompt_B.txt", "w").write(PROMPT_B)
print("Built 5 strategies (meth-24 known truth) + 2 prompts.")
print("KEY pair: A real_strong (true 1.8, bt 2.1, oos 1.7) vs B lucky_fake (true 0.3, bt 2.0, oos 0.2).")
print("PRE-REG: ARM A can't separate A from B (similar est, both shrunk); ARM B separates (A high ~oos1.7, B low ~oos0.2).")
print("DONE")
