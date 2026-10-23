"""Pillar 1<->2 SYNTHESIS: quantify the 'hallucinated progress' = optimism gap (winner's curse) when a search SELECTS
the best-of-K candidates by a noisy validation metric, and show it GROWS with search breadth K (the parallel-agents
hazard) and that a held-out RE-EVAL (the verification layer) removes it.
Model: K candidate solutions, true perf theta_i ~ N(mu, s_true). Validation score = theta_i + N(0, s_val).
  single-holdout: s_val high ; CV: s_val low (k-fold averages down the variance).
Agent picks argmax(val). Optimism gap = val[selected] - theta[selected] (overstated self-reported score).
Held-out re-eval = theta[selected] + small fresh noise (unbiased). REPS sims -> mean gap + CI. numpy only."""
import numpy as np, json
rng = np.random.default_rng(7)
REPS = 4000; MU, S_TRUE = 0.80, 0.05
S_HOLDOUT, S_CV, S_REEVAL = 0.05, 0.02, 0.02   # single-holdout noisy; CV averaged-down; fresh re-eval unbiased
Ks = [1, 2, 5, 10, 20, 50, 100]

def sim(K, s_val):
    gap = np.empty(REPS); regret = np.empty(REPS); reeval_bias = np.empty(REPS)
    for r in range(REPS):
        theta = rng.normal(MU, S_TRUE, K)              # true perf of K candidates
        val = theta + rng.normal(0, s_val, K)          # noisy validation score
        sel = int(np.argmax(val))                      # agent selects best validation
        gap[r] = val[sel] - theta[sel]                 # winner's curse: reported - true
        regret[r] = theta.max() - theta[sel]           # did selection pick a truly-good one?
        reeval_bias[r] = (theta[sel] + rng.normal(0, s_val)) - theta[sel]  # held-out re-eval (unbiased ~0)
    ci = lambda x: 1.96*x.std(ddof=1)/np.sqrt(len(x))
    return gap.mean(), ci(gap), regret.mean(), reeval_bias.mean(), ci(reeval_bias)

print(f"REPS={REPS}, true perf ~ N({MU},{S_TRUE}); single-holdout s_val={S_HOLDOUT}, CV s_val={S_CV}\n")
print(f"{'K':>4} | {'gap_holdout':>22} | {'gap_CV':>16} | {'reeval_bias(holdout)':>22} | regret_holdout")
rows = []
for K in Ks:
    gh, ch, rg, rb, crb = sim(K, S_HOLDOUT)
    gc, cc, _, _, _ = sim(K, S_CV)
    rows.append({"K":K, "gap_holdout":gh, "gap_holdout_ci":ch, "gap_CV":gc, "reeval_bias":rb, "regret_holdout":rg})
    print(f"{K:>4} | {gh:+.4f} ± {ch:.4f}  ({gh/S_TRUE:>4.1f}x s_true) | {gc:+.4f} ± {cc:.4f} | {rb:+.5f} ± {crb:.5f}   | {rg:.4f}")
print("\nINTERPRETATION:")
print(f"  optimism gap (hallucinated progress) GROWS with search breadth K: "
      f"K=1 {rows[0]['gap_holdout']:+.4f} -> K=100 {rows[-1]['gap_holdout']:+.4f}  (single-holdout)")
print(f"  CV selection shrinks the gap ~{rows[-1]['gap_holdout']/max(rows[-1]['gap_CV'],1e-9):.1f}x at K=100 "
      f"({rows[-1]['gap_holdout']:+.4f} -> {rows[-1]['gap_CV']:+.4f})")
print(f"  held-out RE-EVAL bias ~ 0 ({rows[-1]['reeval_bias']:+.5f}) -> the verification layer recovers truth")
json.dump(rows, open("runs/cycle019_survival/wc_score.json","w"), indent=1); print("\nwrote score.json")
