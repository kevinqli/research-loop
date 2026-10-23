"""Cycle 35: given a FIXED compute budget, how should a parallel-agent lab split it between SEARCH (more agents K)
and VERIFICATION (held-out re-evals m)? The practical fleet-design question. Cost(generate a candidate)=1,
cost(a held-out re-eval)=c. Budget B = K + c*m. Strategy: generate K candidates (each a noisy in-sample eval),
re-eval the top-m by in-sample on a held-out set, pick the best held-out. True perf = theta of the final pick.
More K -> a better candidate EXISTS but selection is noisier (winner's curse); more m -> recover truth among the top.
Pre-reg: an INTERIOR optimum m* (pure search OR pure-verify are both suboptimal); m* GROWS with eval-noise and as
re-eval cheapens (c down). numpy-only, REPS reps. MPS-exploratory (sim of the mechanism, like cycles 7-10)."""
import numpy as np
from math import sqrt
B, MU, S_TRUE, REPS = 100, 0.80, 0.05, 3000
rng = np.random.default_rng(0)

def true_of_pick(K, m, s_eval, s_reeval):
    K = max(m, int(K))
    theta = rng.normal(MU, S_TRUE, K)
    val = theta + rng.normal(0, s_eval, K)
    top = np.argsort(-val)[:m]                       # top-m by in-sample
    held = theta[top] + rng.normal(0, s_reeval, m)   # re-eval on held-out
    return float(theta[top[int(np.argmax(held))]])   # pick best held-out -> its TRUE perf

def best_split(s_eval, c, s_reeval=None):
    s_reeval = s_eval if s_reeval is None else s_reeval
    ms = [m for m in range(1, int(B/c)) if (B - c*m) >= m]    # K=B-c*m must be >= m
    perf = {m: np.mean([true_of_pick(B - c*m, m, s_eval, s_reeval) for _ in range(REPS)]) for m in ms}
    mstar = max(perf, key=perf.get)
    return mstar, perf[mstar], perf[1], perf[max(ms)]        # m*, best, pure-search(m=1), most-verify

print(f"budget B={B}, true~N({MU},{S_TRUE}), REPS={REPS}\n")
print("A) optimal verification budget m* vs eval-noise (c=1, re-eval as costly as a candidate)")
print(f"{'s_eval':>7} | {'m* (verify budget)':>18} | {'K* (search)':>11} | {'true@m*':>9} | {'true@pure-search(m=1)':>22} | {'gain':>7}")
import json; A=[]
for s_eval in [0.02, 0.05, 0.1, 0.2]:
    mstar, best, pure, _ = best_split(s_eval, c=1)
    print(f"{s_eval:>7} | {mstar:>18} | {B-mstar:>11} | {best:>9.4f} | {pure:>22.4f} | {best-pure:>+7.4f}")
    A.append({"s_eval":s_eval,"m_star":mstar,"K_star":B-mstar,"true_at_mstar":float(best),"true_pure_search":float(pure)})
print("\nB) optimal verification budget m* vs re-eval cost c (s_eval=0.1)")
print(f"{'c (re-eval cost)':>16} | {'m* (verify budget)':>18} | {'verify fraction c*m*/B':>22}")
Bc=[]
for c in [1.0, 0.5, 0.2, 0.1]:
    mstar, best, pure, _ = best_split(0.1, c=c)
    frac = c*mstar/B
    print(f"{c:>16} | {mstar:>18} | {frac:>22.2f}")
    Bc.append({"c":c,"m_star":mstar,"verify_fraction":float(frac)})
json.dump({"A_vs_noise_c1":A,"B_vs_cost_seval0.1":Bc}, open("runs/cycle035_budget_split/score.json","w"), indent=1)
print("\n--- read the rows above ---")
print(f"(A) SOLID: pure-search (m=1, all budget on agents) is BEATEN by an interior split at every noise level (gain up to +{max(a['true_at_mstar']-a['true_pure_search'] for a in A):.4f}). m* rises from ~{A[0]['m_star']} (low noise) toward ~10 (moderate), roughly but NOT strictly monotone ({[a['m_star'] for a in A]}) — the perf curve is flat-topped so m* jitters.")
print(f"(B) PRE-REG REFUTED: m* is ~CONSTANT, insensitive to re-eval cost ({[b['m_star'] for b in Bc]}) — verification's benefit saturates by ~top-10 (cycle 8). Cheaper re-eval LOWERS the budget FRACTION spent verifying ({Bc[0]['verify_fraction']:.2f}->{Bc[-1]['verify_fraction']:.2f}), not the count.")
print("=> allocation: don't pour all compute into MORE agents — past a point that just feeds the winner's curse; spend an interior fraction on held-out verification, MORE when evals are noisy or re-eval is cheap.")
print("DONE")
