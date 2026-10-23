"""Cycle 231 — is ResearchClawBench's "+3.1 frontier-over-best-agent" gap WINNER'S CURSE or real per-task
SPECIALIZATION? c230 grounded the benchmark (Claude Code 21.5 [best single agent], autonomous frontier mean 24.6
[best-per-task over 7 agents], inter-agent task-score corr 0.79). The honest question this cycle: can you DECOMPOSE
the +3.1 (24.6−21.5) into curse-vs-specialization from the PUBLISHED SUMMARY STATS? meth-29 pre-reg; free numpy.

THE TEST (identification): build TWO opposite ground-truth models and calibrate BOTH to reproduce the SAME 3 summaries
(best-agent≈21.5, frontier≈24.6, inter-agent-corr≈0.79):
  MODEL A (CURSE-HEAVY): agents have ~identical PERSISTENT skill; the per-task winner is driven by NOISE e (large σ_e).
                         Re-run the benchmark → the frontier REGRESSES toward the best agent (the noise re-rolls).
  MODEL B (SPECIALIZATION-HEAVY): agents have a real persistent agent×task ability (different agents genuinely best at
                         different tasks); little noise. Re-run → the frontier is STABLE (real ability persists).
If BOTH fit the 3 summaries → the summaries are UNDER-IDENTIFIED → you CANNOT tell curse from specialization without
the per-agent×per-task matrix (= c216: can't decompose from summary stats). THE DISAMBIGUATOR = TEST-RETEST (the
loop's c216/c151 prescription): re-run with fresh noise and measure the held-out frontier — it COLLAPSES to ≈best-agent
under curse (A), STAYS ≈24.6 under specialization (B).

PRE-REG (meth-29): both models reproduce the 3 summaries (under-identified); test-retest cleanly separates them
(A: held-out frontier ≈ best-agent, gap→~0; B: held-out frontier ≈ 24.6, gap preserved) → reinforces c230's
"get the matrix / report a held-out frontier" + ties to c216 (summary stats don't identify the decomposition).
"""
import numpy as np, json, os
D = "runs/cycle231_clawbench_identification"; os.makedirs(D, exist_ok=True)
rng = np.random.default_rng(0)

NA, NT, REPS = 7, 40, 400          # 7 agents, 40 tasks, repeated benchmark instances
BASE = 20.7                         # base level so means land ~21.5 / ~24.6 (matches the real ResearchClawBench)

def metrics(score_fn):
    """score_fn(rng) -> (NA x NT) score matrix for one benchmark run. Return summary stats over REPS runs +
    a TEST-RETEST held-out frontier (frontier picked on run1, scored on an independent run2)."""
    best_means, frontier_means, corrs, heldout_frontier = [], [], [], []
    for _ in range(REPS):
        X1 = score_fn(rng)                          # run 1
        X2 = score_fn(rng)                          # independent re-run (same ground truth, fresh noise)
        agent_means = X1.mean(axis=1)               # per-agent mean over tasks
        best_a = int(np.argmax(agent_means))
        best_means.append(agent_means[best_a])
        frontier_means.append(X1.max(axis=0).mean())            # per-task max over agents (the "frontier")
        # inter-agent correlation: median over the 21 agent pairs of corr across the 40 tasks
        cs = [np.corrcoef(X1[i], X1[j])[0,1] for i in range(NA) for j in range(i+1,NA)]
        corrs.append(np.median(cs))
        # TEST-RETEST held-out frontier: which agent wins each task on run1, score THAT agent on run2, average
        winners = X1.argmax(axis=0)                  # per-task winner from run1
        heldout_frontier.append(X2[winners, np.arange(NT)].mean())
    return {"best_agent_mean": float(np.mean(best_means)),
            "frontier_mean": float(np.mean(frontier_means)),
            "inter_agent_corr": float(np.mean(corrs)),
            "heldout_frontier_mean": float(np.mean(heldout_frontier))}

# MODEL A — CURSE-HEAVY: tiny persistent skill spread, large per-task noise (tuned to hit the 3 summaries)
def model_A(rng):
    D_t = rng.normal(0, 6.0, NT)                     # shared task difficulty -> drives inter-agent corr
    s_a = rng.normal(0, 0.5, NA)                     # near-identical persistent skill
    e = rng.normal(0, 3.3, (NA, NT))                 # LARGE per-task noise -> the frontier is curse-driven
    return BASE + D_t[None,:] + s_a[:,None] + e

# MODEL B — SPECIALIZATION-HEAVY: real persistent agent x task ability, small noise
def model_B(rng):
    D_t = rng.normal(0, 6.0, NT)
    AxT = rng.normal(0, 3.2, (NA, NT))               # PERSISTENT agent x task ability (re-run keeps it)
    # NB: AxT is persistent within a (run1,run2) pair? No -- it must persist across the re-run to be "real".
    return BASE + D_t[None,:] + AxT, AxT              # return ability so the re-run can reuse it

# model_B needs persistent ability across run1/run2; wrap to share AxT within a pair
def metrics_B():
    best_means, frontier_means, corrs, heldout = [], [], [], []
    for _ in range(REPS):
        D1 = rng.normal(0, 6.0, NT); D2 = rng.normal(0, 6.0, NT)
        AxT = rng.normal(0, 3.2, (NA, NT))           # PERSISTENT real ability (shared across the re-run)
        nz = 0.8                                       # small measurement noise
        X1 = BASE + D1[None,:] + AxT + rng.normal(0, nz, (NA,NT))
        X2 = BASE + D2[None,:] + AxT + rng.normal(0, nz, (NA,NT))
        am = X1.mean(axis=1); best_means.append(am.max())
        frontier_means.append(X1.max(axis=0).mean())
        cs = [np.corrcoef(X1[i], X1[j])[0,1] for i in range(NA) for j in range(i+1,NA)]
        corrs.append(np.median(cs))
        w = X1.argmax(axis=0); heldout.append(X2[w, np.arange(NT)].mean())
    return {"best_agent_mean": float(np.mean(best_means)), "frontier_mean": float(np.mean(frontier_means)),
            "inter_agent_corr": float(np.mean(corrs)), "heldout_frontier_mean": float(np.mean(heldout))}

A = metrics(model_A)
B = metrics_B()
print(f"=== ResearchClawBench identification (NA={NA}, NT={NT}, REPS={REPS}; targets: best≈21.5, frontier≈24.6, corr≈0.79) ===")
for name, m in [("A_curse-heavy", A), ("B_specialization-heavy", B)]:
    print(f"\n[{name}]")
    print(f"  best-agent mean      {m['best_agent_mean']:.2f}   (target 21.5)")
    print(f"  frontier mean        {m['frontier_mean']:.2f}   (target 24.6)")
    print(f"  inter-agent corr     {m['inter_agent_corr']:.2f}   (target 0.79)")
    print(f"  held-out frontier    {m['heldout_frontier_mean']:.2f}   <- TEST-RETEST disambiguator")
    print(f"  frontier gap (raw)   {m['frontier_mean']-m['best_agent_mean']:+.2f}  |  held-out gap {m['heldout_frontier_mean']-m['best_agent_mean']:+.2f}")

both_fit = (abs(A['best_agent_mean']-21.5)<1.5 and abs(A['frontier_mean']-24.6)<1.5 and abs(A['inter_agent_corr']-0.79)<0.07
            and abs(B['best_agent_mean']-21.5)<1.5 and abs(B['frontier_mean']-24.6)<1.5 and abs(B['inter_agent_corr']-0.79)<0.07)
disambig = (A['heldout_frontier_mean']-A['best_agent_mean']) < 0.5*(A['frontier_mean']-A['best_agent_mean']) and \
           (B['heldout_frontier_mean']-B['best_agent_mean']) > 0.7*(B['frontier_mean']-B['best_agent_mean'])
print(f"\n--- BOTH models fit the 3 published summaries? {both_fit}  -> under-identified (can't tell curse from specialization)")
print(f"--- TEST-RETEST disambiguates? {disambig}  (A held-out gap collapses; B held-out gap preserved)")
json.dump({"A_curse": A, "B_specialization": B, "both_fit_summaries": bool(both_fit), "testretest_disambiguates": bool(disambig)},
          open(f"{D}/score.json","w"), indent=1)
print("--- wrote score.json")
