"""Cycle 90: the WINNER'S CURSE, one level up — the BENCHMARK LEADERBOARD is itself a best-of-N selector over
AGENTS, so it suffers the very curse it measures. This is the recursion that makes the thesis a benchmark builder's literal
problem ("lead the benchmark effort"). DISTINCT from the well-known ADAPTIVE leaderboard overfitting (Dwork et al.
reusable holdout = repeated holdout queries → holdout becomes in-sample = our c50 Goodhart): this is the STATIC,
single-shot max-order-statistic curse — even with ZERO gaming, ranking N agents on a finite M-instance test set
inflates the winner. = Deflated Sharpe (Bailey & Lopez de Prado 2014, our c88) / our c7/c57 closed form, with N=#agents.

Model: N agents, true skill s_i ~ N(0, sig_skill^2). Benchmark score = mean over M test instances = s_i + e_i,
e_i ~ N(0, sig_inst^2 / M)  (so test-set size M is the DESIGN LEVER: sig_noise = sig_inst/sqrt(M)).
Leaderboard picks argmax score. REPORTING GAP = E[score_winner - trueskill_winner] (what the board claims vs truth).
FIX = an INDEPENDENT held-out re-eval of the winner (fresh M instances) — should be unbiased (our c7 fix, level up).
Closed form (c57, level up): gap = (sig_noise^2 / sig_tot) * E[max_N standard normals], sig_tot=sqrt(skill^2+noise^2).
numpy-only, REPS for CI."""
import numpy as np, json, os
RNG = np.random.default_rng(0)
REPS = 40000
SIG_INST = 1.0          # per-instance score noise (stddev across test instances)
os.makedirs("runs/cycle090_leaderboard_curse", exist_ok=True)

# --- E[max of N iid standard normals], Monte-Carlo (the asymptotic sqrt(2 ln N) is off for small N) ---
def E_maxN(N, big=400000):
    z = RNG.standard_normal((big, N))
    return float(z.max(axis=1).mean())

def sim_gap(N, M, sig_skill, reps=REPS):
    """Return (reporting_gap, reeval_gap, rank_corr) with bootstrap-ish SE over reps."""
    sig_noise = SIG_INST / np.sqrt(M)
    skill = RNG.normal(0, sig_skill, (reps, N))           # true skills
    score = skill + RNG.normal(0, sig_noise, (reps, N))   # benchmark scores (mean over M instances)
    win = score.argmax(axis=1)                            # leaderboard pick
    r = np.arange(reps)
    rep_gap = score[r, win] - skill[r, win]               # reported - true (the inflation), per rep
    # independent held-out re-eval of the WINNER: fresh noise
    reeval = skill[r, win] + RNG.normal(0, sig_noise, reps)
    rep_reeval_gap = reeval - skill[r, win]
    # does the leaderboard even rank truth? spearman-ish: corr(score-rank, skill-rank) averaged over a few reps
    return (float(rep_gap.mean()), 1.96*float(rep_gap.std()/np.sqrt(reps)),
            float(rep_reeval_gap.mean()), 1.96*float(rep_reeval_gap.std()/np.sqrt(reps)))

def closed(N, M, sig_skill, EN):
    sig_noise = SIG_INST/np.sqrt(M); sig_tot = np.sqrt(sig_skill**2 + sig_noise**2)
    return (sig_noise**2 / sig_tot) * EN

out = {"model": {"SIG_INST": SIG_INST, "REPS": REPS}, "A_vs_N": {}, "B_vs_M": {}, "C_fix": {}, "D_design": {}}

# ============ PART A: gap GROWS with leaderboard size N (fixed M=100, sig_skill=0.3) ============
M0, SK0 = 100, 0.3
print(f"=== A. Reporting gap vs leaderboard size N  (M={M0} test instances, sig_skill={SK0}, sig_inst={SIG_INST}) ===")
print(f"{'N':>6} | {'sim gap':>10} {'95%CI':>8} | {'closed-form':>12} | {'reeval(fix) gap':>16}")
ENs = {}
for N in [1, 2, 5, 10, 30, 100, 300, 1000]:
    EN = E_maxN(N); ENs[N] = EN
    g, gci, rg, rgci = sim_gap(N, M0, SK0)
    cf = closed(N, M0, SK0, EN)
    out["A_vs_N"][N] = {"sim_gap": round(g,5), "ci": round(gci,5), "closed": round(cf,5),
                        "reeval_gap": round(rg,5), "reeval_ci": round(rgci,5), "E_maxN": round(EN,4)}
    print(f"{N:>6} | {g:>10.5f} {gci:>8.5f} | {cf:>12.5f} | {rg:>+10.5f}±{rgci:.5f}")

# ============ PART B: gap FALLS with test-set size M ~ 1/sqrt(M) (fixed N=100, sig_skill=0.3) ============
N0 = 100; EN0 = ENs[100]
print(f"\n=== B. Reporting gap vs test-set size M  (N={N0} agents, sig_skill={SK0}) — the DESIGN LEVER ===")
print(f"{'M':>7} | {'sim gap':>10} {'95%CI':>8} | {'closed-form':>12} | {'gap*sqrt(M)':>11}")
for M in [10, 25, 50, 100, 250, 500, 1000, 4000]:
    g, gci, rg, rgci = sim_gap(N0, M, SK0)
    cf = closed(N0, M, SK0, EN0)
    out["B_vs_M"][M] = {"sim_gap": round(g,5), "ci": round(gci,5), "closed": round(cf,5)}
    print(f"{M:>7} | {g:>10.5f} {gci:>8.5f} | {cf:>12.5f} | {g*np.sqrt(M):>11.4f}")

# ============ PART C: the FIX — independent held-out re-eval is UNBIASED (across N) ============
print(f"\n=== C. The FIX (independent held-out re-eval of the winner) — should be ~0 at every N ===")
maxbias = max(abs(out['A_vs_N'][N]['reeval_gap']) for N in out['A_vs_N'])
print(f"   max |reeval gap| across all N in part A = {maxbias:.5f}  (vs reporting gap up to {max(out['A_vs_N'][N]['sim_gap'] for N in out['A_vs_N']):.5f})")
out["C_fix"]["max_abs_reeval_bias"] = round(maxbias,5)

# ============ PART D: the DESIGN RULE — required M to keep gap < eps (EQUAL-SKILL null = pure curse worst case) ===
# equal-skill (sig_skill=0): gap = sig_noise * E_maxN = (sig_inst/sqrt(M)) * E_maxN  <  eps
#   => M > (sig_inst * E_maxN / eps)^2 ;  with E_maxN ~ sqrt(2 ln N): M > 2 ln N (sig_inst/eps)^2
print(f"\n=== D. DESIGN RULE: test instances M needed so leaderboard gap < eps  (equal-skill null = worst case) ===")
print(f"   M_required = (sig_inst * E[max_N] / eps)^2     [exact, using MC E_maxN]")
print(f"{'eps':>6} | " + " ".join(f"N={N}".rjust(11) for N in [10,100,1000]) + "   (M_exact / M_via_sqrt(2lnN))")
for eps in [0.20, 0.10, 0.05]:
    cells = []
    for N in [10, 100, 1000]:
        EN = ENs.get(N) or E_maxN(N)
        M_exact = (SIG_INST*EN/eps)**2
        M_approx = 2*np.log(N)*(SIG_INST/eps)**2
        cells.append(f"{M_exact:7.0f}/{M_approx:<.0f}")
        out["D_design"][f"eps{eps}_N{N}"] = {"M_exact": round(M_exact,1), "M_approx_2lnN": round(M_approx,1), "E_maxN": round(EN,4)}
    print(f"{eps:>6} | " + " ".join(c.rjust(11) for c in cells))

json.dump(out, open("runs/cycle090_leaderboard_curse/score.json","w"), indent=1)
print("\n--- READ THE RAW ROWS ABOVE (don't trust this summary) ---")
gA = out["A_vs_N"]
print(f"summary_gap_N1: {gA[1]['sim_gap']:.5f}   (K=1: no selection, gap should be ~0)")
print(f"summary_gap_N100: {gA[100]['sim_gap']:.5f}  closed {gA[100]['closed']:.5f}  (sim vs closed-form match)")
print(f"summary_gap_N1000: {gA[1000]['sim_gap']:.5f}")
print(f"summary_max_reeval_bias: {out['C_fix']['max_abs_reeval_bias']:.5f}  (the held-out fix ~ unbiased)")
maxrel = max(abs(gA[N]['sim_gap']-gA[N]['closed'])/max(gA[N]['sim_gap'],1e-9) for N in gA if N>1)
print(f"summary_max_rel_err_sim_vs_closed: {maxrel:.3%}  (N>1)")
print("DONE")
