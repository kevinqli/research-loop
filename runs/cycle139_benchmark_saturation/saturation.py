"""Cycle 139 (NEW, benchmark-design; the TEMPORAL dynamic of c90 — ties c90×c50×c114): benchmark SATURATION. As the agents
a benchmark ranks IMPROVE, their true-skill SPREAD shrinks (they converge toward the frontier/ceiling). σ_noise (the
benchmark's measurement noise = σ_inst/√M) is FIXED. Does a STATIC benchmark just lose RANGE, or become actively MORE
MISLEADING? Model: N agents, true skill ~ N(0, σ_skill); score = skill + N(0, σ_noise=0.3 fixed). Sweep σ_skill from large
(early, diverse field) to small (mature, converged field). Metrics: DISCRIMINATION (Spearman score-vs-trueskill; P(top-by-
score = true-best)) and the WINNER'S-CURSE GAP (c90: E[score_winner − trueskill_winner]) + the RELATIVE curse gap/σ_skill.

PRE-REG: P1 discrimination DECAYS as σ_skill/σ_noise falls. P2 the c90 gap GROWS (toward σ_noise·E[max_N]) and the RELATIVE
curse gap/σ_skill BLOWS UP → the saturating leaderboard ranks increasingly on NOISE (doubly bad: less range AND more curse).
P3 fix: hold σ_skill/σ_noise (raise difficulty / lower σ_noise) → a benchmark must EVOLVE to track the frontier."""
import numpy as np, json, os
RNG = np.random.default_rng(0)
N, SIG_NOISE, REPS = 50, 0.3, 40000
os.makedirs("runs/cycle139_benchmark_saturation", exist_ok=True)

def spearman(a, b):  # mean Spearman across reps; a,b shape (reps,N)
    ra = a.argsort(1).argsort(1).astype(float); rb = b.argsort(1).argsort(1).astype(float)
    ra -= ra.mean(1, keepdims=True); rb -= rb.mean(1, keepdims=True)
    num = (ra*rb).sum(1); den = np.sqrt((ra**2).sum(1)*(rb**2).sum(1))
    return float(np.nanmean(num/den))

def run(sig_skill, reps=REPS):
    skill = RNG.normal(0, sig_skill, (reps, N))
    score = skill + RNG.normal(0, SIG_NOISE, (reps, N))
    r = np.arange(reps); w = score.argmax(1)
    gap = float((score[r, w] - skill[r, w]).mean())            # c90 winner's-curse gap
    p_best = float((w == skill.argmax(1)).mean())               # picks the true-best?
    sp = spearman(score, skill)
    return {"sig_skill": sig_skill, "ratio_skill_noise": round(sig_skill/SIG_NOISE,2),
            "spearman": round(sp,3), "p_pick_true_best": round(p_best,3),
            "wc_gap": round(gap,4), "relative_curse_gap_over_skill": round(gap/sig_skill,3) if sig_skill>0 else None}

print(f"N={N} agents, σ_noise={SIG_NOISE} FIXED. As agents IMPROVE, σ_skill SHRINKS (field converges).\n")
print(f"{'σ_skill':>8} {'σ_skill/σ_noise':>15} | {'Spearman':>9} | {'P(pick best)':>12} | {'WC gap':>8} | {'gap/σ_skill':>12}")
rows=[]
for ss in [1.0, 0.6, 0.4, 0.3, 0.2, 0.1, 0.05]:
    rr=run(ss); rows.append(rr)
    print(f"{ss:>8} {rr['ratio_skill_noise']:>15} | {rr['spearman']:>9} | {rr['p_pick_true_best']:>12} | {rr['wc_gap']:>8} | {rr['relative_curse_gap_over_skill']:>12}")

# P3: to maintain a target Spearman as agents converge, what σ_skill/σ_noise is needed?
print("\n=== P3: the fix — to hold discrimination you must hold σ_skill/σ_noise (raise difficulty / lower σ_noise) ===")
target=0.9
need=None
for ss in np.arange(0.05,1.5,0.05):
    if run(round(ss,2), reps=8000)["spearman"]>=target: need=round(ss/SIG_NOISE,2); break
print(f"  to keep Spearman≥{target}: need σ_skill/σ_noise ≥ ~{need} → as agents converge (σ_skill↓), you must LOWER σ_noise (more instances M, c90's M-rule) or RAISE difficulty (harder tasks re-spread σ_skill) to stay above it.")

json.dump({"N":N,"SIG_NOISE":SIG_NOISE,"vs_sig_skill":rows,"target_spearman":target,"need_ratio":need}, open("runs/cycle139_benchmark_saturation/score.json","w"), indent=1)
print("\n--- READ THE RAW ROWS ---")
print("Spearman vs σ_skill: " + " ".join(f"{r['sig_skill']}:{r['spearman']}" for r in rows) + "  (DECAYS as agents converge)")
print("P(pick true-best):   " + " ".join(f"{r['sig_skill']}:{r['p_pick_true_best']}" for r in rows) + "  (collapses)")
print("WC gap (absolute):   " + " ".join(f"{r['sig_skill']}:{r['wc_gap']}" for r in rows) + "  (GROWS toward σ_noise·E[max])")
print("RELATIVE curse gap/σ_skill: " + " ".join(f"{r['sig_skill']}:{r['relative_curse_gap_over_skill']}" for r in rows) + "  (BLOWS UP — curse dominates the shrinking skill range)")
print("=> a STATIC benchmark, as the agents it ranks IMPROVE (σ_skill→0), is DOUBLY bad: discrimination DECAYS (Spearman/P-best↓) AND the winner's curse GROWS (it ranks increasingly on NOISE — gap/σ_skill blows up). Fix: the benchmark must EVOLVE — raise difficulty / lower σ_noise to hold σ_skill/σ_noise (track the frontier), or it becomes a noise-ranker (c90 worsens under saturation; ties c50 Goodhart, c114 scarcity).")
print("DONE")
