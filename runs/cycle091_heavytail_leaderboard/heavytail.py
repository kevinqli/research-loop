"""Cycle 91: HARDEN c90 (leaderboard winner's curse) — does the test-set-size design rule survive NON-GAUSSIAN
skill/noise? c90 assumed Gaussian skill+noise; real agent-skill / per-instance scores are plausibly heavy-tailed.
Three arms, all holding VARIANCE fixed (sig_skill=0.3, sig_noise=0.1) so ONLY the tail-shape changes (t df=3 = heavy
tail, finite variance):
  ARM 1: heavy-tailed SKILL,  Gaussian score-noise  -> predict gap SMALLER (winner is a true star; rule conservative)
  ARM 2: Gaussian skill, heavy-tailed SCORE-NOISE     -> predict gap LARGER, Gaussian M-rule UNDER-sizes by (E_hv/E_g)^2
  ARM 3 (CLT reconciliation): score = MEAN of M heavy-tailed (finite-var) instances; sweep M -> gap converges to the
         Gaussian-noise gap as M grows (CLT), so the heavy-NOISE danger is a SMALL-M / infinite-variance phenomenon.
FIX check (all arms): independent held-out re-eval of the winner should be unbiased REGARDLESS of distribution
(fresh noise is independent of selection). numpy-only, REPS for CI. Reuses the c90 model."""
import numpy as np, json, os
RNG = np.random.default_rng(0)
REPS = 60000
os.makedirs("runs/cycle091_heavytail_leaderboard", exist_ok=True)
SK, NO = 0.3, 0.1                       # fixed stddevs (NO=0.1 == sig_inst=1 / sqrt(M=100), c90's working point)

def draw(dist, var, shape):
    """zero-mean draws with the given VARIANCE; dist in {'gauss','t3','t5'}."""
    if dist == "gauss": return RNG.normal(0, np.sqrt(var), shape)
    df = int(dist[1:]); scale = np.sqrt(var*(df-2)/df)   # standard_t(df) has variance df/(df-2)
    return RNG.standard_t(df, shape)*scale

def gap_vs_N(skill_dist, noise_dist, Ns, reps=REPS):
    out = {}
    for N in Ns:
        skill = draw(skill_dist, SK**2, (reps, N))
        noise = draw(noise_dist, NO**2, (reps, N))
        win = (skill+noise).argmax(axis=1); r = np.arange(reps)
        rep_gap = noise[r, win]                                  # reported-true = the winner's noise
        reeval  = draw(noise_dist, NO**2, reps)                  # fresh independent re-eval noise
        out[N] = {"gap": round(float(rep_gap.mean()),5), "ci": round(1.96*float(rep_gap.std()/np.sqrt(reps)),5),
                  "reeval_gap": round(float(reeval.mean()),5)}   # E[fresh noise]=0 by construction (distribution-free)
    return out

Ns = [1, 2, 10, 100, 1000]
res = {"model": {"SK": SK, "NO": NO, "REPS": REPS}, "arms": {}}
print(f"sig_skill={SK}, sig_noise={NO}, REPS={REPS}. Reporting gap = E[winner's noise] (reported - true).\n")
print(f"{'N':>6} | {'GAUSS base':>16} | {'ARM1 heavy-SKILL':>18} | {'ARM2 heavy-NOISE':>18}")
base = gap_vs_N("gauss","gauss",Ns); a1 = gap_vs_N("t3","gauss",Ns); a2 = gap_vs_N("gauss","t3",Ns)
res["arms"]["gauss_base"]=base; res["arms"]["heavy_skill"]=a1; res["arms"]["heavy_noise"]=a2
for N in Ns:
    print(f"{N:>6} | {base[N]['gap']:>8.5f}±{base[N]['ci']:.5f} | {a1[N]['gap']:>10.5f}±{a1[N]['ci']:.5f} | {a2[N]['gap']:>10.5f}±{a2[N]['ci']:.5f}")

# under-sizing factor of the Gaussian M-rule under heavy NOISE = (gap_heavy / gap_gauss)^2 at large N
us = (a2[1000]['gap']/base[1000]['gap'])
res["undersizing"] = {"gap_ratio_N1000": round(us,3), "M_undersize_factor": round(us**2,3)}
fixmax = max(abs(d[N]['reeval_gap']) for d in (base,a1,a2) for N in Ns)
res["fix_max_abs_reeval_gap"] = round(fixmax,5)

# ARM 3: CLT — score = mean of M heavy-tailed (t3, finite-var) instances; per-instance var=1 so score-noise var=1/M.
# Compare the heavy (per-instance t3, averaged) gap to a Gaussian-noise gap at the SAME score-noise var, vs M. N=100.
print(f"\nARM3 (CLT): score = mean of M heavy-tailed (t3, per-instance var=1) instances, N=100 agents.")
print(f"{'M':>6} | {'heavy-inst gap':>15} | {'gauss-noise gap':>16} | {'ratio':>7}  (-> 1 as M grows = CLT)")
arm3 = {}
N3 = 100
for M in [3, 10, 30, 100, 300]:
    inst = RNG.standard_t(3, (REPS, N3, M))*np.sqrt((3-2)/3)     # per-instance var=1, heavy
    score_noise_hv = inst.mean(axis=2)                           # mean over M instances -> var 1/M
    skill = RNG.normal(0, SK, (REPS, N3))
    win = (skill+score_noise_hv).argmax(axis=1); r=np.arange(REPS)
    g_hv = float(score_noise_hv[r,win].mean())
    gn = RNG.normal(0, 1/np.sqrt(M), (REPS, N3))                 # Gaussian score-noise, SAME var 1/M
    win2 = (skill+gn).argmax(axis=1); g_g = float(gn[r,win2].mean())
    arm3[M] = {"heavy_inst_gap": round(g_hv,5), "gauss_gap": round(g_g,5), "ratio": round(g_hv/g_g,3)}
    print(f"{M:>6} | {g_hv:>15.5f} | {g_g:>16.5f} | {g_hv/g_g:>7.3f}")
res["arm3_clt"]=arm3
json.dump(res, open("runs/cycle091_heavytail_leaderboard/score.json","w"), indent=1)

print("\n--- READ THE RAW ROWS ABOVE (don't trust this summary) ---")
print(f"summary_N1_control: base {base[1]['gap']:+.5f} / heavySkill {a1[1]['gap']:+.5f} / heavyNoise {a2[1]['gap']:+.5f}  (all ~0, no selection)")
print(f"summary_heavySKILL_vs_base_N1000: {a1[1000]['gap']:.5f} vs {base[1000]['gap']:.5f}  (P1: heavy skill -> SMALLER gap?)")
print(f"summary_heavyNOISE_vs_base_N1000: {a2[1000]['gap']:.5f} vs {base[1000]['gap']:.5f}  (P2: heavy noise -> LARGER gap?)")
print(f"summary_M_undersize_factor: {us**2:.2f}x  (Gaussian M-rule under-sizes test set by this under heavy score-noise)")
print(f"summary_fix_max_abs_reeval_gap: {fixmax:.5f}  (P4: held-out fix unbiased in ALL arms, distribution-free)")
print(f"summary_arm3_clt_ratio_M3_to_M300: {arm3[3]['ratio']:.3f} -> {arm3[300]['ratio']:.3f}  (P3: -> 1 as M grows = CLT rescues large-M)")
print("DONE")
