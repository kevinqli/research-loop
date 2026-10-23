"""Cycle 107 (HARDEN c90, the centerpiece recursion): does CORRELATED eval-noise (a SHARED test set) change the
leaderboard winner's curse? c90 assumed INDEPENDENT eval-noise across agents. But a SHARED test set induces POSITIVE
cross-agent noise correlation (a 'hard' instance lowers everyone's score together = a common shock). Opus hand-waved
this in c99 ('correlated noise reduces effective K'); here we MEASURE it.

Model: N agents, true skill s_i ~ N(0, sig_s^2). Eval noise e_i = sqrt(rho)*z_common + sqrt(1-rho)*z_i, with
z_common shared across agents, z_i idiosyncratic, both N(0, sig_n^2) -> Corr(e_i,e_j)=rho, Var(e_i)=sig_n^2 (marginal
noise FIXED, only the cross-agent CORRELATION varies). score=s+e; select argmax; gap = E[score_sel - s_sel] = E[e_sel].
Predict gap(rho) = sqrt(1-rho)*gap(0) (the common shock shifts all agents equally -> doesn't bias the RANKING; only
the idiosyncratic sqrt(1-rho) part drives selection). => M_needed proportional to (1-rho). numpy-only, REPS reps."""
import numpy as np, json, os
RNG = np.random.default_rng(0)
REPS, N, SIG_S, SIG_N = 60000, 100, 0.3, 1.0
os.makedirs("runs/cycle107_correlated_eval", exist_ok=True)

def gap(rho):
    s = RNG.normal(0, SIG_S, (REPS, N))
    zc = RNG.normal(0, SIG_N, (REPS, 1))                 # common shock (shared across agents this rep)
    zi = RNG.normal(0, SIG_N, (REPS, N))                 # idiosyncratic
    e = np.sqrt(rho)*zc + np.sqrt(1-rho)*zi
    score = s + e; r = np.arange(REPS); win = score.argmax(1)
    g = float(e[r, win].mean()); ci = 1.96*float(e[r, win].std()/np.sqrt(REPS))
    return g, ci

rhos = [0.0, 0.2, 0.5, 0.8, 0.95, 1.0]
g0, _ = gap(0.0)
out = {"model": {"REPS":REPS,"N":N,"SIG_S":SIG_S,"SIG_N":SIG_N}, "sweep": {}}
print(f"N={N}, sig_skill={SIG_S}, sig_noise={SIG_N} (marginal noise FIXED), REPS={REPS}.")
print(f"gap(0)={g0:.4f} (the c90 iid leaderboard curse). Predict gap(rho)=sqrt(1-rho)*gap(0).\n")
print(f"{'rho':>6} | {'sim gap':>9} {'95%CI':>8} | {'pred sqrt(1-rho)*gap0':>21} | {'M_factor (1-rho)':>16}")
for rho in rhos:
    g, ci = gap(rho); pred = np.sqrt(1-rho)*g0
    out["sweep"][rho] = {"gap": round(g,5), "ci": round(ci,5), "pred": round(float(pred),5)}
    print(f"{rho:>6} | {g:>9.5f} {ci:>8.5f} | {pred:>21.5f} | {1-rho:>16.2f}")

json.dump(out, open("runs/cycle107_correlated_eval/score.json","w"), indent=1)
print("\n--- READ THE RAW ROWS (don't trust this summary) ---")
sw = out["sweep"]
maxrel = max(abs(sw[r]["gap"]-sw[r]["pred"])/max(sw[r]["gap"],1e-6) for r in [0.2,0.5,0.8] )
print(f"gap vs rho: " + " ".join(f"{r}:{sw[r]['gap']:.4f}" for r in rhos) + "  (falls to ~0 at rho=1?)")
print(f"sim vs sqrt(1-rho) closed form, max rel-err (rho .2-.8): {maxrel:.1%}")
print(f"rho=1.0 gap: {sw[1.0]['gap']:.5f} (shared test set -> ranks on pure skill, curse ~0?)")
print(f"=> a SHARED test set (corr rho) shrinks the leaderboard curse by sqrt(1-rho); M_needed ~ (1-rho)x the iid rule (c90).")
print("DONE")
