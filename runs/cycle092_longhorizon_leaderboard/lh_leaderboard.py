"""Cycle 92: do the TWO verification roles COMPOUND at a LONG-HORIZON-agent leaderboard? Composes c82 (per-run error
PROPAGATION in a length-L chain) UNDER c90 (best-of-N SELECTION over agents). On-mission (the long-horizon tail).

Model: N agents, each with a per-step error prob p_i ~ U(0.02,0.12) (skill). A long-horizon RUN = a length-L chain;
within a checkpoint segment of length k, the first error corrupts the rest of the segment (c82 propagation); a
checkpoint re-grounds the next segment. End-score (what the leaderboard sees) = fraction of USEFUL steps in ONE run.
TRUE skill = E[end-score | p_i, regime], ANALYTIC: E[useful/seg]= (1-p)(1-(1-p)^k)/p  => E[end-score]=that/k.
Leaderboard picks argmax(observed 1-run end-score). GAP = observed(winner) - trueskill(winner) = the SELECTION
inflation, in end-score units. Regimes: NO-CKPT (k=L, errors propagate to the end) vs CKPT every k=5 (c82 fix).
FIX (c90): an independent held-out re-eval = a FRESH run of the winner -> unbiased. numpy-only, REPS for CI."""
import numpy as np, json, os
RNG = np.random.default_rng(0)
REPS, N = 20000, 100
KCK = 5                                   # checkpoint interval for the CKPT regime
os.makedirs("runs/cycle092_longhorizon_leaderboard", exist_ok=True)

def true_skill(p, k):                     # analytic E[end-score]; E[useful/seg]=(1-p)(1-(1-p)^k)/p, /k
    return (1-p)*(1-(1-p)**k)/(p*k)

def one_run_endscore(p, L, k):            # ONE noisy run per agent; p shape (REPS,N) -> end-score (REPS,N)
    nseg = L//k
    err = (RNG.random((REPS, N, L)) < p[:,:,None]).reshape(REPS, N, nseg, k)
    anyerr = err.any(axis=3); first = err.argmax(axis=3)          # first error idx in segment (0-based)
    useful = np.where(anyerr, first, k)                           # steps before first error, else full k
    return useful.sum(axis=2)/L

def measure(L, k):
    p = RNG.uniform(0.02, 0.12, (REPS, N))
    ts = true_skill(p, k)
    obs = one_run_endscore(p, L, k)                               # the leaderboard's single run
    obs2 = one_run_endscore(p, L, k)                              # an INDEPENDENT held-out re-run (the fix)
    r = np.arange(REPS); win = obs.argmax(axis=1)
    gap = obs[r, win] - ts[r, win]                                # selection inflation
    reeval_gap = obs2[r, win] - ts[r, win]                        # fix: fresh run of winner, indep of selection
    sig_run = float((obs - ts).std())                            # run-to-run noise level (drives the curse)
    return (float(gap.mean()), 1.96*float(gap.std()/np.sqrt(REPS)),
            float(reeval_gap.mean()), 1.96*float(reeval_gap.std()/np.sqrt(REPS)),
            sig_run, float(ts.mean()), float(obs[r,win].mean()))

out = {"model": {"REPS": REPS, "N": N, "KCK": KCK, "p_range": [0.02,0.12]}, "no_ckpt": {}, "ckpt": {}}
Ls = [10, 20, 40, 80]
print(f"N={N} agents, p_i~U(0.02,0.12), checkpoint k={KCK}, REPS={REPS}.")
print(f"GAP = winner's observed-1-run end-score MINUS its TRUE expected skill (the leaderboard selection inflation).\n")
for label, kfun, store in [("NO-CKPT (k=L, c82 propagation to end)", lambda L: L, out["no_ckpt"]),
                           (f"CKPT every k={KCK} (c82 fix)",            lambda L: KCK, out["ckpt"])]:
    print(f"=== {label} ===")
    print(f"{'L':>5} | {'gap':>9} {'95%CI':>8} | {'reeval(fix)':>12} | {'sig_run':>8} | {'true_skill':>10} {'obs_winner':>10}")
    for L in Ls:
        k = kfun(L); g, gci, rg, rgci, sr, tsm, owm = measure(L, k)
        store[L] = {"gap": round(g,5), "ci": round(gci,5), "reeval_gap": round(rg,5), "reeval_ci": round(rgci,5),
                    "sig_run": round(sr,5), "true_skill_mean": round(tsm,4), "obs_winner_mean": round(owm,4)}
        print(f"{L:>5} | {g:>9.5f} {gci:>8.5f} | {rg:>+8.5f}±{rgci:.4f} | {sr:>8.5f} | {tsm:>10.4f} {owm:>10.4f}")
    print()

json.dump(out, open("runs/cycle092_longhorizon_leaderboard/score.json","w"), indent=1)
print("--- READ THE RAW ROWS ABOVE (don't trust this summary) ---")
nc, ck = out["no_ckpt"], out["ckpt"]
print(f"gap_NOCKPT vs L: " + " ".join(f"L{L}:{nc[L]['gap']:.4f}" for L in Ls) + "   (P1: grows with L? compounding)")
print(f"gap_CKPT   vs L: " + " ".join(f"L{L}:{ck[L]['gap']:.4f}" for L in Ls) + "   (P2: smaller + ~flat in L?)")
ratios = [nc[L]['gap']/ck[L]['gap'] for L in Ls]
print(f"gap ratio NOCKPT/CKPT vs L: " + " ".join(f"L{L}:{x:.1f}x" for L,x in zip(Ls,ratios)) + "   (checkpoint's SELECTION benefit)")
print(f"sig_run NOCKPT/CKPT @L80: {nc[80]['sig_run']:.4f} / {ck[80]['sig_run']:.4f}  (checkpoint cuts run-variance => smaller curse)")
maxfix = max(abs(d[L]['reeval_gap']) for d in (nc,ck) for L in Ls)
print(f"max |reeval gap| both regimes: {maxfix:.5f}  (P3: held-out re-eval unbiased regardless of checkpointing)")
print(f"true-skill LEVEL @L80 NOCKPT/CKPT: {nc[80]['true_skill_mean']:.3f} / {ck[80]['true_skill_mean']:.3f}  (c82 wasted-work benefit, for reference)")
print("DONE")
