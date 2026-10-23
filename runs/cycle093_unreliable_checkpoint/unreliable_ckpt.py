"""Cycle 93: HARDEN c92 (long-horizon leaderboard) — does "need BOTH fixes" survive an UNRELIABLE checkpoint?
c92 used perfect checkpoint recall r=1. c83 showed a costly+unreliable checkpoint can fail to pay. So sweep recall r
in [0,1]: a checkpoint catches+fixes a corrupted segment w.p. r; w.p. (1-r) it MISSES and corruption propagates into
later segments until some checkpoint catches it (so r interpolates ckpt[r=1] <-> no-ckpt[r=0]).
Measure: (1) leaderboard SELECTION gap vs r; (2) held-out re-eval gap vs r (should stay ~0 for ALL r, distribution-
free); (3) true_skill vs r (the work axis). Adversarial check: analytic true-skill recurrence vs direct MC.
Builds on c92. numpy-only, REPS reps. N agents, per-step error p~U(0.02,0.12), L=80, checkpoint every k=5."""
import numpy as np, json, os
RNG = np.random.default_rng(0)
REPS, N, L, K = 20000, 100, 80, 5
os.makedirs("runs/cycle093_unreliable_checkpoint", exist_ok=True)

def simulate(p, r, n):
    """n independent runs per agent; p shape (REPS,N). Returns end-quality (n,REPS,N). Segment-sequential with recall."""
    nseg = L//K; shape = (n,)+p.shape
    corrupted = np.zeros(shape, dtype=bool); useful = np.zeros(shape)
    for _ in range(nseg):
        err = RNG.random(shape+(K,)) < p[...,None]                 # (n,REPS,N,K), p broadcasts on last axis
        anyerr = err.any(axis=-1); first = err.argmax(axis=-1)
        useful_clean = np.where(anyerr, first, K)
        useful += np.where(corrupted, 0, useful_clean)             # carried-over corruption => 0 useful
        end_corrupt = corrupted | anyerr
        caught = (RNG.random(shape) < r) & end_corrupt             # checkpoint catches w.p. r
        corrupted = end_corrupt & ~caught
    return useful/L

def true_skill_recurrence(p, r):
    """exact E[end-quality] via deterministic corruption-prob propagation over segments."""
    nseg = L//K; q = 1-(1-p)**K; u_clean = (1-p)*(1-(1-p)**K)/p
    x = np.zeros_like(p); ts = np.zeros_like(p)                    # x = P(enter segment corrupted), starts 0
    for _ in range(nseg):
        ts += (1-x)*u_clean
        x = (x + (1-x)*q)*(1-r)                                    # end-corrupt prob, survives checkpoint w.p.(1-r)
    return ts/L

# --- adversarial check: recurrence vs MC at r in {0,0.5,1} (one fixed p) ---
print("ADVERSARIAL CHECK: analytic true-skill recurrence vs direct MC (p=0.07, 200k runs):")
pchk = np.array([0.07])
for r in [0.0, 0.5, 1.0]:
    mc = float(simulate(pchk, r, 4000).mean()); an = float(true_skill_recurrence(pchk, r)[0])
    print(f"  r={r}: recurrence {an:.4f}  MC {mc:.4f}  diff {abs(an-mc):.4f}")

# --- main sweep: gap, held-out fix, true_skill vs recall r ---
print(f"\nN={N}, p~U(0.02,0.12), L={L}, checkpoint k={K}, REPS={REPS}.")
print(f"{'recall r':>9} | {'sel. gap':>9} {'95%CI':>8} | {'reeval(fix)':>12} | {'true_skill':>10} | {'obs_winner':>10}")
out = {"model": {"REPS":REPS,"N":N,"L":L,"K":K}, "sweep": {}}
for r in [1.0, 0.9, 0.7, 0.5, 0.3, 0.1, 0.0]:
    p = RNG.uniform(0.02, 0.12, (REPS, N))
    ts = true_skill_recurrence(p, r)
    obs = simulate(p, r, 1)[0]; obs2 = simulate(p, r, 1)[0]
    rr = np.arange(REPS); win = obs.argmax(axis=1)
    gap = obs[rr,win]-ts[rr,win]; reeval = obs2[rr,win]-ts[rr,win]
    out["sweep"][r] = {"gap": round(float(gap.mean()),5), "ci": round(1.96*float(gap.std()/np.sqrt(REPS)),5),
                       "reeval_gap": round(float(reeval.mean()),5), "true_skill": round(float(ts.mean()),4),
                       "obs_winner": round(float(obs[rr,win].mean()),4)}
    d = out["sweep"][r]
    print(f"{r:>9} | {d['gap']:>9.5f} {d['ci']:>8.5f} | {d['reeval_gap']:>+8.5f}    | {d['true_skill']:>10.4f} | {d['obs_winner']:>10.4f}")

json.dump(out, open("runs/cycle093_unreliable_checkpoint/score.json","w"), indent=1)
print("\n--- READ THE RAW ROWS ABOVE (don't trust this summary) ---")
sw = out["sweep"]
print(f"gap vs r: " + " ".join(f"r{r}:{sw[r]['gap']:.3f}" for r in [1.0,0.7,0.5,0.3,0.1,0.0]) + "   (P1: rises as r drops, ckpt<->no-ckpt)")
print(f"true_skill vs r: " + " ".join(f"r{r}:{sw[r]['true_skill']:.3f}" for r in [1.0,0.7,0.5,0.3,0.1,0.0]) + "   (P4: degrades too = work axis)")
maxfix = max(abs(sw[r]['reeval_gap']) for r in sw)
print(f"max |reeval gap| over ALL r: {maxfix:.5f}  (P3: held-out re-eval unbiased REGARDLESS of recall = the robust fix)")
print(f"gap r=1 (perfect ckpt) {sw[1.0]['gap']:.3f} -> r=0 (broken ckpt) {sw[0.0]['gap']:.3f}  ({sw[0.0]['gap']/sw[1.0]['gap']:.1f}x worse)")
print("DONE")
