"""Cycle 118 (NEW, the under-explored GENERATION side; re-frames c7 with a generator lens): verification does NOT raise
the outcome CEILING (set by the generator — you can only select a candidate that exists); it gives an HONEST READ of
it. Decompose verification's value into (a) OUTCOME-gain (capped by the generator's best, modest on a weak generator)
vs (b) EPISTEMIC delusion-removal (the winner's-curse gap, large at all generator qualities).

Model: generator quality τ = fraction of GOOD candidates (mixture: τ good [true q~N(1,.3)], 1−τ noise [q~N(0,.3)]).
K=50 candidates; observed = q + N(0,σ). NO-VERIFY: select argmax(observed). VERIFY: re-eval top-3 on an INDEPENDENT
held-out, pick best (c8 recipe). ORACLE: max true q (the generator's ceiling). Metrics: true outcome of the selected
(no-verify/verify/oracle) + the reporting GAP (observed − true of the no-verify selected). numpy-only, REPS reps."""
import numpy as np, json, os
RNG = np.random.default_rng(0)
REPS, K, M, SIG = 20000, 50, 3, 0.5
os.makedirs("runs/cycle118_generator_ceiling", exist_ok=True)

def run(tau):
    good = RNG.random((REPS,K)) < tau
    q = np.where(good, RNG.normal(1,0.3,(REPS,K)), RNG.normal(0,0.3,(REPS,K)))
    obs = q + RNG.normal(0,SIG,(REPS,K)); r = np.arange(REPS)
    oracle = q.max(1)
    nv = obs.argmax(1)                                   # no-verify: pick best observed
    out_nv = q[r,nv]; gap_nv = obs[r,nv] - q[r,nv]       # winner's-curse delusion
    topm = np.argsort(-obs,1)[:,:M]                       # verify: re-eval top-m on independent held-out
    qh = np.take_along_axis(q,topm,1) + RNG.normal(0,SIG,(REPS,M))
    win = np.take_along_axis(topm, qh.argmax(1)[:,None], 1)[:,0]
    out_v = q[r,win]
    return (round(float(oracle.mean()),4), round(float(out_v.mean()),4),
            round(float(out_nv.mean()),4), round(float(gap_nv.mean()),4))

out = {"model":{"REPS":REPS,"K":K,"M":M,"SIG":SIG}, "results":{}}
print(f"K={K} candidates, eval-noise σ={SIG}, verify=re-eval top-{M}. τ = generator quality (frac good). REPS={REPS}.\n")
print(f"{'τ (gen)':>8} | {'ORACLE(ceiling)':>15} | {'VERIFY':>8} {'NO-VERIFY':>10} | {'verify−nv gain':>14} | {'GAP(delusion)':>13}")
for tau in [0.02, 0.1, 0.3, 0.6]:
    orc, v, nv, gap = run(tau)
    out["results"][tau] = {"oracle":orc, "verify":v, "no_verify":nv, "gain":round(v-nv,4), "gap":gap}
    print(f"{tau:>8} | {orc:>15} | {v:>8} {nv:>10} | {v-nv:>+14.4f} | {gap:>13}")

json.dump(out, open("runs/cycle118_generator_ceiling/score.json","w"), indent=1)
print("\n--- READ THE RAW ROWS (don't trust this summary) ---")
r = out["results"]
print(f"P1 ceiling rises w/ τ: oracle τ.02 {r[0.02]['oracle']} -> τ.6 {r[0.6]['oracle']}; verify tracks it (≈oracle)?")
print(f"P2 outcome gain (verify−nv) vs τ: " + " ".join(f"τ{t}:{r[t]['gain']:+.3f}" for t in [0.02,0.1,0.3,0.6]) + " (modest/shrinks at low τ?)")
print(f"P2 GAP (delusion) vs τ: " + " ".join(f"τ{t}:{r[t]['gap']:.3f}" for t in [0.02,0.1,0.3,0.6]) + " (large at ALL τ?)")
print("=> verification gives an HONEST READ of the generator's ceiling (removes the delusion-GAP at all τ) but doesn't RAISE it (outcome capped by the generator). Raise the ceiling with the GENERATOR; know it with verification.")
print("DONE")
