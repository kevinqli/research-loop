"""Cycle 117 (NEW long-horizon RECOVERY decision; composes c82/c102): when a grounding checkpoint catches an error in
an L-step chain, RESTART (discard all, fresh) or REPAIR (roll back to checkpoint, continue)? Hypothesis: it depends on
the ERROR TYPE (c102). LOCAL error (one bad step) → REPAIR (cheap, keep the good work). DRIFT (a persistent
contaminated approach/belief) → REPAIR rolls back the STATE but the drifted BELIEF persists → re-drifts → loops
forever; only RESTART (fresh belief) clears it.

Model: chain L steps, checkpoint every K (segment). LOCAL: each segment errs iid w.p. `rate` (a fresh redo likely
passes). DRIFT: a persistent `drifted` flag onsets w.p. `rate`/segment; once drifted, every segment is bad until a
RESTART resets it (REPAIR does not). On a caught error: REPAIR retries the same segment (state rolled back, belief
unchanged); RESTART resets progress→0 AND belief→good. Metric: completion rate within a work budget + total work.
numpy-only, REPS reps."""
import numpy as np, json, os
RNG = np.random.default_rng(0)
L, K, REPS = 60, 6, 20000
BUDGET = 6*L
os.makedirs("runs/cycle117_restart_vs_repair", exist_ok=True)

def run(regime, strategy, rate):
    comp = np.zeros(REPS); work = np.zeros(REPS)
    for r in range(REPS):
        prog = 0; w = 0; drifted = False
        while prog < L and w < BUDGET:
            w += K                                    # attempt the next segment (K steps)
            if regime == "local":
                bad = RNG.random() < rate
            else:                                     # drift
                if not drifted and RNG.random() < rate: drifted = True
                bad = drifted
            if not bad:
                prog += K                             # segment good → advance
            else:                                     # caught at the checkpoint
                if strategy == "restart":
                    prog = 0; drifted = False          # discard all + reset belief (the loop redoes from scratch)
                # repair: prog unchanged, belief unchanged → retry same segment next loop
        comp[r] = prog >= L; work[r] = min(w, BUDGET)
    return round(float(comp.mean()),3), round(float(work.mean()),1)

out = {"model":{"L":L,"K":K,"BUDGET":BUDGET,"REPS":REPS}, "results":{}}
print(f"L={L} steps, checkpoint K={K}, work BUDGET={BUDGET}, REPS={REPS}. (completion rate / total work)\n")
print(f"{'regime':>8} {'rate':>5} | {'REPAIR comp/work':>18} | {'RESTART comp/work':>19} | winner")
for regime, rate in [("local",0.12), ("drift",0.12)]:
    rc, rw = run(regime,"repair",rate); sc, sw = run(regime,"restart",rate)
    out["results"][regime] = {"repair":{"comp":rc,"work":rw}, "restart":{"comp":sc,"work":sw}}
    win = "REPAIR" if (rc>sc or (rc==sc and rw<sw)) else "RESTART"
    print(f"{regime:>8} {rate:>5} | {rc:>8}/{rw:>8} | {sc:>9}/{sw:>8} | {win}")

json.dump(out, open("runs/cycle117_restart_vs_repair/score.json","w"), indent=1)
print("\n--- READ THE RAW ROWS (don't trust this summary) ---")
r = out["results"]
print(f"LOCAL: repair {r['local']['repair']} vs restart {r['local']['restart']} (repair cheaper, both complete?)")
print(f"DRIFT: repair {r['drift']['repair']} vs restart {r['drift']['restart']} (repair LOOPS/fails, restart completes?)")
print("=> recovery ∝ error type: REPAIR local errors (cheap, keep work); RESTART systematic drift (repair re-drifts — the contaminated belief persists). The grounding checkpoint must DIAGNOSE the type to choose.")
print("DONE")
