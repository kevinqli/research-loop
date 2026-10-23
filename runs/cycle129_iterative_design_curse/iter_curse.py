"""Cycle 129 (COMPOSE c128xc50xc115; the REUSE/iterative version of the benchmark-design curse): c128's fix was
'HOLD OUT calibration agents you don't use to select the design, report discrimination on those' — unbiased for ONE
selection. But a real builder ITERATES: propose a design tweak, check it on the held-out, keep it if better, repeat.
That makes the HELD-OUT itself the selection signal across rounds -> it gets winner's-cursed (Dwork's REUSABLE-HOLDOUT:
querying a holdout adaptively/repeatedly overfits it). So c128's single-shot fix is INSUFFICIENT for the iterative builder.

Model (reuses c128's AUC machinery): a candidate eval-DESIGN has true discrimination AUC=0.5(1+erf(δ/σ)); each round the
builder draws a fresh candidate design (σ_r), measures its AUC on a calibration set. Three arms over R rounds:
  (A) SEPARATE-ONCE (c128 fix): pick the best design by a SELECTION set, report its score ONCE on a held-out set. (R=1 use)
  (B) REUSE-HELDOUT: keep the running-best design by its HELD-OUT score across all R rounds, and REPORT that held-out
      score. The held-out is now the selector -> cursed; gap = reported(held-out) - true grows with R.
  (C) ROTATE: fresh calibration agents each round for the keep-decision + a FRESH set for the final report. honest.

PRE-REG: A unbiased (gap~0). B gap GROWS with R (held-out re-cursed by reuse). C gap~0 at all R (rotation/Ladder fixes it).
CONTROL: R=1 -> B == A (one query can't overfit). numpy-only, bootstrap CI."""
import numpy as np, json, os, math
DELTA = 1.0
A_CAL = 40                      # calibration-set size (agents)
REPS = 4000
os.makedirs("runs/cycle129_iterative_design_curse", exist_ok=True)

def true_auc(sigma): return 0.5 * (1 + math.erf(DELTA / sigma))

def measure_auc(sigma, reps, rng, A=A_CAL):
    """empirical AUC of a design (per-rep noise σ, shape (reps,)) on a fresh A-agent calibration set. -> (reps,)"""
    ng = nb = A // 2
    s = np.asarray(sigma, dtype=float).reshape(-1, 1)        # (reps,1) for broadcast
    good = DELTA + rng.normal(0, 1, (reps, ng)) * s
    bad = -DELTA + rng.normal(0, 1, (reps, nb)) * s
    return (good[:, :, None] > bad[:, None, :]).mean(axis=(1, 2))

def boot_ci(x, Bb=2000, seed=1):
    rng = np.random.default_rng(seed); n = len(x)
    m = np.array([x[rng.integers(0, n, n)].mean() for _ in range(Bb)])
    return float(x.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))

# design space: each round draws a candidate design with σ ~ U[0.8,1.6] (true AUC ranges ~0.74..0.94)
def draw_sigmas(R, reps, rng): return rng.uniform(0.8, 1.6, (reps, R))

def run(R, arm, seed=0):
    rng = np.random.default_rng(seed)
    sig = draw_sigmas(R, REPS, rng)                       # (reps, R) candidate designs
    trues = np.vectorize(true_auc)(sig)                   # their true AUCs
    r = np.arange(REPS)
    if arm == "A_separate_once":
        # pick best by a SELECTION set, report ONCE on an independent held-out set (c128 fix)
        sel_scores = np.stack([measure_auc(sig[:, j], REPS, np.random.default_rng(seed*9973+j)) for j in range(R)], 1)
        best = sel_scores.argmax(1)
        held = measure_auc(sig[r, best], REPS, np.random.default_rng(seed+12345))   # ONE fresh held-out read
        believed = held; true_sel = trues[r, best]
    elif arm == "B_reuse_heldout":
        # keep running-best by the held-out across all R rounds AND REPORT that held-out score (select & report conflated
        # on the SAME set across rounds -> the held-out becomes the selector -> winner's-cursed over R). Each design scores
        # the held-out with its OWN independent noise (independent seed per design column).
        held_scores = np.stack([measure_auc(sig[:, j], REPS, np.random.default_rng(seed*6311+j)) for j in range(R)], 1)
        best = held_scores.argmax(1)
        believed = held_scores[r, best]; true_sel = trues[r, best]
    elif arm == "C_rotate":
        # fresh calibration agents each round for keep-decision; FRESH set for final report
        keep_scores = np.stack([measure_auc(sig[:, j], REPS, np.random.default_rng(seed*31+j)) for j in range(R)], 1)
        best = keep_scores.argmax(1)
        believed = measure_auc(sig[r, best], REPS, np.random.default_rng(seed+99999))   # fresh final report
        true_sel = trues[r, best]
    gap = believed - true_sel
    m, lo, hi = boot_ci(gap)
    return {"R": R, "arm": arm, "believed": round(float(believed.mean()),4),
            "true": round(float(true_sel.mean()),4), "gap": round(m,4), "ci":[round(lo,4),round(hi,4)]}

out = {"model":{"A_CAL":A_CAL,"REPS":REPS,"DELTA":DELTA}, "arms":{}}
print(f"A_CAL={A_CAL} calibration agents, designs σ~U[0.8,1.6] (true AUC ~0.74..0.94).\n")
for arm in ["A_separate_once", "B_reuse_heldout", "C_rotate"]:
    out["arms"][arm] = []
    print(f"=== {arm} ===")
    print(f"{'R':>4} | {'believed':>9} {'true':>7} {'GAP':>8} | 95% CI")
    for R in [1, 3, 10, 30, 100]:
        rr = run(R, arm); out["arms"][arm].append(rr)
        print(f"{R:>4} | {rr['believed']:>9} {rr['true']:>7} {rr['gap']:>8} | [{rr['ci'][0]},{rr['ci'][1]}]")
    print()

json.dump(out, open("runs/cycle129_iterative_design_curse/score.json","w"), indent=1)
print("--- READ THE RAW ROWS (don't trust this summary) ---")
def gaps(a): return " ".join(f"R{x['R']}:{x['gap']:+.4f}" for x in out["arms"][a])
print(f"A separate-once (c128 fix): {gaps('A_separate_once')}  (should be ~0 at all R)")
print(f"B reuse-heldout: {gaps('B_reuse_heldout')}  (should GROW with R = held-out re-cursed by reuse)")
print(f"C rotate: {gaps('C_rotate')}  (should be ~0 at all R)")
print("=> c128's held-out fix is SINGLE-USE: iterating against the SAME held-out re-curses it (reusable-holdout); ROTATE calibration agents / use a Ladder across design rounds.")
print("DONE")
