"""Cycle 126 (COMPOSE: c82 x c83 x c102): cost-optimal MIX of cheap-LOCAL + expensive-GROUNDING checks under a
verification BUDGET. c102 found a GROUNDING check (re-verify vs the original spec) catches systematic DRIFT while a
LOCAL-consistency check does NOT (catch-rate 0). Naive read: "grounding dominates -> just use grounding." BUT grounding
is more EXPENSIVE (c83 checkpoint-cost). Under a fixed budget B, with TWO error classes -- iid JUMPS (discontinuities a
cheap local check catches) AND slow DRIFT (only grounding catches, c102) -- how should you SPLIT the budget?

Model: chain of L steps. Per step: a JUMP error w.p. p_J (magnitude |N(0,sig_J)|) and a constant DRIFT delta (locally
invisible, accumulates). Cost = INTEGRAL of total uncaught error over the chain (c82 propagation: an uncaught error
corrupts every downstream step until caught). A LOCAL check (cost 1) resets the JUMP accumulator since the last check
(a jump is a visible discontinuity) but NOT drift (c102). A GROUNDING check (cost r>1) re-anchors to spec -> resets
BOTH. Budget B = n_L*1 + n_G*r. Place each type evenly. Sweep n_G; find the cost-min allocation. numpy-only, bootstrap CI.

PRE-REG: P1 drift-only -> pure grounding (corner). P2 jumps-only -> pure local (corner). P3 BOTH -> INTERIOR optimum,
most budget on cheap local + a LOW grounding rate. P4 optimal n_G tracks DRIFT (delta), ~insensitive to jump rate p_J."""
import numpy as np, json, os
RNG = np.random.default_rng(0)
L, B, R = 100, 40, 20000
SIG_J = 1.0
os.makedirs("runs/cycle126_check_budget_mix", exist_ok=True)

def even_positions(n, L):
    if n <= 0: return np.array([], dtype=int)
    # n evenly-spaced check positions in [0, L-1] (centered in their segments)
    return np.unique(((np.arange(n) + 0.5) * L / n).astype(int))

def sim_cost(n_L, n_G, p_J, delta, reps=R, seed_off=0):
    """Mean integral-of-error cost over reps for a given allocation. Grounding positions reset both; local resets jumps."""
    rng = np.random.default_rng(seed_off)
    gpos = set(even_positions(n_G, L).tolist())
    # local placed evenly among steps NOT used by grounding
    free = [i for i in range(L) if i not in gpos]
    lpos = set()
    if n_L > 0 and free:
        idx = np.unique(((np.arange(n_L) + 0.5) * len(free) / n_L).astype(int))
        idx = idx[idx < len(free)]
        lpos = set(np.array(free)[idx].tolist())
    jumps = (rng.random((reps, L)) < p_J) * np.abs(rng.normal(0, SIG_J, (reps, L)))
    jump_acc = np.zeros(reps); drift_acc = np.zeros(reps)
    per_rep_cost = np.zeros(reps)
    for i in range(L):
        jump_acc += jumps[:, i]
        drift_acc += delta
        per_rep_cost += jump_acc + drift_acc          # c82: integrate uncaught error (propagation)
        if i in gpos:                                  # grounding: re-anchor to spec -> reset BOTH (c102)
            jump_acc[:] = 0.0; drift_acc[:] = 0.0
        elif i in lpos:                                # local: catch the discontinuity only (NOT drift, c102)
            jump_acc[:] = 0.0
    return per_rep_cost                                 # (reps,) for bootstrap

def boot_ci(x, B_boot=2000, seed=1):
    rng = np.random.default_rng(seed); n = len(x)
    means = np.array([x[rng.integers(0, n, n)].mean() for _ in range(B_boot)])
    return float(x.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))

def sweep(p_J, delta, r, label):
    rows = []
    nG_max = B // r
    for n_G in range(0, nG_max + 1):
        n_L = B - r * n_G
        c = sim_cost(n_L, n_G, p_J, delta)
        m, lo, hi = boot_ci(c)
        rows.append({"n_G": n_G, "n_L": n_L, "cost": round(m, 3), "ci": [round(lo,3), round(hi,3)]})
    best = min(rows, key=lambda x: x["cost"])
    print(f"\n=== {label}  (p_J={p_J}, delta={delta}, r={r}, B={B}) ===")
    print(f"{'n_G':>4} {'n_L':>4} | {'cost (integ. err)':>18} | 95% CI")
    for rw in rows:
        star = "  <== MIN" if rw is best else ""
        print(f"{rw['n_G']:>4} {rw['n_L']:>4} | {rw['cost']:>18} | [{rw['ci'][0]}, {rw['ci'][1]}]{star}")
    interior = 0 < best["n_G"] < nG_max
    print(f"  -> argmin n_G*={best['n_G']} (of 0..{nG_max}); {'INTERIOR (mix)' if interior else 'CORNER'}")
    return {"label": label, "p_J": p_J, "delta": delta, "r": r, "rows": rows, "best": best, "interior": interior, "nG_max": nG_max}

out = {"model": {"L": L, "B": B, "R": R, "SIG_J": SIG_J}, "scenarios": {}}
# Controls (validate the model fires at the extremes) + the real both-present case
out["scenarios"]["P1_drift_only"] = sweep(p_J=0.0,  delta=0.05, r=4, label="P1 CONTROL: drift only (p_J=0)")
out["scenarios"]["P2_jumps_only"] = sweep(p_J=0.15, delta=0.0,  r=4, label="P2 CONTROL: jumps only (delta=0)")
out["scenarios"]["P3_both"]       = sweep(p_J=0.15, delta=0.05, r=4, label="P3 BOTH present")
# P4: does optimal n_G track DRIFT (not jumps)?  vary delta and p_J independently
print("\n=== P4: does n_G* track DRIFT (delta) or JUMPS (p_J)? ===")
p4 = {}
for delta in [0.02, 0.05, 0.10]:
    s = sweep(p_J=0.15, delta=delta, r=4, label=f"P4 vary delta={delta} (p_J fixed 0.15)")
    p4[f"delta_{delta}"] = s["best"]["n_G"]
for p_J in [0.05, 0.15, 0.30]:
    s = sweep(p_J=p_J, delta=0.05, r=4, label=f"P4 vary p_J={p_J} (delta fixed 0.05)")
    p4[f"pJ_{p_J}"] = s["best"]["n_G"]
out["P4_nGstar"] = p4

json.dump(out, open("runs/cycle126_check_budget_mix/score.json", "w"), indent=1)
print("\n--- READ THE RAW ROWS (don't trust this summary) ---")
b1, b2, b3 = (out["scenarios"][k]["best"] for k in ["P1_drift_only","P2_jumps_only","P3_both"])
print(f"P1 drift-only: n_G*={b1['n_G']}/{out['scenarios']['P1_drift_only']['nG_max']} ({'CORNER-grounding' if b1['n_G']==out['scenarios']['P1_drift_only']['nG_max'] else 'interior'})")
print(f"P2 jumps-only: n_G*={b2['n_G']} ({'CORNER-local' if b2['n_G']==0 else 'interior'})")
print(f"P3 BOTH: n_G*={b3['n_G']} (n_L={b3['n_L']}); interior={out['scenarios']['P3_both']['interior']}; budget-on-local={b3['n_L']}/{B}")
print(f"P4 n_G* vs delta(0.02,0.05,0.10)={[p4['delta_0.02'],p4['delta_0.05'],p4['delta_0.1']]}; vs p_J(0.05,0.15,0.30)={[p4['pJ_0.05'],p4['pJ_0.15'],p4['pJ_0.3']]}")
print("=> if P3 interior AND n_G* rises with delta but flat across p_J: grounding-rate set by DRIFT, local-rate by JUMPS; don't pay grounding-cost to catch jumps.")
print("DONE")
