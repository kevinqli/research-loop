"""Cycle 218 (#34 follow-up — the DEPLOYABILITY power analysis of c216's task-pricing recipe). c216 (#165)
established the actionable recipe: value a candidate benchmark task by its INCREMENTAL test-retest ranking
reliability (a duplicate HURTS it −0.011, a diverse task helps +0.011). But that ±0.011 signal was measured at
REPS=1000 — a builder has ONE fixed agent panel of size N and must DECIDE 'is this candidate task redundant?'
from it. meth-28 reuses c216's EXACT model. KEY question: how large an agent panel N makes that decision reliable?

DECISION (what a builder actually does): hold a panel of N agents, measure rank-retest reliability with vs without
the candidate task → Δrank. Decide 'redundant/wasted' if Δrank ≤ 0, 'valuable' if Δrank > 0. We measure the
DECISION ACCURACY: over many independent panels, P(call a true-duplicate wasted) and P(call a true-diverse valuable).

MODEL (c216 exact): x_{i,t}=s_i+e_{i,t}; s~N(0,σs) skill; diverse task e indep N(0,σe); near-duplicate of task 0:
e = e_{·,0}+δ·N(0,σe). Two independent re-eval draws per panel (test-retest). Add one task to a base of T0 diverse.

PRE-REG (meth-29; before peeking): (P1) decision accuracy RISES with N. (P2) the ±0.011 signal needs a MODERATE
panel — at small N (10–20) the per-panel Δrank sampling noise (~0.05–0.1) SWAMPS the 0.011 separation → accuracy
near chance (~0.5–0.6); a reliable (≥90%) call needs N in the high tens / low hundreds. (P3) the DUPLICATE is
easier to call than the diverse (its hit to reliability, plus the visible noise-clone correlation, is more separable).
If accuracy is ≥90% even at N=10 → the recipe is cheap to deploy (refutes P2); if it needs N≳200 → the recipe is
real but expensive, a genuine deployability caveat for the product.
"""
import numpy as np, json, os
D = "runs/cycle218_task_pricing_power"; os.makedirs(D, exist_ok=True)

REPS, SIG_S, SIG_E = 600, 1.0, 1.0       # REPS independent panels per (N, kind)
T0 = 8
DUP_DELTA = 0.15
N_VALUES = [10, 20, 40, 80, 160, 320]

def spearman(a, b):
    ra = a.argsort().argsort(); rb = b.argsort().argsort()
    return np.corrcoef(ra, rb)[0, 1]

def panel_delta_rank(rng, N, kind):
    """One panel of N agents: Δ(test-retest ranking reliability) from adding the candidate task.
    base = T0 diverse tasks; +1 task of `kind`. Each measured twice (independent noise) = retest."""
    s = rng.normal(0, SIG_S, N)
    def draw(include):
        E = rng.normal(0, SIG_E, (N, T0))
        if not include:
            return s[:, None] + E
        if kind == "dup":
            extra = E[:, [0]] + DUP_DELTA * rng.normal(0, SIG_E, (N, 1))
        else:  # "div"
            extra = rng.normal(0, SIG_E, (N, 1))
        return s[:, None] + np.concatenate([E, extra], axis=1)
    # base rank-retest (2 indep draws of the base set) and with-candidate rank-retest
    base_rr = spearman(draw(False).mean(1), draw(False).mean(1))
    cand_rr = spearman(draw(True).mean(1), draw(True).mean(1))
    return cand_rr - base_rr

def run(N):
    rng = np.random.default_rng(1234 + N)
    d_dup = np.array([panel_delta_rank(rng, N, "dup") for _ in range(REPS)])
    d_div = np.array([panel_delta_rank(rng, N, "div") for _ in range(REPS)])
    # decision accuracy: duplicate correctly called wasted (Δ≤0); diverse correctly called valuable (Δ>0)
    acc_dup = float(np.mean(d_dup <= 0))
    acc_div = float(np.mean(d_div > 0))
    # separation: P(a random diverse panel out-scores a random duplicate panel) — the ranking-of-candidates power
    sep = float(np.mean(d_div[:, None] > d_dup[None, :]))   # full REPS×REPS comparison
    return {"N": N, "dup_mean": float(d_dup.mean()), "div_mean": float(d_div.mean()),
            "dup_sd": float(d_dup.std()), "acc_dup_wasted": acc_dup, "acc_div_valuable": acc_div,
            "acc_balanced": (acc_dup + acc_div) / 2, "sep_div_gt_dup": sep}

rows = [run(N) for N in N_VALUES]
print(f"=== #34 deployability power: panel size N to PRICE a task via test-retest rank reliability "
      f"(T0={T0}, σs={SIG_S}, σe={SIG_E}, REPS={REPS} panels, dup_δ={DUP_DELTA}) ===")
print(f"{'N':>5} {'dupΔ':>8} {'divΔ':>8} {'panel_sd':>9} | {'acc_dup':>8} {'acc_div':>8} {'acc_bal':>8} {'sep':>6}")
for r in rows:
    print(f"{r['N']:>5} {r['dup_mean']:>+8.4f} {r['div_mean']:>+8.4f} {r['dup_sd']:>9.4f} | "
          f"{r['acc_dup_wasted']:>8.3f} {r['acc_div_valuable']:>8.3f} {r['acc_balanced']:>8.3f} {r['sep_div_gt_dup']:>6.3f}")

# find the smallest N reaching >=0.90 balanced accuracy and >=0.90 separation
n90_acc = next((r["N"] for r in rows if r["acc_balanced"] >= 0.90), None)
n90_sep = next((r["N"] for r in rows if r["sep_div_gt_dup"] >= 0.90), None)
print(f"--- smallest N (of those tried) with balanced single-task decision acc ≥0.90: {n90_acc}")
print(f"--- smallest N (of those tried) with rank-the-two-candidates separation ≥0.90: {n90_sep}")
print(f"--- panel-Δ sd at N=20: {rows[1]['dup_sd']:.4f}  (vs the ~0.011 signal → SNR {0.011/rows[1]['dup_sd']:.2f})")

# DEPLOYABILITY extrapolation from the measured sd ∝ 1/√N scaling (fit k in sd = k/√N over the rows)
import numpy as _np
ks = _np.array([r["dup_sd"] * _np.sqrt(r["N"]) for r in rows])
k_fit = float(ks.mean())                                  # sd ≈ k_fit / sqrt(N)
signal = 0.009                                            # the stable |dup_mean| ≈ |div_mean| separation from 0
sd_needed = signal / 1.28                                 # one-sided 90% call: |mean|/sd ≥ 1.28
N_oneshot = (k_fit / sd_needed) ** 2                      # agents needed in a single panel
N80 = next(r for r in rows if r["N"] == 80)               # or fix N=80 and average k replicates
reps_at_N80 = (N80["dup_sd"] / sd_needed) ** 2
print(f"--- sd ≈ {k_fit:.3f}/√N (fit). For a 90% single-task call (|mean|≈{signal} ≥1.28·sd → sd≤{sd_needed:.4f}):")
print(f"    EITHER one-shot N ≈ {int(N_oneshot)} agents,  OR  N=80 agents × ≈{int(round(reps_at_N80))} independent re-eval replicates")

score = {"cycle": 218, "item": "#34-power", "REPS": REPS, "T0": T0, "dup_delta": DUP_DELTA, "rows": rows,
         "n90_balanced_acc": n90_acc, "n90_separation": n90_sep,
         "sd_scaling_k": k_fit, "N_oneshot_90pct": int(N_oneshot), "reps_at_N80_90pct": int(round(reps_at_N80)),
         "verdict_P1_acc_rises_with_N": bool(rows[-1]["acc_balanced"] > rows[0]["acc_balanced"]),
         "verdict_P2_needs_moderate_N": bool(n90_acc is not None and n90_acc >= 40),
         "verdict_P3_dup_easier": bool(rows[2]["acc_dup_wasted"] > rows[2]["acc_div_valuable"])}
json.dump(score, open(f"{D}/score.json", "w"), indent=2)
print("--- wrote", f"{D}/score.json")
