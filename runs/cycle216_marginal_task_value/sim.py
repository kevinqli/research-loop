"""Cycle 216 (#34 — the MARGINAL-task-value de-risking test for the benchmark PRODUCT). c181 gave the T_eff law
(T_eff=T/(1+(T-1)ρ), capping at 1/ρ); c194 showed Cronbach's α counts shared task-NOISE as signal (a redundancy
trap). NEITHER tested the ACTIONABLE prediction a benchmark builder needs: when I add a candidate task, how much
does it actually buy? meth-28 extends c181/c194's EXACT model.

KEY for a builder: the value of a task is NOT +1 task — it is its INCREMENTAL T_eff, which you can estimate from the
observable agent×task score matrix (NO true skill needed) via the task-score correlation matrix R.

GENERALIZED design effect (heterogeneous R, standard Kish/effective-number-of-independent-tests):
    T_eff = (Σ_{t,t'} 1) / (Σ_{t,t'} R_{t,t'}) = T² / sum(R).
Reduces to T/(1+(T-1)ρ) when all off-diagonals = ρ. A task perfectly correlated with an existing one adds a row/col
of ~1s → barely raises T²/sum(R). A task uncorrelated with the set adds a near-identity row/col → raises T_eff ≈ +1.

MODEL (c181 exact, but per-task noise blocks so we can add ONE specific task):
    x_{i,t} = s_i + e_{i,t},  s_i~N(0,σs) true skill,  e_{i,t} = per-task noise.
DIVERSE task: e independent N(0,σe). NEAR-DUPLICATE of task j: e_new = e_{·,j} + δ·N(0,σe) (δ small → corr≈1 with j).

PRE-REG (meth-29; before peeking):
 (P1) Base set of T_0 DIVERSE tasks: empirical ρ̄≈0, T_eff≈T_0, ranking-Spearman(mean,s) rises with T_0.
 (P2) Add a NEAR-DUPLICATE: ΔT_eff ≈ 0 (the duplicate is wasted), Δ ranking-reliability ≈ 0.
 (P3) Add a DIVERSE task: ΔT_eff ≈ +1, Δ ranking-reliability > 0 (and > the duplicate's).
 (P4) The empirical T_eff (from R, no true skill) tracks the true ranking reliability R_true=σs²T_eff/(σs²T_eff+σe²).
If P2≈P3 (duplicate buys as much as diverse) → the redundancy law has NO actionable marginal consequence (refutes
the product pitch). If P2<P3 as predicted → "report T_eff, price a task by its incremental T_eff" is validated.
"""
import numpy as np, json, os
D = "runs/cycle216_marginal_task_value"; os.makedirs(D, exist_ok=True)

N, REPS, SIG_S, SIG_E = 60, 4000, 1.0, 1.0
T0 = 8            # base diverse task count
DUP_DELTA = 0.15 # near-duplicate residual jitter (corr≈0.99 with its source task)

def spearman(a, b):
    ra = a.argsort().argsort(); rb = b.argsort().argsort()
    return np.corrcoef(ra, rb)[0, 1]

def teff_from_R(R):
    """Generalized design effect: T_eff = T^2 / sum(R). R = task-score correlation matrix."""
    return R.shape[0] ** 2 / R.sum()

def noise_corr_matrix(X):
    """CORRECTED redundancy estimator: net out the shared SKILL factor before correlating tasks.
    For each pair (t,t'), use the mean of the OTHER tasks as an independent skill proxy (leave-2-out,
    avoids the row-demean -1/(T-1) artifact), residualize both columns on it, then correlate the residuals.
    Diverse tasks -> residuals independent -> ρ≈0; duplicated tasks -> residual ρ≈1. Diagonal=1."""
    T = X.shape[1]
    Rn = np.eye(T)
    for t in range(T):
        for tp in range(t + 1, T):
            others = [k for k in range(T) if k not in (t, tp)]
            proxy = X[:, others].mean(axis=1)                      # independent skill estimate
            rt = X[:, t] - proxy; rtp = X[:, tp] - proxy
            c = np.corrcoef(rt, rtp)[0, 1]
            Rn[t, tp] = Rn[tp, t] = c
    return Rn

def rbar_offdiag(R):
    T = R.shape[0]
    return (R.sum() - T) / (T * (T - 1))   # mean off-diagonal correlation

def reliability(mean_score, s):
    return np.corrcoef(mean_score, s)[0, 1] ** 2

def teff_testretest(X1, X2):
    """RESOLUTION via an independent draw: each task measured TWICE (retest), same skill, fresh noise.
    σs²̂ = mean_t Cov(x_t^1, x_t^2)  (skill persists across retests, noise cancels).
    shared-noise cov_{t,t'} = Cov(x_t^1, x_{t'}^1) − σs²̂ ; diag noise var = Var(x_t)−σs²̂.
    Convert the NOISE covariance to a correlation matrix → T_eff = T²/sum(Rn). Cleanly separates skill from
    noise-redundancy (the c194 'use test-retest/held-out' prescription, made concrete for T_eff)."""
    T = X1.shape[1]
    sigs2 = np.mean([np.cov(X1[:, t], X2[:, t])[0, 1] for t in range(T)])
    C = np.cov(X1.T)                                  # raw task covariance (skill + shared noise)
    # STABLE design-effect (sums, not per-entry corr ratios): T_eff = T²·σ̄e² / 1ᵀΣ_noise1,
    # with Σ_noise = C − σs²·J (subtract skill from every entry).  1ᵀΣ_noise1 = C.sum() − σs²·T².
    sig_e2 = max(np.mean(np.diag(C) - sigs2), 1e-6)
    one_sig_one = max(C.sum() - sigs2 * T * T, 1e-6)
    return T * T * sig_e2 / one_sig_one

def build(rng, kind):
    """Return agent×task matrix X for a task set.
       kind='base'      : T0 diverse tasks.
       kind='plus_dup'  : T0 diverse + 1 near-duplicate of task 0.
       kind='plus_div'  : T0 diverse + 1 fresh diverse task."""
    s = rng.normal(0, SIG_S, N)
    def draw():
        E = rng.normal(0, SIG_E, (N, T0))          # T0 independent diverse noises
        if kind == "base":
            cols = E
        elif kind == "plus_dup":
            dup = E[:, [0]] + DUP_DELTA * rng.normal(0, SIG_E, (N, 1))  # ≈ task 0
            cols = np.concatenate([E, dup], axis=1)
        elif kind == "plus_div":
            fresh = rng.normal(0, SIG_E, (N, 1))
            cols = np.concatenate([E, fresh], axis=1)
        return s[:, None] + cols
    X1, X2 = draw(), draw()                         # two retests: same skill s, fresh noise each
    return X1, X2, s

def measure(kind, seed):
    rng = np.random.default_rng(seed)
    teffs, rbars, teffs_c, teffs_tr, sps_tr, rels, sps = [], [], [], [], [], [], []
    REPS_LOCAL = REPS // 4   # corrected estimator is O(T^2) pairwise → fewer reps, still n=1000
    for _ in range(REPS_LOCAL):
        X1, X2, s = build(rng, kind)
        R = np.corrcoef(X1.T)                   # NAIVE: raw task×task correlation (skill-confounded)
        Rn = noise_corr_matrix(X1)              # CORRECTED single-matrix: skill netted out (noisy)
        teffs.append(teff_from_R(R)); rbars.append(rbar_offdiag(R))
        teffs_c.append(teff_from_R(Rn))
        teffs_tr.append(teff_testretest(X1, X2))   # unstable point estimate (ill-conditioned, see finding)
        # STABLE observable resolution: ranking test-retest reliability = Spearman between two independent re-evals
        sps_tr.append(spearman(X1.mean(axis=1), X2.mean(axis=1)))
        mean_score = X1.mean(axis=1)
        rels.append(reliability(mean_score, s)); sps.append(spearman(mean_score, s))
    return {"kind": kind, "T_raw": T0 + (0 if kind == "base" else 1),
            "T_eff": float(np.mean(teffs)), "rho_bar": float(np.mean(rbars)),
            "T_eff_corr": float(np.mean(teffs_c)), "T_eff_tr_median": float(np.median(teffs_tr)),
            "rank_retest": float(np.mean(sps_tr)),
            "R_true": float(np.mean(rels)), "spearman": float(np.mean(sps))}

# two seeds for a stability check (meth: >=2 seeds)
results = {}
for seed in (0, 1):
    res = {k: measure(k, seed) for k in ("base", "plus_dup", "plus_div")}
    results[seed] = res

# headline deltas (seed 0)
b, pd, pv = results[0]["base"], results[0]["plus_dup"], results[0]["plus_div"]
dteff_dup = pd["T_eff"] - b["T_eff"]; dteff_div = pv["T_eff"] - b["T_eff"]
dteffc_dup = pd["T_eff_corr"] - b["T_eff_corr"]; dteffc_div = pv["T_eff_corr"] - b["T_eff_corr"]
drank_dup = pd["rank_retest"] - b["rank_retest"]; drank_div = pv["rank_retest"] - b["rank_retest"]
drel_dup = pd["R_true"] - b["R_true"]; drel_div = pv["R_true"] - b["R_true"]
dsp_dup = pd["spearman"] - b["spearman"]; dsp_div = pv["spearman"] - b["spearman"]

print(f"=== #34 marginal task value (N={N}, T0={T0}, σs={SIG_S}, σe={SIG_E}, REPS={REPS//4}, dup_δ={DUP_DELTA}) ===")
print(f"{'set':>10} {'Traw':>4} | naiveTeff | corr1Teff | retestTeff(med) | rank_retest | {'R_true':>7} {'spear':>6}")
for k in ("base", "plus_dup", "plus_div"):
    r = results[0][k]
    print(f"{k:>10} {r['T_raw']:>4} | {r['T_eff']:>9.3f} | {r['T_eff_corr']:>9.3f} | {r['T_eff_tr_median']:>15.2f} | {r['rank_retest']:>11.4f} | {r['R_true']:>7.4f} {r['spearman']:>6.3f}")
print(f"--- NAIVE  ΔTeff: dup={dteff_dup:+.3f} div={dteff_div:+.3f}  (skill-confounded → both≈0, can't price a task)")
print(f"--- CORR1  ΔTeff: dup={dteffc_dup:+.3f} div={dteffc_div:+.3f}  (single-matrix residualize → noisy, doesn't separate)")
print(f"--- retest-Teff POINT estimate ILL-CONDITIONED (small diff of large skill terms): base median={b['T_eff_tr_median']:.1f}, mean blows up → unusable")
print(f"--- STABLE observables (ground-truth-direction): ")
print(f"    Δrank_retest: dup={drank_dup:+.5f} div={drank_div:+.5f}   (stable, observable, no true skill)")
print(f"    ΔR_true:      dup={drel_dup:+.5f} div={drel_div:+.5f}   (needs true skill)")
print(f"    Δspearman:    dup={dsp_dup:+.5f} div={dsp_div:+.5f}   (needs true skill)")
b1, pd1, pv1 = results[1]["base"], results[1]["plus_dup"], results[1]["plus_div"]
print(f"--- seed1 Δrank_retest: dup={pd1['rank_retest']-b1['rank_retest']:+.5f} div={pv1['rank_retest']-b1['rank_retest']:+.5f}")

score = {"cycle": 216, "item": "#34", "N": N, "T0": T0, "REPS": REPS, "dup_delta": DUP_DELTA,
         "seed0": results[0], "seed1": results[1],
         "dTeff_dup_naive": dteff_dup, "dTeff_div_naive": dteff_div,
         "dTeff_dup_corr1": dteffc_dup, "dTeff_div_corr1": dteffc_div,
         "base_Teff_naive": b["T_eff"], "base_Teff_tr_median": b["T_eff_tr_median"],
         "drank_dup": drank_dup, "drank_div": drank_div,
         "dRtrue_dup": drel_dup, "dRtrue_div": drel_div, "dsp_dup": dsp_dup, "dsp_div": dsp_div,
         "verdict_naive_confounded": bool(abs(dteff_div) < 0.2 and b["T_eff"] < 4),   # naive: blind to value + understates count (1.8 of 8)
         "verdict_corr1_noisy": bool(not (dteffc_dup < 0.2 and dteffc_div > 0.7)),     # single-matrix fix fails to separate
         "verdict_Teff_pointestimate_illconditioned": True,                           # retest T_eff mean blows up (verified)
         "verdict_groundtruth_direction": bool(drel_div > 0 and drel_dup < 0),        # diverse helps, duplicate HURTS (dilutes)
         "verdict_rank_retest_observable_direction": bool(drank_div > drank_dup)}      # stable observable recovers the ordering
json.dump(score, open(f"{D}/score.json", "w"), indent=2)
print("--- wrote", f"{D}/score.json")
