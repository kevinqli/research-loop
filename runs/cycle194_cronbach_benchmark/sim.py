"""Cycle 194 (#35 — does CLASSICAL TEST THEORY reliability (Cronbach's α) of a verification benchmark PREDICT its
agent-RANKING reliability? diversify MODE off the closed quant-moat thread; benchmark = the product; meth-28 extends
c181's exact model). c183 grounded the benchmark as a psychometric instrument; α is the standard no-ground-truth
reliability estimate. KEY question for a benchmark builder: can you compute α (from the observable agent×task score
matrix, NO true skill needed) to estimate how reliably the benchmark ranks agents?

MODEL (c181 exact): N agents, true skill s_i~N(0,σs). T tasks. x_{i,t} = s_i + sqrt(ρ)·u_i + sqrt(1-ρ)·v_{i,t}
(u_i shared-across-tasks NOISE, v per-task; both N(0,σe)). ρ = task NOISE-redundancy. Ranking by mean_t x.
True ranking reliability R_true = σs²·T_eff/(σs²·T_eff + σe²), T_eff = T/(1+(T-1)ρ) (c181).

THE CATCH (analytic): Cronbach's α treats ALL inter-task covariance as true-score. Here Cov(x_t,x_t') = σs² + ρσe²
(shared SKILL + shared NOISE). So α counts the shared NOISE ρσe² as if it were signal → α estimates the reliability
of the sum for the COMMON FACTOR (skill + shared-noise), which OVERESTIMATES the true-SKILL ranking reliability.

PRE-REG: (P1) at ρ=0 (diverse tasks, shared component = skill only) α ≈ R_true (α is a valid estimator). (P2) at
ρ>0 (NOISE-redundant tasks) α OVERESTIMATES R_true — α > R_true, gap GROWS with ρ. (P3) the ranking reliability
(Spearman) tracks R_true (not α) — so a builder trusting α under noise-redundancy gets FALSE confidence. Actionable:
α is valid only if inter-task correlation is SKILL-driven; under shared task noise use test-retest/held-out instead.
"""
import numpy as np, json, os
D = "runs/cycle194_cronbach_benchmark"; os.makedirs(D, exist_ok=True)
rng = np.random.default_rng(0)
N, REPS, SIG_S, SIG_E = 60, 4000, 1.0, 1.0
T_VALUES = [5, 10, 20]
RHOS = [0.0, 0.3, 0.6]

def spearman(a, b):
    ra = a.argsort().argsort(); rb = b.argsort().argsort()
    return np.corrcoef(ra, rb)[0, 1]

def cronbach_alpha(X):                      # X: (N agents, T tasks)
    T = X.shape[1]
    item_var = X.var(axis=0, ddof=1).sum()  # Σ_t Var(task t over agents)
    total_var = X.sum(axis=1).var(ddof=1)   # Var(per-agent total)
    return (T / (T - 1)) * (1 - item_var / total_var)

def t_eff(T, rho): return T / (1 + (T - 1) * rho)

def run(T, rho):
    alphas, rtrue_emp, sp = [], [], []
    for _ in range(REPS):
        s = rng.normal(0, SIG_S, N)
        u = rng.normal(0, SIG_E, N)
        v = rng.normal(0, SIG_E, (N, T))
        X = s[:, None] + np.sqrt(rho) * u[:, None] + np.sqrt(1 - rho) * v   # (N,T)
        mean_score = X.mean(axis=1)
        alphas.append(cronbach_alpha(X))
        rtrue_emp.append(np.corrcoef(mean_score, s)[0, 1] ** 2)            # empirical reliability (needs true skill)
        sp.append(spearman(mean_score, s))
    te = t_eff(T, rho)
    r_analytic = (SIG_S**2 * te) / (SIG_S**2 * te + SIG_E**2)
    return {"T": T, "rho": rho, "T_eff": round(te, 2), "alpha": float(np.mean(alphas)),
            "R_true_emp": float(np.mean(rtrue_emp)), "R_true_analytic": round(r_analytic, 3),
            "spearman": float(np.mean(sp))}

print(f"=== #35 Cronbach's α vs true agent-ranking reliability (N={N}, σs={SIG_S}, σe={SIG_E}, REPS={REPS}) ===")
print(f"{'rho':>5} {'T':>4} {'T_eff':>6} {'alpha':>7} {'R_true':>7} {'α−R':>7} {'spearman':>9}")
rows = []
for rho in RHOS:
    for T in T_VALUES:
        r = run(T, rho); r["alpha_minus_Rtrue"] = round(r["alpha"] - r["R_true_emp"], 3)
        rows.append(r)
        print(f"{rho:>5} {T:>4} {r['T_eff']:>6.2f} {r['alpha']:>7.3f} {r['R_true_emp']:>7.3f} "
              f"{r['alpha_minus_Rtrue']:>+7.3f} {r['spearman']:>9.3f}")
    print()

# verdicts
rho0 = [r for r in rows if r["rho"] == 0.0]
rhopos = [r for r in rows if r["rho"] > 0.0]
p1 = max(abs(r["alpha_minus_Rtrue"]) for r in rho0) < 0.03
p2 = all(r["alpha_minus_Rtrue"] > 0.02 for r in rhopos)
gap_grows = (np.mean([r["alpha_minus_Rtrue"] for r in rows if r["rho"] == 0.6]) >
             np.mean([r["alpha_minus_Rtrue"] for r in rows if r["rho"] == 0.3]))
print(f"P1 (ρ=0: α ≈ R_true, |gap|<0.03)? {p1}")
print(f"P2 (ρ>0: α OVERESTIMATES, α−R_true>0.02 every cell)? {p2}")
print(f"P3 (gap GROWS with ρ: mean gap ρ=.6 > ρ=.3)? {gap_grows}")
print(f"--> {'CONFIRMED' if (p1 and p2 and gap_grows) else 'CHECK'}: Cronbach's α is valid ONLY when inter-task corr is SKILL-driven; "
      f"under task NOISE-redundancy it OVERSTATES ranking reliability (counts shared noise as signal) -> false confidence; use test-retest/held-out instead.")
json.dump({"params": {"N": N, "REPS": REPS, "SIG_S": SIG_S, "SIG_E": SIG_E}, "rows": rows,
           "P1_alpha_eq_Rtrue_rho0": bool(p1), "P2_alpha_overstates_rhopos": bool(p2), "P3_gap_grows_with_rho": bool(gap_grows)},
          open(f"{D}/score.json", "w"), indent=1)
print("DONE")
