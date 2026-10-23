"""Cycle 181 (COMPOSE — how many DIVERSE tasks does a verification benchmark need to rank agents? Lifts c48/51
[diversity-beats-count, at the CANDIDATE level] x c90 [M-rule] to the TASK-SET level; directly informs the benchmark.
Distinct from c107 (correlated noise ACROSS AGENTS, which cancels in ranking); here noise is correlated ACROSS TASKS
for the same agent = task REDUNDANCY, which does NOT cancel.

MODEL: N agents, true skill s_i ~ N(0,1). T tasks. score x_{i,t} = s_i + e_{i,t}; redundancy ρ = inter-task corr of
an agent's noise: e_{i,t} = sqrt(ρ)*u_i + sqrt(1-ρ)*v_{i,t} (u_i shared across tasks, v_{i,t} per-task; both N(0,σe)).
Benchmark ranks agents by mean_t x. Var(mean noise) = σe²·(1+(T-1)ρ)/T → the EFFECTIVE task count is
  T_eff = T / (1 + (T-1)ρ).   Ranking SNR = σ_skill²·T_eff/σe²  → reliability should be a function of T_eff ALONE.

PRE-REG: P1 reliability (Spearman of estimated vs true rank; P(top-1 correct)) RISES with T, governed by T_eff.
P2 at ρ>0 it SATURATES at a ceiling: T_eff → 1/ρ as T→∞, so a REDUNDANT benchmark caps out no matter how many tasks.
P3 diverse (ρ=0) BEATS redundant at every T (diversity-beats-count at the task level). P4 (clean test) reliability
COLLAPSES onto a single function of T_eff across all (T,ρ) — T_eff is the sufficient statistic.
"""
import numpy as np
rng = np.random.default_rng(0)
N, REPS, SIG_SKILL, SIG_E = 50, 3000, 1.0, 1.0
T_VALUES = [1, 2, 5, 10, 20, 50]
RHOS = [0.0, 0.3, 0.7]

def spearman(a, b):                       # rank-correlation over the N agents
    ra = a.argsort().argsort(); rb = b.argsort().argsort()
    return np.corrcoef(ra, rb)[0, 1]

def run(T, rho):
    sp, top1 = [], []
    for _ in range(REPS):
        s = rng.normal(0, SIG_SKILL, N)
        u = rng.normal(0, SIG_E, N)
        v = rng.normal(0, SIG_E, (N, T))
        noise = np.sqrt(rho) * u + np.sqrt(1 - rho) * v.mean(axis=1)
        score = s + noise
        sp.append(spearman(score, s))
        top1.append(int(np.argmax(score) == np.argmax(s)))
    return float(np.mean(sp)), float(np.mean(top1))

def t_eff(T, rho): return T / (1 + (T - 1) * rho)

out = {"N": N, "REPS": REPS, "rows": []}
print(f"=== Task diversity vs count for ranking {N} agents (sigma_skill={SIG_SKILL}, sigma_e={SIG_E}, REPS={REPS}) ===")
print(f"{'rho':>5} {'T':>4} {'T_eff':>7} {'spearman':>9} {'P(top1)':>8}")
for rho in RHOS:
    for T in T_VALUES:
        sp, p1 = run(T, rho); te = t_eff(T, rho)
        out["rows"].append({"rho": rho, "T": T, "T_eff": round(te, 2), "spearman": round(sp, 3), "p_top1": round(p1, 3)})
        print(f"{rho:>5} {T:>4} {te:>7.2f} {sp:>9.3f} {p1:>8.3f}")
    print(f"   -> rho={rho}: T_eff CEILING as T->inf = {('inf' if rho==0 else round(1/rho,2))}")

# P4 collapse test: group rows by ~equal T_eff across different (T,rho); reliability should match
print("\n--- P4 collapse: rows with similar T_eff (different T,rho) should have ~equal spearman ---")
rows = sorted(out["rows"], key=lambda r: r["T_eff"])
for r in rows:
    print(f"  T_eff={r['T_eff']:>6} (rho={r['rho']}, T={r['T']:>2})  spearman={r['spearman']:.3f}  P(top1)={r['p_top1']:.3f}")
import json, os; os.makedirs("runs/cycle181_task_diversity_count", exist_ok=True)
json.dump(out, open("runs/cycle181_task_diversity_count/score.json", "w"), indent=1)
print("\nPRE-REG: P1 reliability rises w/ T via T_eff; P2 ceiling 1/rho (redundant benchmark caps out); "
      "P3 diverse beats redundant at every T; P4 reliability collapses onto T_eff. Wrote score.json.")
