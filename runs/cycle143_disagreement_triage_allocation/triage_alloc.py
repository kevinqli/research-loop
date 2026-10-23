"""Cycle 143 (COMPOSE c133×c97×c45; BACKLOG #18): does triaging a VERIFICATION budget by inter-agent DISAGREEMENT help —
and what does it systematically MISS? c133 showed disagreement is a DIFFICULTY signal, not a reliability signal. Sharp
consequence: independent errors AVERAGE DOWN (high disagreement, but consensus-error ≈ σ/√N — moderate) while a SHARED
blind-spot error PERSISTS (low disagreement, consensus-error ≈ |bias| — high). So verifying HIGH-disagreement targets the
LOWER-error tasks and MISSES the highest-error (shared-bias) ones — disagreement-triage MIS-PRIORITIZES.

Model: M tasks, θ=0, N=8 agents. Fraction f_shared are SHARED-BIAS (shared b~N(0,1.5) + small indep noise 0.2 → low
disagreement, high consensus-error); the rest INDEPENDENT-difficulty (σ_t~U[0.5,2] → high disagreement, consensus-error
σ_t/√N). Verify a budget B=20% of tasks; CAUGHT-ERROR = Σ consensus-error over verified. Triage: DISAGREEMENT (top-B by
std) vs UNIFORM (random) vs ORACLE (top-B by true consensus-error). Sweep f_shared. numpy, REPS reps.

PRE-REG: P1 f_shared=0 → disagreement≈oracle≫uniform (disagreement tracks σ_t). P2 as f_shared↑, disagreement-triage FALLS
(misses the shared-bias high-error tasks) toward/below uniform; oracle stays high → gap-to-oracle blows up. P3 disagreement-
triage catches FEW shared-bias tasks (they're low-disagreement)."""
import numpy as np, json, os
RNG = np.random.default_rng(0)
M, N, REPS = 200, 8, 4000
BUDGET = int(M*0.2)
os.makedirs("runs/cycle143_disagreement_triage_allocation", exist_ok=True)

def one_rep(f_shared, rng):
    n_shared = int(round(M*f_shared))
    is_shared = np.zeros(M, bool); is_shared[:n_shared] = True
    # independent tasks
    sig = rng.uniform(0.5, 2.0, M)
    est = rng.normal(0, 1, (M, N)) * sig[:, None]                  # independent agent errors
    # shared-bias tasks: replace with shared bias + small indep noise
    b = rng.normal(0, 1.5, M)
    est_shared = b[:, None] + rng.normal(0, 0.2, (M, N))
    est = np.where(is_shared[:, None], est_shared, est)
    disagree = est.std(1, ddof=1)
    cons_err = np.abs(est.mean(1))
    # triage: verify top-BUDGET by signal; caught = sum of cons_err over verified
    def caught(order): return cons_err[order[:BUDGET]].sum()
    dis_order = np.argsort(-disagree); uni_order = rng.permutation(M); ora_order = np.argsort(-cons_err)
    # shared-bias tasks caught by disagreement-triage
    shared_caught_dis = is_shared[dis_order[:BUDGET]].sum()
    return caught(dis_order), caught(uni_order), caught(ora_order), shared_caught_dis, n_shared

print(f"M={M} tasks, N={N} agents, verify budget B={BUDGET} (20%). caught-error = Σ consensus-error over verified tasks.\n")
print(f"{'f_shared':>9} | {'DISAGREE':>9} {'UNIFORM':>9} {'ORACLE':>9} | {'disagree/oracle':>15} | {'shared-tasks caught by disagree':>31}")
rows=[]
for f in [0.0, 0.1, 0.2, 0.4]:
    res = np.array([one_rep(f, np.random.default_rng(1000+i)) for i in range(REPS)])
    dis, uni, ora = res[:,0].mean(), res[:,1].mean(), res[:,2].mean()
    shared_caught, n_shared = res[:,3].mean(), int(res[0,4])
    frac = f"{shared_caught:.1f}/{n_shared}" if n_shared>0 else "n/a"
    rows.append({"f_shared":f, "disagree":round(float(dis),2), "uniform":round(float(uni),2), "oracle":round(float(ora),2),
                 "disagree_over_oracle":round(float(dis/ora),2), "shared_caught_by_disagree":frac, "n_shared":n_shared})
    print(f"{f:>9} | {dis:>9.2f} {uni:>9.2f} {ora:>9.2f} | {dis/ora:>15.2f} | {frac:>31}")

json.dump({"M":M,"N":N,"BUDGET":BUDGET,"vs_f_shared":rows}, open("runs/cycle143_disagreement_triage_allocation/score.json","w"), indent=1)
print("\n--- READ THE RAW ROWS ---")
print("caught-error DISAGREE: " + " ".join(f"f{r['f_shared']}:{r['disagree']}" for r in rows))
print("caught-error UNIFORM:  " + " ".join(f"f{r['f_shared']}:{r['uniform']}" for r in rows))
print("caught-error ORACLE:   " + " ".join(f"f{r['f_shared']}:{r['oracle']}" for r in rows))
print("disagree/oracle ratio: " + " ".join(f"f{r['f_shared']}:{r['disagree_over_oracle']}" for r in rows) + "  (falls as shared-bias rises)")
print("shared-bias tasks caught by disagree-triage: " + " ".join(f"f{r['f_shared']}:{r['shared_caught_by_disagree']}" for r in rows))
print("=> disagreement-triage MIS-PRIORITIZES: it verifies high-disagreement (independent, averaged-DOWN) tasks and MISSES the low-disagreement-HIGH-error shared-bias tasks. At f_shared>0 it falls toward/below UNIFORM and far below ORACLE. So don't triage verification by disagreement alone (it's blind to the worst, shared-blind-spot failures); reserve budget for high-stakes / use an independent held-out (c45) for the systematic risk.")
print("DONE")
