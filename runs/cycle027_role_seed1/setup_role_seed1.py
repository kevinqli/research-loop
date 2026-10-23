"""Cycle 17 setup: test whether a claude -p agent supplies the pre/post-treatment causal ROLE that the
mechanical shrink-vs-grow rule lacks (cycle 16: it scored 0.05 on the mediator). Reuses cycle-16's generators
(seed 0). Builds a researcher_view with, per task: the SAME summary stats the mechanical rule sees (marginal &
conditioned effect + p), PLUS an operational data-collection context (temporal order of Z; NO textbook labels).
The agent answers TWICE: (1) DATA-ONLY (stats alone — same info as the mechanical rule), (2) WITH-ROLE (stats +
context). Contrast isolates whether the causal-role info is the missing ingredient and whether the LLM uses it.
numpy-only OLS. MPS-exploratory (toy synthetic, known truth)."""
import numpy as np, json, os
from math import erf, sqrt

SEED = 1
N = 2000
rng = np.random.default_rng(SEED)
D = "runs/cycle027_role_seed1/"
os.makedirs(D + "researcher_view", exist_ok=True)

def ols_t(X, y, j):
    XtXi = np.linalg.inv(X.T @ X); beta = XtXi @ (X.T @ y)
    resid = y - X @ beta; n, k = X.shape
    se = np.sqrt(np.diag((resid @ resid) / (n - k) * XtXi))
    t = beta[j] / se[j]; p = 2 * (1 - 0.5 * (1 + erf(abs(t) / sqrt(2))))
    return float(beta[j]), float(p)

def make(structure):
    one = np.ones(N)
    if structure == "confounder":
        Z = rng.normal(0, 1, N); T = (Z + rng.normal(0, 1, N) > 0).astype(float)
        Y = 0.0 * T + 1.5 * Z + rng.normal(0, 1, N)
        ctx = ("Z is a pre-treatment baseline characteristic measured BEFORE the intervention T was assigned. "
               "Units with higher Z were more likely to receive T. Z plausibly affects the outcome Y directly.")
        gt, should_condition = "NULL", True
    elif structure == "collider":
        T = (rng.normal(0, 1, N) > 0).astype(float); U = rng.normal(0, 1, N)
        Z = 1.5 * T + 1.5 * U + rng.normal(0, 1, N); Y = 0.0 * T + 1.5 * U + rng.normal(0, 1, N)
        ctx = ("Z is measured AFTER the intervention T. Z is influenced by T AND by a separate unmeasured factor; "
               "that same unmeasured factor also affects the outcome Y. Z itself has no causal effect on Y.")
        gt, should_condition = "NULL", False
    elif structure == "mediator":
        T = (rng.normal(0, 1, N) > 0).astype(float); Z = 2.0 * T + rng.normal(0, 1, N)
        Y = 0.0 * T + 1.5 * Z + rng.normal(0, 1, N)
        ctx = ("Z is measured AFTER the intervention T and is believed to lie ON the causal pathway from T to the "
               "outcome Y (T changes Z, and Z changes Y).")
        gt, should_condition = "REAL", False
    bm, pm = ols_t(np.column_stack([one, T]), Y, 1)
    bc, pc = ols_t(np.column_stack([one, T, Z]), Y, 1)
    return dict(marginal_effect=round(bm, 3), marginal_p=round(pm, 4),
                conditioned_effect=round(bc, 3), conditioned_p=round(pc, 4),
                shrinks=bool(abs(bc) < abs(bm)), n=N, context=ctx, gt=gt, should_condition=should_condition)

# shuffle task order so the agent can't pattern-match (deterministic on SEED)
structs = ["confounder", "collider", "mediator"]
order = list(rng.permutation(structs))
tasks_view, truth = {}, {}
for i, s in enumerate(order, 1):
    d = make(s); tid = f"q{i}"
    tasks_view[tid] = {k: d[k] for k in ("marginal_effect", "marginal_p", "conditioned_effect",
                                         "conditioned_p", "shrinks", "n", "context")}
    truth[tid] = {"structure": s, "gt": d["gt"], "should_condition": d["should_condition"]}

json.dump(tasks_view, open(D + "researcher_view/tasks.json", "w"), indent=1)
json.dump(truth, open(D + "ground_truth.json", "w"), indent=1)

prompt = """You are a careful research-verification agent. You are given 3 tasks (q1,q2,q3). For EACH task you must
decide whether a treatment T has a REAL causal effect on an outcome Y (verdict REAL or NULL).

For each task you are given, in `tasks.json`:
- `marginal_effect`/`marginal_p`: the estimated effect of T on Y from a regression Y ~ T (NOT adjusting for Z).
- `conditioned_effect`/`conditioned_p`: the estimated effect of T on Y from Y ~ T + Z (adjusting for a covariate Z).
- `shrinks`: whether adjusting for Z made |effect| smaller.
- `context`: a description of how Z was collected (its timing relative to T and its causal relationships).

Answer in TWO passes and write BOTH to `answers.json`:
PASS 1 — DATA_ONLY: using ONLY the numeric statistics (ignore `context`), give your best verdict per task and a
  confidence, and note if the statistics alone are insufficient to decide.
PASS 2 — WITH_ROLE: now ALSO use `context`. Decide whether to condition on Z (i.e., trust the conditioned estimate)
  or not (trust the marginal), then give your final verdict per task with a one-sentence causal justification.

Think about WHICH estimate (marginal vs conditioned) correctly answers "does T causally affect Y" given what Z is.

Write ONLY this JSON to a file named `answers.json` (no prose outside it):
{
 "data_only": {"q1": {"verdict":"REAL|NULL","note":"..."}, "q2": {...}, "q3": {...}},
 "with_role": {"q1": {"verdict":"REAL|NULL","condition_on_Z":true|false,"reason":"..."}, "q2": {...}, "q3": {...}}
}
"""
open(D + "researcher_view/PROMPT.md", "w").write(prompt)
print("wrote tasks.json, ground_truth.json, PROMPT.md")
print("order (hidden from agent):", order)
for tid, v in tasks_view.items():
    print(f"  {tid}: marg={v['marginal_effect']}(p{v['marginal_p']}) cond={v['conditioned_effect']}(p{v['conditioned_p']}) shrinks={v['shrinks']}")
print("DONE")
