"""Cycle 18 (backlog 3b): the realistic MULTI-COVARIATE case. One dataset, treatment T, outcome Y, and THREE
covariates of DIFFERENT causal roles simultaneously: C=confounder (pre-T, must condition), M=mediator (post-T,
T->M->Y, must NOT condition), K=collider (post-T, T->K<-U, U->Y, must NOT condition). Question: estimate the
TRUE TOTAL causal effect of T on Y. Compare conditioning policies. numpy-only OLS, 100 seeds, across-seed CIs.
MPS-exploratory (toy synthetic, known truth).

Structural equations (true total effect tau = direct h + mediated b*f):
  C ~ N(0,1);  T = 1[C + e>0];  M = b*T + e;  U ~ N(0,1);  K = c*T + d*U + e;  Y = e_*C + f*M + g*U + h*T + e
Pre-reg: marginal biased by C; condition-on-ALL blocks the mediator (-> ~direct only) + collider bias;
shrink-vs-grow STILL includes M (it shrinks like a confounder) -> underestimates; role-correct (condition on
the pre-treatment C only) recovers tau. => no single DATA-driven global policy works; need per-covariate role."""
import numpy as np, json
from math import erf, sqrt

N = 4000; SEEDS = 100
b, f_, g, h, e_, c, d = 2.0, 1.5, 1.5, 0.5, 1.0, 1.5, 1.5
TAU = h + b * f_     # true total effect = 0.5 + 2*1.5 = 3.5

def ols_coef(X, y, j=1):
    XtXi = np.linalg.inv(X.T @ X); beta = XtXi @ (X.T @ y)
    resid = y - X @ beta; n, k = X.shape
    se = np.sqrt(np.diag((resid @ resid) / (n - k) * XtXi))
    t = beta[j] / se[j]; p = 2 * (1 - 0.5 * (1 + erf(abs(t) / sqrt(2))))
    return float(beta[j]), float(p)

def gen(rng):
    one = np.ones(N)
    C = rng.normal(0, 1, N)
    T = (C + rng.normal(0, 1, N) > 0).astype(float)
    M = b * T + rng.normal(0, 1, N)
    U = rng.normal(0, 1, N)
    K = c * T + d * U + rng.normal(0, 1, N)
    Y = e_ * C + f_ * M + g * U + h * T + rng.normal(0, 1, N)
    cols = {"C": C, "M": M, "K": K}
    def design(covs): return np.column_stack([one, T] + [cols[k] for k in covs])
    return Y, design

policies = {}
shrink_decisions = {"C": [], "M": [], "K": []}  # does adding this covariate ALONE shrink |bT| vs marginal?
ests = {"marginal": [], "condition_C(role)": [], "condition_all": [], "shrink_grow": []}

for seed in range(SEEDS):
    rng = np.random.default_rng(seed)
    Y, design = gen(rng)
    bT_marg, _ = ols_coef(design([]), Y)
    ests["marginal"].append(bT_marg)
    ests["condition_C(role)"].append(ols_coef(design(["C"]), Y)[0])
    ests["condition_all"].append(ols_coef(design(["C", "M", "K"]), Y)[0])
    # shrink-vs-grow per covariate: include covariate x iff adding x ALONE shrinks |bT| vs marginal
    include = []
    for x in ["C", "M", "K"]:
        bT_x, _ = ols_coef(design([x]), Y)
        shrinks = abs(bT_x) < abs(bT_marg)
        shrink_decisions[x].append(shrinks)
        if shrinks: include.append(x)
    ests["shrink_grow"].append(ols_coef(design(include), Y)[0])

def ci(x):
    x = np.asarray(x); return float(x.mean()), float(1.96 * x.std(ddof=1) / sqrt(len(x)))

print(f"N={N}/dataset, seeds={SEEDS}; TRUE total effect tau = {TAU}\n")
# correct decision = condition ONLY on pre-treatment covariates (C); M,K are post-treatment -> must NOT condition
SHOULD_CONDITION = {"C": True, "M": False, "K": False}
ROLE = {"C": "confounder/pre-T", "M": "mediator/post-T", "K": "collider/post-T"}
print("per-covariate shrink-vs-grow decision (mean over seeds):")
for x in ["C", "M", "K"]:
    includes = np.mean(shrink_decisions[x]) > 0.5
    correct = (includes == SHOULD_CONDITION[x])
    print(f"  {x}: conditioning-alone SHRINKS |bT| in {100*np.mean(shrink_decisions[x]):.0f}% of seeds "
          f"-> shrink-grow {'INCLUDES' if includes else 'EXCLUDES'} {x} [{ROLE[x]}] "
          f"-> {'CORRECT' if correct else 'WRONG'} (should condition={SHOULD_CONDITION[x]})")
print(f"\n{'policy':22s} {'est. total effect bT':>22s} {'|error vs tau|':>15s}")
out = {"tau": TAU, "N": N, "seeds": SEEDS, "policies": {}, "shrink_decisions": {x: float(np.mean(shrink_decisions[x])) for x in shrink_decisions}}
for name, vals in ests.items():
    m, s = ci(vals); err = abs(m - TAU)
    print(f"{name:22s} {m:>10.3f} ± {s:.3f}      {err:>10.3f}")
    out["policies"][name] = {"bT": m, "ci": s, "abs_err": err}
json.dump(out, open("runs/cycle018_multicovariate/score.json", "w"), indent=1)
print("\n--- read the rows above, not this line ---")
c_err = out['policies']['condition_C(role)']['abs_err']; sg_err = out['policies']['shrink_grow']['abs_err']
print(f"role-correct (condition on pre-treatment C only) err={c_err:.3f}; shrink-grow err={sg_err:.3f}; "
      f"shrink-grow worse by {sg_err - ce if (ce:=c_err) else 0:.3f} (it conditions on the mediator).")
print("wrote runs/cycle018_multicovariate/score.json ; DONE")
