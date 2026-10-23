"""Cycle 16: does the collider-aware SHRINK-vs-GROW heuristic (cycle 3's winner) GENERALIZE beyond
confounder-vs-collider? Adversarial self-test on a MEDIATOR. numpy-only OLS, 100 seeds, across-seed CIs.
MPS-exploratory (toy synthetic, known ground truth).

3 causal structures, same research question each: "does T have a REAL total causal effect on Y?"
  CONFOUNDER : Z->T, Z->Y, no direct T->Y.   truth=NULL (marginal is spurious; conditioning on Z SHRINKS it -> recover NULL)
  COLLIDER   : T->Z, U->Z, U->Y, no T->Y.     truth=NULL (marginal ~0; conditioning on Z GROWS a spurious effect)
  MEDIATOR   : T->Z->Y, no direct T->Y.        truth=REAL (total effect flows through Z; conditioning on Z SHRINKS it to ~0)

Rules compared:
  A   no-check (marginal coef on T, significance)        -> conf WRONG, coll right, med right
  Bc  shrink-vs-grow (condition iff conditioning shrinks) -> conf right, coll right, MED ??? (pre-reg: WRONG)
  Bpp pre/post gate (condition iff Z is PRE-treatment)    -> all right, but needs Z's causal ROLE (not in the data signature)
Pre-reg: Bc fails on the mediator (shrinks like a confounder -> conditions -> calls the REAL effect NULL),
showing the shrink/grow DATA signature is insufficient; the correct discriminator is pre/post-treatment (causal role)."""
import numpy as np
from math import erf, sqrt

N = 2000      # samples per dataset
SEEDS = 100   # across-seed CIs
ALPHA = 0.05  # significance for REAL/NULL verdict

def ols_t_coef(X, y, j):
    """OLS; return (coef_j, two-sided p of coef_j) via normal approx (N large)."""
    XtX = X.T @ X
    XtXi = np.linalg.inv(XtX)
    beta = XtXi @ (X.T @ y)
    resid = y - X @ beta
    n, k = X.shape
    sigma2 = (resid @ resid) / (n - k)
    se = np.sqrt(np.diag(sigma2 * XtXi))
    t = beta[j] / se[j]
    p = 2 * (1 - 0.5 * (1 + erf(abs(t) / sqrt(2))))
    return beta[j], p

def gen(structure, rng):
    one = np.ones(N)
    if structure == "confounder":
        Z = rng.normal(0, 1, N)
        T = (Z + rng.normal(0, 1, N) > 0).astype(float)   # Z drives T (pre-treatment common cause)
        Y = 0.0 * T + 1.5 * Z + rng.normal(0, 1, N)        # NO direct T->Y ; Z->Y -> truth NULL
        truth = "NULL"
    elif structure == "collider":
        T = (rng.normal(0, 1, N) > 0).astype(float)
        U = rng.normal(0, 1, N)
        Z = 1.5 * T + 1.5 * U + rng.normal(0, 1, N)        # common EFFECT of T and U (post-treatment)
        Y = 0.0 * T + 1.5 * U + rng.normal(0, 1, N)        # NO direct T->Y -> truth NULL
        truth = "NULL"
    elif structure == "mediator":
        T = (rng.normal(0, 1, N) > 0).astype(float)
        Z = 2.0 * T + rng.normal(0, 1, N)                  # T->Z (post-treatment, on the path)
        Y = 0.0 * T + 1.5 * Z + rng.normal(0, 1, N)        # Z->Y, no direct T->Y -> TOTAL T->Y is REAL via Z
        truth = "REAL"
    Xm = np.column_stack([one, T])          # marginal: Y ~ 1 + T
    Xc = np.column_stack([one, T, Z])       # conditioned: Y ~ 1 + T + Z
    bm, pm = ols_t_coef(Xm, Y, 1)
    bc, pc = ols_t_coef(Xc, Y, 1)
    return truth, bm, pm, bc, pc

# Z's causal role (KNOWN ground truth) for the pre/post gate — NOT derivable from shrink/grow:
PRE_TREATMENT = {"confounder": True, "collider": False, "mediator": False}

structures = ["confounder", "collider", "mediator"]
agg = {s: {"bm": [], "bc": [], "shrinks": [], "A": [], "Bc": [], "Bpp": [], "truth": None} for s in structures}

for seed in range(100, 100+SEEDS):
    rng = np.random.default_rng(seed)
    for s in structures:
        truth, bm, pm, bc, pc = gen(s, rng)
        agg[s]["truth"] = truth
        agg[s]["bm"].append(bm); agg[s]["bc"].append(bc)
        shrinks = abs(bc) < abs(bm)
        agg[s]["shrinks"].append(shrinks)
        # A no-check: verdict by marginal significance
        agg[s]["A"].append("REAL" if pm < ALPHA else "NULL")
        # Bc shrink-vs-grow: if conditioning shrinks -> trust conditioned; if grows -> trust marginal
        if shrinks: chosen_p = pc
        else:       chosen_p = pm
        agg[s]["Bc"].append("REAL" if chosen_p < ALPHA else "NULL")
        # Bpp pre/post gate: condition ONLY if Z is pre-treatment (uses causal role, not data signature)
        chosen_p_pp = pc if PRE_TREATMENT[s] else pm
        agg[s]["Bpp"].append("REAL" if chosen_p_pp < ALPHA else "NULL")

def ci(x):
    x = np.asarray(x, float); m = x.mean(); s = 1.96 * x.std(ddof=1) / sqrt(len(x))
    return m, s
def acc(verdicts, truth):
    return np.mean([v == truth for v in verdicts])

print(f"N={N}/dataset, seeds={SEEDS}, alpha={ALPHA}\n")
print(f"{'structure':11s} {'truth':5s} | {'marginal bT':>16s} {'cond bT':>16s} {'shrinks%':>9s} | "
      f"{'A acc':>7s} {'Bc acc':>7s} {'Bpp acc':>8s}")
rule_acc = {"A": [], "Bc": [], "Bpp": []}
import json
out = {"N": N, "seeds": SEEDS, "structures": {}}
for s in structures:
    t = agg[s]["truth"]
    mm, ms = ci(agg[s]["bm"]); cm, cs = ci(agg[s]["bc"])
    shr = 100 * np.mean(agg[s]["shrinks"])
    aA = acc(agg[s]["A"], t); aBc = acc(agg[s]["Bc"], t); aBpp = acc(agg[s]["Bpp"], t)
    rule_acc["A"].append(aA); rule_acc["Bc"].append(aBc); rule_acc["Bpp"].append(aBpp)
    print(f"{s:11s} {t:5s} | {mm:>7.3f}±{ms:.3f}   {cm:>7.3f}±{cs:.3f}   {shr:>7.0f}%  | "
          f"{aA:>7.2f} {aBc:>7.2f} {aBpp:>8.2f}")
    out["structures"][s] = {"truth": t, "marginal_bT": [mm, ms], "cond_bT": [cm, cs],
                            "shrinks_pct": shr, "acc_A": aA, "acc_Bc": aBc, "acc_Bpp": aBpp}
print(f"\n{'MEAN ACC':11s}       |                                            | "
      f"{np.mean(rule_acc['A']):>7.2f} {np.mean(rule_acc['Bc']):>7.2f} {np.mean(rule_acc['Bpp']):>8.2f}")
out["mean_acc"] = {k: float(np.mean(v)) for k, v in rule_acc.items()}
json.dump(out, open("runs/cycle019_survival/med_score.json", "w"), indent=1)

print("\n--- INTERPRETATION (read the raw rows above, not this line) ---")
med = agg["mediator"]
print(f"mediator: conditioning SHRINKS the effect in {100*np.mean(med['shrinks']):.0f}% of seeds "
      f"(marginal {ci(med['bm'])[0]:.2f} -> cond {ci(med['bc'])[0]:.2f}); "
      f"Bc therefore CONDITIONS and reports {acc(med['Bc'],'REAL')*100:.0f}% REAL (truth=REAL).")
print("KEY: shrink-vs-grow can't tell confounder from mediator (both shrink); need the pre/post-treatment role.")
print("wrote runs/cycle019_survival/med_score.json ; DONE")
