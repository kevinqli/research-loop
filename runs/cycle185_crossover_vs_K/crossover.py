"""Cycle 185 (#32 — does the ensemble-vs-select crossover sigma_v* shift with FLEET SIZE K?). Extends c175 (tail-grab)
and grounds c176 (forecast-combination). KEY DISAMBIGUATION found while reading c175: the backlog #32 pre-reg assumes a
sigma_v* at which ENSEMBLE overtakes SELECT in realized quality. But that crossover DEPENDS ON THE GENERATIVE MODEL:

  MODEL A — TAIL-GRAB (c175 exact): K candidates, true q~N(mu,sig_q); pick argmax of a noisy verifier; payoff = true q
    of the picked one. ENSEMBLE = mean true q = mu (regresses to mean). SELECT_true ranges mu+sig_q*sqrt(2 ln K) (perfect
    verifier) DOWN TO mu (random verifier) — i.e. SELECT_true >= mu = ENSEMBLE at ALL sigma_v. PREDICTION: NO true-quality
    crossover under a POSITIVE verifier, at ANY K. What grows with K is the CURSE (reported-minus-true), not true quality.

  MODEL B — ESTIMATOR-COMBINATION (c176 forecast-combination): K candidates are noisy ESTIMATORS of a common target;
    loss = squared error. ENSEMBLE = average prediction -> loss ~ sig^2/K (variance reduction, verifier-independent).
    SELECT = pick argmin of a noisy validation-loss estimate -> realized loss from ~0 (perfect verifier) to sig^2 (random).
    PREDICTION: a real crossover sigma_v* EXISTS (ensemble's sig^2/K beats a verifier-degraded select). #32's directional
    pre-reg: larger K -> ensemble favored at LOWER sigma_v (crossover sigma_v* DECREASES with K). FALSIFIABLE — measure it.

So the headline test of #32 is: (A) confirm no positive-verifier crossover in the tail model at any K; (B) in the
estimator model, find sigma_v*(K) and its monotonicity. Computable ground truth, free numpy. Vectorized.
"""
import numpy as np, json, os
D = "runs/cycle185_crossover_vs_K"; os.makedirs(D, exist_ok=True)
rng = np.random.default_rng(0)
MU, SIG, REPS = 1.0, 1.0, 30000
KS = [2, 5, 10, 20, 50, 100, 200]

# ---------- MODEL A: tail-grab (c175 exact), positive verifier, sweep K x sigma_v ----------
def tail(K, sv):
    q = rng.normal(MU, SIG, (REPS, K))
    obs = q + rng.normal(0, sv, (REPS, K))
    i = obs.argmax(1); r = np.arange(REPS)
    st = q[r, i]                                  # SELECT realized true quality
    return float(st.mean()), float(obs[r, i].mean() - st.mean()), MU  # select_true, curse, ensemble_true(=mu)

A = {}
SV_A = [0.3, 1.0, 2.0, 4.0, 8.0]
for K in KS:
    A[K] = {sv: tail(K, sv) for sv in SV_A}

# ---------- MODEL B: estimator-combination (c176), sweep K x sigma_v, find crossover ----------
def estim(K, sv):
    eps = rng.normal(0, SIG, (REPS, K))           # candidate i prediction error of a common target (true=0)
    ens_loss = (eps.mean(1) ** 2).mean()          # ENSEMBLE: mean prediction -> squared error ~ sig^2/K
    vobs = eps ** 2 + rng.normal(0, sv, (REPS, K))# verifier: noisy estimate of each candidate's loss
    i = vobs.argmin(1); r = np.arange(REPS)
    sel_loss = (eps[r, i] ** 2).mean()            # SELECT: realized loss of the argmin-validation candidate
    return float(sel_loss), float(ens_loss)

SV_B = np.round(np.linspace(0.0, 6.0, 31), 2)     # fine grid to locate the crossover
B, CROSS = {}, {}
for K in KS:
    curve = {float(sv): estim(K, sv) for sv in SV_B}
    B[K] = curve
    # crossover sigma_v*: smallest sv where ensemble_loss < select_loss (ensemble overtakes)
    star = None
    for sv in SV_B:
        sl, el = curve[float(sv)]
        if el < sl:
            star = float(sv); break
    CROSS[K] = star

json.dump({"MU": MU, "SIG": SIG, "REPS": REPS, "KS": KS, "modelA_tail": {str(k): {str(s): v for s, v in d.items()} for k, d in A.items()},
           "modelB_estim_crossover_sigma_v_star": {str(k): CROSS[k] for k in KS}}, open(f"{D}/score.json", "w"), indent=1)

print(f"=== #32: does the ensemble-vs-select crossover shift with K? (mu={MU}, sig={SIG}, REPS={REPS}) ===\n")
print("MODEL A (TAIL-GRAB, c175 exact, POSITIVE verifier): SELECT_true vs ENSEMBLE_true(=mu=1.0). 'ens>sel?' = does ensemble overtake?")
print(f"{'K':>4} | " + " ".join(f"sv={sv}:sel_t(curse)" for sv in SV_A))
anyA = False
for K in KS:
    cells = []
    for sv in SV_A:
        st, curse, et = A[K][sv]
        if et > st: anyA = True
        cells.append(f"{st:>5.3f}({curse:>4.2f})")
    print(f"{K:>4} | " + "  ".join(cells))
print(f"  --> ANY (K,sigma_v) where ENSEMBLE_true > SELECT_true under positive verifier? {anyA}  "
      f"(PRE-REG A: expect FALSE — no crossover; select->mu+ from above)\n")

print("MODEL B (ESTIMATOR-COMBINATION, c176): crossover sigma_v* = smallest verifier-noise where ENSEMBLE loss < SELECT loss")
print(f"{'K':>5} {'ens_loss(~sig^2/K)':>18} {'sigma_v*':>9}")
for K in KS:
    el0 = B[K][float(SV_B[len(SV_B)//2])][1]
    print(f"{K:>5} {el0:>18.4f} {str(CROSS[K]):>9}")
star_seq = [CROSS[K] for K in KS]
mono_dec = all(a is not None and b is not None and b <= a + 1e-9 for a, b in zip(star_seq, star_seq[1:]))
print(f"  --> sigma_v* sequence over K={KS}: {star_seq}")
print(f"  --> PRE-REG B (#32): sigma_v* DECREASES with K? monotone-non-increasing = {mono_dec}")
print("DONE")
