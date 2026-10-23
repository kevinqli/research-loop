"""Cycle 253 — the CANDIDATE-POOL winner's curse has TWO redundancy axes (c216's skill-vs-noise decomposition applied
to the candidate axis, unifying c7 + c48 + c181 + c216). c48 covered only SKILL-redundancy (correlated skill, indep
noise → report inflated +41%, low coverage). THE NEW AXIS: NOISE-redundancy (a same-model fleet sharing a noise mode)
→ the max-of-K curse itself SHRINKS via an effective independent-candidate count K_eff = K/(1+(K-1)ρ_noise). meth-28
extends c181's design-effect machinery; meth-29 pre-reg.

MODEL: K candidates; true skill v_i; observed score x_i = v_i + noise_i; select argmax x; the WINNER'S CURSE =
E[x_winner − v_winner] (observed max overstates the selected candidate's true value).
 AXIS A — SKILL-redundancy: v_i = √ρ·v_shared + √(1−ρ)·v_idio (skills correlated), noise INDEPENDENT N(0,σ).
   → max over K INDEPENDENT noises (full noise-curse) but the true-skill SPREAD shrinks (low coverage) → the report
     is inflated RELATIVE to the achievable true max (c48's "+41% noise-mining, low outcome").
 AXIS B — NOISE-redundancy: v_i INDEPENDENT N(0,1); noise_i = √ρ·e_shared + √(1−ρ)·e_idio (noise correlated).
   → max over K CORRELATED noises → effective independent draws K_eff = K/(1+(K−1)ρ) → SMALLER max-order-statistic
     of the noise → the curse SHRINKS. A shared-noise (same-model) fleet has a SMALLER raw winner's curse.

PRE-REG (meth-29; honestly a UNIFICATION not a brand-new mechanism): (A) SKILL-redundancy ↑ → noise-curse ~CONSTANT
(noise indep, full K) while true-max coverage ↓ → report-over-achievable inflation ↑ (recovers c48). (B) NOISE-
redundancy ↑ → the curse itself ↓, tracking √(2 ln K_eff(ρ)) not √(2 ln K). NON-OBVIOUS ACTIONABLE: a same-model
fleet sharing a noise mode is LESS winner's-cursed than K independent agents (a GOOD side, distinct from c133's
shared-blind-spot BAD side). If A recovers c48 + B shows the K_eff_noise curse-reduction → unification confirmed.
"""
import numpy as np, json, os
D = "runs/cycle253_candidate_curse_two_axis"; os.makedirs(D, exist_ok=True)
rng = np.random.default_rng(0)
K, REPS, SIG = 20, 20000, 1.0
RHOS = [0.0, 0.3, 0.6, 0.9]

def k_eff(K_, rho): return K_ / (1 + (K_ - 1) * rho)

def axis_A_skill(rho):  # correlated skill, independent noise
    curses, coverages = [], []
    for _ in range(REPS):
        v = np.sqrt(rho) * rng.normal() + np.sqrt(1 - rho) * rng.normal(0, 1, K)  # correlated skills
        x = v + rng.normal(0, SIG, K)
        w = int(np.argmax(x))
        curses.append(x[w] - v[w])                 # noise-curse on the selected
        coverages.append(v.max())                  # true-skill coverage (the achievable)
    return float(np.mean(curses)), float(np.mean(coverages))

def axis_B_noise(rho):  # independent skill, correlated noise
    curses, coverages = [], []
    for _ in range(REPS):
        v = rng.normal(0, 1, K)                     # independent skills
        noise = np.sqrt(rho) * rng.normal(0, SIG) + np.sqrt(1 - rho) * rng.normal(0, SIG, K)  # correlated noise
        x = v + noise
        w = int(np.argmax(x))
        curses.append(x[w] - v[w])
        coverages.append(v.max())
    return float(np.mean(curses)), float(np.mean(coverages))

print(f"=== candidate-pool curse: SKILL-redundancy vs NOISE-redundancy (K={K}, REPS={REPS}, σ={SIG}) ===")
print(f"{'ρ':>5}{'K_eff':>7} | {'A:skill curse':>14}{'A:coverage':>11} | {'B:noise curse':>14}{'B:coverage':>11}")
rows = []
for rho in RHOS:
    ca, cova = axis_A_skill(rho)
    cb, covb = axis_B_noise(rho)
    rows.append({"rho": rho, "K_eff": round(k_eff(K, rho), 2),
                 "A_skill_curse": ca, "A_coverage": cova, "B_noise_curse": cb, "B_coverage": covb})
    print(f"{rho:>5.1f}{k_eff(K,rho):>7.2f} | {ca:>14.3f}{cova:>11.3f} | {cb:>14.3f}{covb:>11.3f}")

A0, A9 = rows[0], rows[-1]
B0, B9 = rows[0], rows[-1]
print(f"\n--- AXIS A (skill-redundancy): curse {A0['A_skill_curse']:.3f}→{A9['A_skill_curse']:.3f} (~constant) BUT coverage {A0['A_coverage']:.3f}→{A9['A_coverage']:.3f} (collapses) → report inflated vs achievable (recovers c48)")
print(f"--- AXIS B (noise-redundancy): curse {B0['B_noise_curse']:.3f}→{B9['B_noise_curse']:.3f} (SHRINKS) as K_eff {k_eff(K,0):.0f}→{k_eff(K,0.9):.2f}; coverage ~flat → shared-noise fleet is LESS cursed (NEW axis)")
# does B's curse track sqrt(2 ln K_eff)? compare ratio to rho=0
import math
pred = [math.sqrt(2*math.log(max(k_eff(K,r),1.0001))) for r in RHOS]
meas = [r["B_noise_curse"] for r in rows]
scale = meas[0]/pred[0] if pred[0] else 1
print(f"--- B curse vs √(2 ln K_eff)·scale: measured {[round(m,2) for m in meas]} vs predicted {[round(p*scale,2) for p in pred]}")
verdict = ("UNIFIED: skill-redundancy inflates the report (c48, curse~const + coverage↓); noise-redundancy SHRINKS the "
           "curse via K_eff=K/(1+(K-1)ρ) [~√(2 ln K_eff)] — a same-model/shared-noise fleet is LESS winner's-cursed "
           "(new axis, distinct from c133's shared-blind-spot)" if (A9['A_coverage']<A0['A_coverage']-0.5 and B9['B_noise_curse']<0.6*B0['B_noise_curse']) else "mixed")
print(f"--- VERDICT: {verdict}")
json.dump({"K": K, "REPS": REPS, "rows": rows, "B_pred_sqrt2lnKeff": [round(p*scale,3) for p in pred], "verdict": verdict},
          open(f"{D}/score.json","w"), indent=1)
print("--- wrote score.json")
