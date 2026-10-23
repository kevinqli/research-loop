"""Cycle 225 — the BIASED-VERIFIER blind spot of the independence lever (motivated by the explore loop's biased-judge
finding: a biased LLM-judge transmits its bias). c45 established: a verifier corrects the winner's curse in proportion
to its INDEPENDENCE from the selector (residual ∝ shared-blind-spot fraction α). c45 models verifier error as
NOISE + shared-blindness. THE GAP: a verifier can be fully independent of the selection (α=0, fresh noise draw) yet
carry its OWN SYSTEMATIC BIAS toward a feature f (e.g. verbosity/persona an LLM-judge over-rewards). Does independence
fix that? And what if the verifier's bias DIRECTION aligns with the selection criterion (the same trait that helped a
candidate get selected is the one the judge rewards)?

MODEL: N candidates; true skill θ_i~N(0,1); a feature f_i~N(0,1) (verbosity/persona). Selection metric
s_i = θ_i + γ_sel·f_i + N(0,σ_sel) → pick argmax (winner's curse). Held-out verifier of the selected candidate:
v = θ + β·f + N(0,σ_v), with an INDEPENDENT noise draw (α=0 — independent of the selection noise by construction).
β = verifier's systematic feature-bias. Compare estimators of the SELECTED candidate's TRUE θ across REPS.

PRE-REG (meth-29):
 P1 naive (report the selection metric s) → winner's-cursed, overstates θ (bias > 0, grows with K).
 P2 unbiased held-out (β=0) → recovers θ (selective-bias fixed; c45/c223 baseline; residual ≈ 0).
 P3 biased held-out, FEATURE-NEUTRAL selection (γ_sel=0) → f_selected ≈ independent of selection → β·f adds VARIANCE
    but ≈ 0 MEAN bias (the bias averages out; independence "mostly" works on the mean, precision suffers).
 P4 biased held-out, FEATURE-ALIGNED selection (γ_sel>0) → selected candidates are ENRICHED in high-f → E[f|selected]>0
    → the biased verifier carries a SYSTEMATIC residual bias β·E[f|selected] that INDEPENDENCE DOES NOT REMOVE, and it
    "confirms" the inflated pick (shared bias DIRECTION masquerading as independent confirmation) → false confidence.
If P4 shows a surviving bias → a THIRD verifier requirement beyond independent (c45) + secret/rotating (c50):
UNBIASED/CALIBRATED — especially when the verifier's bias aligns with the selection criterion.
"""
import numpy as np, json, os
D = "runs/cycle225_biased_verifier"; os.makedirs(D, exist_ok=True)
rng = np.random.default_rng(0)

N, REPS = 50, 20000
SIG_SEL, SIG_V, BETA = 1.0, 1.0, 0.8          # selection noise, verifier noise, verifier feature-bias
K_VALUES = [50]                                # candidates per selection (N)

def trial(gamma_sel, beta, rng):
    theta = rng.normal(0, 1, N)
    f = rng.normal(0, 1, N)
    s = theta + gamma_sel * f + rng.normal(0, SIG_SEL, N)     # selection metric
    w = int(np.argmax(s))                                      # winner (best-of-N) — the winner's curse
    theta_w, f_w = theta[w], f[w]
    naive = s[w]                                              # report the selection metric (cursed)
    v_unbiased = theta_w + rng.normal(0, SIG_V)              # independent held-out, UNBIASED (β=0)
    v_biased = theta_w + beta * f_w + rng.normal(0, SIG_V)   # independent held-out, BIASED toward f
    return theta_w, naive, v_unbiased, v_biased, f_w

def run(gamma_sel, beta, label):
    rows = np.array([trial(gamma_sel, beta, rng) for _ in range(REPS)])
    theta_w, naive, v_unb, v_bias, f_w = rows.T
    # bias of each estimator relative to the SELECTED candidate's TRUE θ (selective bias, c223 vocab)
    out = {"label": label, "gamma_sel": gamma_sel, "beta": beta,
           "true_theta_selected": float(theta_w.mean()),
           "naive_bias": float((naive - theta_w).mean()),
           "unbiased_heldout_bias": float((v_unb - theta_w).mean()),
           "biased_heldout_bias": float((v_bias - theta_w).mean()),
           "biased_heldout_sd": float((v_bias - theta_w).std()),
           "E_f_given_selected": float(f_w.mean())}
    return out

print(f"=== biased-verifier blind spot (N={N}, REPS={REPS}, σ_sel={SIG_SEL}, σ_v={SIG_V}, β={BETA}) ===")
results = []
for gamma, lab in [(0.0, "feature-NEUTRAL selection"), (1.0, "feature-ALIGNED selection")]:
    r = run(gamma, BETA, lab); results.append(r)
    print(f"\n[{lab}]  E[f|selected]={r['E_f_given_selected']:+.3f}")
    print(f"  true θ of selected (after winner's curse on skill): {r['true_theta_selected']:+.3f}")
    print(f"  naive (report selection metric)       bias {r['naive_bias']:+.3f}   <- the winner's curse")
    print(f"  UNBIASED independent held-out         bias {r['unbiased_heldout_bias']:+.3f}   <- c45/c223: fixes it")
    print(f"  BIASED  independent held-out (β={BETA})  bias {r['biased_heldout_bias']:+.3f}  (sd {r['biased_heldout_sd']:.3f})")

# headline: does independence remove the verifier's systematic bias?
neutral = next(r for r in results if r["gamma_sel"] == 0.0)
aligned = next(r for r in results if r["gamma_sel"] == 1.0)
print(f"\n--- biased-held-out residual bias: feature-NEUTRAL {neutral['biased_heldout_bias']:+.3f}  vs  feature-ALIGNED {aligned['biased_heldout_bias']:+.3f}")
print(f"--- predicted aligned bias β·E[f|sel] = {BETA*aligned['E_f_given_selected']:+.3f}  (matches measured?)")
verdict = ("CONFIRMED: independence does NOT remove a verifier's systematic bias when it aligns with selection "
           "→ 3rd requirement = UNBIASED/CALIBRATED verifier" if aligned["biased_heldout_bias"] > 0.1 else "weak/null")
print(f"--- VERDICT: {verdict}")
json.dump({"params": {"N": N, "REPS": REPS, "beta": BETA}, "results": results,
           "predicted_aligned_bias": BETA*aligned["E_f_given_selected"], "verdict": verdict},
          open(f"{D}/score.json", "w"), indent=1)
print("--- wrote score.json")
