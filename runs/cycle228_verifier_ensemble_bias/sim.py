"""Cycle 228 — does a VERIFIER ENSEMBLE cancel bias? Unifies c225/c227 (a single biased verifier carries β·f) +
c133 (a correlated fleet shares blind spots) + c48/51 (diversity beats count). The question: if you average K
verifiers, when does the bias cancel? meth-29 pre-reg; free numpy.

MODEL: N candidates, true skill θ_i~N(0,1). K verifiers; verifier k scores s_{k,i} = θ_i + β·f_{k,i} + N(0,σ_v).
Two regimes for the bias features f_{k}:
  DIVERSE  — each verifier biased toward an INDEPENDENT feature f_k (different verifiers fooled by different things).
  SHARED   — ALL verifiers biased toward the SAME feature f (e.g. every LLM over-rewards verbosity/sycophancy).
Ensemble estimate of a candidate = mean_k s_{k,i}. Measure the ensemble's MEAN bias (over the selected/all candidates)
vs K, for DIVERSE vs SHARED.

PRE-REG (meth-29):
 P1 DIVERSE biases → ensemble mean bias DECREASES ~1/√K (the β·f_k terms average like independent noise → 0),
    approaching the unbiased ensemble; verifier diversity CANCELS independent bias.
 P2 SHARED bias → ensemble mean bias FLAT at ≈ β·E[f|·] regardless of K (averaging cuts VARIANCE not the shared
    bias = c133's shared-blind-spot persistence); diversity-in-count does NOT help a common bias.
 P3 takeaway: to de-bias verification, ensemble DIVERSELY-biased verifiers (models/prompts that fail DIFFERENTLY) —
    but biases COMMON to all verifiers (verbosity/sycophancy across LLMs) are the irreducible residual that needs
    the STRUCTURAL anchor (c225's 3rd requirement), NOT more verifiers.
We test on the SELECTION case (the winner's-curse setting): pick best-of-N by a noisy selection metric, then ESTIMATE
the selected candidate's θ via the K-verifier ensemble. Bias = mean(ensemble_est − true θ of selected).
"""
import numpy as np, json, os
D = "runs/cycle228_verifier_ensemble_bias"; os.makedirs(D, exist_ok=True)
rng = np.random.default_rng(0)

N, REPS = 50, 8000
SIG_SEL, SIG_V, BETA, GAMMA_SEL = 1.0, 1.0, 0.8, 1.0   # GAMMA_SEL>0: selection also rewards the (shared) feature
K_VALUES = [1, 2, 4, 8, 16, 32]

def trial(K, regime, rng):
    theta = rng.normal(0, 1, N)
    # the feature that drives SELECTION (and, in SHARED regime, every verifier's bias)
    f_shared = rng.normal(0, 1, N)
    s = theta + GAMMA_SEL * f_shared + rng.normal(0, SIG_SEL, N)
    w = int(np.argmax(s))                                    # winner (best-of-N) — winner's curse
    # K verifier scores of the winner
    if regime == "shared":
        feats = np.repeat(f_shared[w], K)                    # all verifiers biased toward the SAME feature value
    else:  # diverse — each verifier biased toward an independent feature for this candidate
        feats = rng.normal(0, 1, K)                          # independent f_k per verifier (orthogonal biases)
    v = theta[w] + BETA * feats + rng.normal(0, SIG_V, K)
    return float(v.mean() - theta[w])                        # ensemble-mean estimate bias vs true θ of selected

def run(regime):
    out = {}
    for K in K_VALUES:
        biases = np.array([trial(K, regime, rng) for _ in range(REPS)])
        out[K] = {"mean_bias": float(biases.mean()), "sd": float(biases.std())}
    return out

print(f"=== verifier-ensemble bias cancellation (N={N}, REPS={REPS}, β={BETA}, γ_sel={GAMMA_SEL}) ===")
res = {}
for regime in ("diverse", "shared"):
    res[regime] = run(regime)
    print(f"\n[{regime} biases]  K → ensemble mean bias (sd):")
    for K in K_VALUES:
        r = res[regime][K]
        print(f"   K={K:>2}: bias {r['mean_bias']:+.3f}  (sd {r['sd']:.3f})")

d1, d32 = res["diverse"][1]["mean_bias"], res["diverse"][32]["mean_bias"]
s1, s32 = res["shared"][1]["mean_bias"], res["shared"][32]["mean_bias"]
print(f"\n--- DIVERSE: bias K=1 {d1:+.3f} → K=32 {d32:+.3f}  (cancels toward 0?)  ratio {d32/d1 if d1 else float('nan'):.2f}")
print(f"--- SHARED : bias K=1 {s1:+.3f} → K=32 {s32:+.3f}  (persists?)  ratio {s32/s1 if s1 else float('nan'):.2f}")
# sd should fall ~1/sqrt(K) in BOTH (averaging cuts variance); the MEAN is what differs
verdict = ("CONFIRMED: diversity cancels INDEPENDENT verifier bias (→0) but a SHARED bias PERSISTS regardless of K "
           "→ ensemble diverse-biased verifiers; common LLM-biases need the structural anchor (c225)"
           if (abs(d32) < 0.3*abs(d1) and abs(s32) > 0.7*abs(s1)) else "weak/mixed")
print(f"--- VERDICT: {verdict}")
json.dump({"params": {"N": N, "REPS": REPS, "beta": BETA, "gamma_sel": GAMMA_SEL}, "K_values": K_VALUES,
           "diverse": res["diverse"], "shared": res["shared"], "verdict": verdict},
          open(f"{D}/score.json", "w"), indent=1)
print("--- wrote score.json")
