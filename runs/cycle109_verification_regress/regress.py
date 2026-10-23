"""Cycle 109 (NEW mechanism, verification pillar): the VERIFICATION REGRESS — how many verification passes until you
trust a finding, and when does the regress TERMINATE? The body assumes ONE held-out re-eval; here: with MULTIPLE
IMPERFECT verifiers, does stacking passes drive false-acceptance to 0 or plateau? Formalizes c45 (independence) as a
Bayesian regress.

Model: a finding is TRUE with prior p0. A TRUE finding always passes an honest check (TPR=1). A FALSE finding is in a
SHARED BLIND SPOT with prob s (fools ALL verifiers -> passes every pass), else each INDEPENDENT verifier catches it
with prob q (passes w.p. 1-q). After k passes: P(false passes k) = s + (1-s)(1-q)^k. Posterior that a finding which
passed k checks is true: P(true|k) = p0 / (p0 + (1-p0)*P(false passes k)). s=0 => regress terminates (->1); s>0 =>
plateaus at the floor p0/(p0+(1-p0)*s). numpy-only; analytic + Monte-Carlo cross-check."""
import numpy as np, json, os
RNG = np.random.default_rng(0)
P0, Q, REPS = 0.30, 0.70, 400000
os.makedirs("runs/cycle109_verification_regress", exist_ok=True)

def post_analytic(k, s):
    pf = s + (1-s)*(1-Q)**k                      # P(false finding passes k checks)
    return P0 / (P0 + (1-P0)*pf), pf

def post_mc(k, s):                                # Monte-Carlo cross-check
    true = RNG.random(REPS) < P0
    blind = RNG.random(REPS) < s                  # false-finding in the shared blind spot
    # pass all k: true always; false passes if blind, else survives k independent catches
    passes = np.where(true, True,
              np.where(blind, True, (RNG.random((REPS,k)) > Q).all(axis=1)))
    sel = passes
    return float(true[sel].mean())                # P(true | passed k)

out = {"model": {"P0":P0,"Q":Q,"REPS":REPS}, "grid": {}}
S = [0.0, 0.05, 0.2, 0.5]; Ks = [1,2,3,4,6]
print(f"prior p0(true)={P0}, per-verifier catch q={Q}.  Posterior P(true | passed k checks):\n")
print(f"{'k':>3} | " + " ".join(f"s={s}".rjust(11) for s in S) + "      (s = shared-blind-spot fraction)")
for k in Ks:
    row=[]
    for s in S:
        pa,_ = post_analytic(k,s); out["grid"][f"k{k}_s{s}"]=round(float(pa),4); row.append(pa)
    print(f"{k:>3} | " + " ".join(f"{v:.4f}".rjust(11) for v in row))
# floors + MC check
print("\nfloor (k->inf) = p0/(p0+(1-p0)s):", {s: round(P0/(P0+(1-P0)*s),4) for s in S})
print("MC cross-check (k=3): " + " ".join(f"s={s}:{post_mc(3,s):.4f} vs analytic {post_analytic(3,s)[0]:.4f}" for s in [0.0,0.2]))
json.dump(out, open("runs/cycle109_verification_regress/score.json","w"), indent=1)
print("\n--- READ THE RAW ROWS (don't trust this summary) ---")
g=out["grid"]
print(f"P1 s=0 (independent): P(true|k) k1->k6: {g['k1_s0.0']} -> {g['k6_s0.0']}  (->1? regress TERMINATES)")
print(f"P2 s=0.2 (correlated): P(true|k) k1->k6: {g['k1_s0.2']} -> {g['k6_s0.2']}  (PLATEAUS at floor {round(P0/(P0+(1-P0)*0.2),4)}?)")
print(f"P2 marginal gain k3->k6 at s=0.2: {round(g['k6_s0.2']-g['k3_s0.2'],4)} (~0 => more passes worthless); at s=0: {round(g['k6_s0.0']-g['k3_s0.0'],4)}")
print("=> independent checks terminate the regress fast (~2-3 passes); correlated checks plateau at the shared-blind-spot floor. Lower s (diverse verifiers), don't add passes.")
print("DONE")
