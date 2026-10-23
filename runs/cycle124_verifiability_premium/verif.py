"""Cycle 124 (NEW, benchmark-design; ties c7+c114): VERIFIABILITY > STAKES. An UNVERIFIED result is indistinguishable
from a winner's-curse delusion (c7), so a benchmark should favor tasks whose ground-truth is CHEAP to independently
check — even at lower stakes — over high-stakes-but-unverifiable ones. Under a verification budget, cheaper-to-verify
tasks let you verify MORE candidates → the best-you-can-CONFIDENTLY-ship is higher + trustworthy.

Model: N=100 candidates, true value v~N(μ,1); noisy proxy = v+N(0,σ_p). A verification budget lets you verify K of
them (verify the top-K by proxy, c110); verification reveals true v; you SHIP the verified-best (trustworthy). Outcome
= true v of the verified-best. Cheap task → large K; expensive task → small K. Higher-stakes pool → larger μ. Find the
crossover: does cheap-verify-many beat expensive-verify-few even at lower μ? numpy-only, REPS reps."""
import numpy as np, json, os
RNG = np.random.default_rng(0)
REPS, N, SIG_P = 20000, 100, 1.0
os.makedirs("runs/cycle124_verifiability_premium", exist_ok=True)

def outcome(K, mu):                              # true v of the verified-best (= ship-able, trustworthy)
    v = RNG.normal(mu, 1, (REPS, N)); proxy = v + RNG.normal(0, SIG_P, (REPS, N))
    topK = np.argsort(-proxy, 1)[:, :K]          # the K we can afford to verify (top by proxy, c110)
    return float(np.take_along_axis(v, topK, 1).max(1).mean())   # verification reveals v; ship the best verified

out = {"model":{"REPS":REPS,"N":N,"SIG_P":SIG_P}, "vs_K":{}, "crossover":{}}
print(f"N={N} candidates, proxy-noise σ={SIG_P}. Outcome = true value of the verified-best (trustworthy, ship-able).\n")
print(f"=== P1: outcome vs #verifiable K (μ=0; cheaper task → larger K) ===")
print(f"{'K (verifiable)':>15} | {'trustworthy outcome':>20}")
for K in [2,5,10,20,40,80]:
    o = outcome(K, 0.0); out["vs_K"][K] = round(o,4); print(f"{K:>15} | {o:>20.4f}")

print(f"\n=== P2: CHEAP-verify-many (μ=0) vs EXPENSIVE-verify-few (K=10) at rising stakes μ_E ===")
print(f"{'expensive μ_E':>14} | {'EXPENSIVE (K=10)':>17} | {'CHEAP (K=80,μ=0)':>17} | winner")
cheap = outcome(80, 0.0)
for muE in [0.0, 0.3, 0.6, 1.0, 1.5]:
    exp = outcome(10, muE); out["crossover"][muE] = {"expensive_K10":round(exp,4), "cheap_K80":round(cheap,4)}
    win = "EXPENSIVE" if exp>cheap else "CHEAP"
    print(f"{muE:>14} | {exp:>17.4f} | {cheap:>17.4f} | {win}")

json.dump(out, open("runs/cycle124_verifiability_premium/score.json","w"), indent=1)
print("\n--- READ THE RAW ROWS (don't trust this summary) ---")
print(f"P1 outcome vs K: " + " ".join(f"K{k}:{out['vs_K'][k]}" for k in [2,5,10,20,40,80]) + " (rises with #verifiable?)")
cx = out["crossover"]
prem = next((mu for mu in [0.0,0.3,0.6,1.0,1.5] if cx[mu]['expensive_K10']>cheap), None)
print(f"P2 verifiability PREMIUM: cheap (K=80,μ=0)={cheap}; expensive (K=10) overtakes only at μ_E≈{prem} → cheap verifiability is worth ~{prem}σ of raw stakes.")
print("=> VERIFIABILITY > STAKES: prefer cheaply-verifiable tasks; an unverified high-stakes result is a winner's-curse delusion (c7), and the verifiability premium is large.")
print("DONE")
