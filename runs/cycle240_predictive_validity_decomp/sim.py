"""Cycle 240 — decompose the c238/#176 paper's PREDICTIVE VALIDITY criterion into RELIABILITY × TRANSFER, connecting
the loop's c181 T_eff (the loop's domain) to the paper's in-sample→out-of-sample rank correlation (the paper's domain),
and filling the gap the paper explicitly left ("tier independence is asserted, not tested"). meth-29 pre-reg; free numpy.

The paper 2606.19704 ranks agents by PREDICTIVE VALIDITY = Spearman(in-sample rank, out-of-sample rank), and observed
ρ=0.69 (planning) / −0.13 (execution). CLAIM to test: predictive validity FACTORIZES into two independent killers —
  (A) RELIABILITY of the in-sample ranking (c181: governed by T_eff = T/(1+(T−1)ρ_red); a redundant benchmark → low R), and
  (B) TRANSFER τ = corr(in-sample true skill, out-of-sample true skill) (the paper's domain: does the skill measured
      in-sample even predict OOS skill? distribution shift can make τ low or NEGATIVE — cf the execution ρ=−0.13).
PREDICTION: PV ≈ √R · τ → killed by EITHER a redundant benchmark (low T_eff → low R) OR non-transferring tasks (low τ).
The loop's T_eff fixes only factor (A); you ALSO need transfer-valid tasks (B) that T_eff cannot give you.

MODEL: N agents; in-sample true skill s_in~N(0,1); OOS true skill s_oos = τ·s_in + √(1−τ²)·N(0,1) (corr=τ). T tasks,
inter-task NOISE-redundancy ρ_red (c181): x_{i,t}=s_in_i + √ρ_red·u_i + √(1−ρ_red)·v_{i,t} (shared + per-task noise,
both N(0,σe)). In-sample ranking = rank by mean_t x. PV = Spearman(in-sample mean rank, s_oos). R = corr(in-sample
mean, s_in)². Sweep ρ_red (→T_eff) × τ; check PV ≈ √R·τ.
"""
import numpy as np, json, os
D = "runs/cycle240_predictive_validity_decomp"; os.makedirs(D, exist_ok=True)
rng = np.random.default_rng(0)
N, T, REPS, SIG_E = 60, 20, 3000, 1.0

def spearman(a, b):
    ra = a.argsort().argsort(); rb = b.argsort().argsort()
    return np.corrcoef(ra, rb)[0, 1]

def teff(T_, rho): return T_ / (1 + (T_ - 1) * rho)

def run(rho_red, tau):
    Rs, PVs = [], []
    for _ in range(REPS):
        s_in = rng.normal(0, 1, N)
        s_oos = tau * s_in + np.sqrt(max(1 - tau**2, 0)) * rng.normal(0, 1, N)
        u = rng.normal(0, SIG_E, N)
        v = rng.normal(0, SIG_E, (N, T))
        x = s_in[:, None] + np.sqrt(rho_red) * u[:, None] + np.sqrt(1 - rho_red) * v
        in_mean = x.mean(axis=1)
        Rs.append(np.corrcoef(in_mean, s_in)[0, 1] ** 2)     # reliability (vs true in-sample skill)
        PVs.append(spearman(in_mean, s_oos))                  # predictive validity (in-sample rank vs OOS skill)
    R = float(np.mean(Rs)); PV = float(np.mean(PVs))
    te = teff(T, rho_red)
    R_analytic = (1.0 * te) / (1.0 * te + SIG_E**2)           # c181 reliability closed form (σs=1)
    return {"rho_red": rho_red, "tau": tau, "T_eff": round(te, 2), "R": R, "R_analytic": round(R_analytic, 3),
            "PV": PV, "pred_sqrtR_tau": round(np.sqrt(R) * tau, 3)}

RHOS = [0.0, 0.3, 0.6]; TAUS = [1.0, 0.7, 0.3, 0.0, -0.2]
print(f"=== predictive validity = √(reliability) × transfer  (N={N}, T={T}, REPS={REPS}) ===")
print(f"{'ρ_red':>6}{'T_eff':>7}{'R':>7}{'τ':>6}{'PV':>7}{'√R·τ':>8}")
rows = []
for rho in RHOS:
    for tau in TAUS:
        r = run(rho, tau); rows.append(r)
        print(f"{rho:>6.1f}{r['T_eff']:>7.2f}{r['R']:>7.3f}{tau:>6.1f}{r['PV']:>7.3f}{r['pred_sqrtR_tau']:>8.3f}")

# headline: does PV factorize as √R·τ? and do BOTH low-T_eff and low-τ independently kill it?
err = np.mean([abs(r["PV"] - r["pred_sqrtR_tau"]) for r in rows])
hiR_hiTau = next(r for r in rows if r["rho_red"] == 0.0 and r["tau"] == 1.0)
loR_hiTau = next(r for r in rows if r["rho_red"] == 0.6 and r["tau"] == 1.0)   # redundant benchmark kills PV
hiR_loTau = next(r for r in rows if r["rho_red"] == 0.0 and r["tau"] == 0.3)   # non-transfer kills PV
print(f"\n--- mean|PV − √R·τ| = {err:.3f}  (factorization holds if ~0)")
print(f"--- redundancy kills PV (τ=1.0): ρ_red 0.0 PV {hiR_hiTau['PV']:.2f} → ρ_red 0.6 PV {loR_hiTau['PV']:.2f}  (via T_eff↓R, c181's half)")
print(f"--- non-transfer kills PV (ρ_red 0): τ=1.0 PV {hiR_hiTau['PV']:.2f} → τ=0.3 PV {hiR_loTau['PV']:.2f}  (the paper's half — T_eff CAN'T fix)")
verdict = ("CONFIRMED: PV ≈ √R·τ — predictive validity factorizes into RELIABILITY (c181 T_eff) × TRANSFER (the paper's "
           "axis); BOTH a redundant benchmark AND non-transferring tasks independently kill it; the loop's T_eff fixes "
           "only the reliability factor" if err < 0.05 else "factorization approximate/off")
print(f"--- VERDICT: {verdict}")
json.dump({"rows": rows, "factorization_err": err, "verdict": verdict}, open(f"{D}/score.json","w"), indent=1)
print("--- wrote score.json")
