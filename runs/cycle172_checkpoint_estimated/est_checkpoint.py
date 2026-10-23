"""Cycle 172 (#29 — HARDEN c169 + the c101 cold-start lens: does the heterogeneous-checkpoint gain survive when the
per-step difficulties must be ESTIMATED, not known? tail/checkpoint pillar). c169: with KNOWN p_i, place checks right
after hard steps (4.3x less waste than even). But real agents don't know which steps are hard a priori — they ESTIMATE
p_hat_i (noisily, and possibly with cold-start BIAS, c101). Does the gain survive, or REVERSE like c101's successive-
halving (which flipped to worse-than-uniform under anti-correlated early signal)?

MODEL (c169 waste): L steps, true p_i (SPIKY: 5 hard spikes), B checks; waste = sum_i p_i*(next_check_after_i - i).
Placement uses an ESTIMATE p_hat (placement by p_hat, cost evaluated with TRUE p):
  p_hat_i = mean_p + (1-2*rho)*(p_i - mean_p) + eta*noise
  rho=0: p_hat ~ true (correlated); rho=0.5: p_hat ~ uninformative (mean); rho=1: ANTI-correlated (hard steps look easy).
  eta: unbiased estimation noise.
Policies: EVEN (ignores p_hat - robust baseline, c82) | DIFFICULTY-by-p_hat (the realistic c169 policy on estimates)
| ORACLE (true p - the c169 ceiling).

PRE-REG: P1 unbiased noisy p_hat (rho=0, eta>0) -> DIFFICULTY beats EVEN but DEGRADES toward even as eta grows. P2 the
c101 REVERSAL: under anti-correlated estimates (rho>0.5) DIFFICULTY goes WORSE than even (confidently mis-places +
loses even-spacing robustness). P3 rule: difficulty-checkpointing pays with low/unbiased estimation error; under
cold-start/biased estimates fall back to even (robust) = c101's 'warm past the cold-start before racing'.
"""
import numpy as np, json, os
os.makedirs("runs/cycle172_checkpoint_estimated", exist_ok=True)
rng = np.random.default_rng(0)
L, B, REPS = 50, 8, 2000

def waste(p_true, checks):
    checks = sorted(set(list(checks) + [L])); w = 0.0; ci = 0
    for i in range(1, L + 1):
        while checks[ci] < i: ci += 1
        w += p_true[i - 1] * (checks[ci] - i)
    return w

def even_checks():
    return [round(L * k / B) for k in range(1, B + 1)]

def difficulty_checks(p_est):
    p = np.clip(p_est, 1e-6, None); cum = np.cumsum(p); tot = cum[-1]
    return [int(np.searchsorted(cum, tot * k / B) + 1) for k in range(1, B + 1)]

def spiky_p():
    p = np.full(L, 0.01); idx = rng.choice(L, 5, replace=False); p[idx] = rng.uniform(0.3, 0.6, 5); return p

def est(p_true, rho, eta):
    m = p_true.mean()
    return np.clip(m + (1 - 2 * rho) * (p_true - m) + eta * rng.normal(0, p_true.std() + 0.1, L), 0, None)

out = {"L": L, "B": B, "REPS": REPS, "P1_unbiased_eta_sweep": {}, "P2_anticorr_rho_sweep": {}}
print(f"=== #29: checkpoint placement under ESTIMATED difficulty (L={L}, B={B}, spiky, REPS={REPS}) ===")
# P1: unbiased (rho=0), vary eta
print("P1 UNBIASED noise (rho=0): does difficulty-by-estimate beat even, degrading with eta?")
print(f"{'eta':>6} {'even':>8} {'difficulty(est)':>16} {'oracle(true)':>13} {'diff<even?':>10}")
for eta in [0.0, 0.5, 1.0, 2.0, 4.0]:
    we, wd, wo = [], [], []
    for _ in range(REPS):
        p = spiky_p(); ph = est(p, 0.0, eta)
        we.append(waste(p, even_checks())); wd.append(waste(p, difficulty_checks(ph))); wo.append(waste(p, difficulty_checks(p)))
    me, md, mo = np.mean(we), np.mean(wd), np.mean(wo)
    out["P1_unbiased_eta_sweep"][eta] = {"even": round(float(me),3), "difficulty": round(float(md),3), "oracle": round(float(mo),3)}
    print(f"{eta:>6} {me:>8.3f} {md:>16.3f} {mo:>13.3f} {'YES' if md<me else 'NO':>10}")
# P2: anti-correlation (eta fixed small), vary rho
print("\nP2 BIAS/anti-correlation (eta=0.5), vary rho (0=correlated, 0.5=uninformative, 1=anti-correlated):")
print(f"{'rho':>6} {'even':>8} {'difficulty(est)':>16} {'diff<even?':>10}")
for rho in [0.0, 0.25, 0.5, 0.75, 1.0]:
    we, wd = [], []
    for _ in range(REPS):
        p = spiky_p(); ph = est(p, rho, 0.5)
        we.append(waste(p, even_checks())); wd.append(waste(p, difficulty_checks(ph)))
    me, md = np.mean(we), np.mean(wd)
    out["P2_anticorr_rho_sweep"][rho] = {"even": round(float(me),3), "difficulty": round(float(md),3), "diff_worse_than_even": bool(md>me)}
    print(f"{rho:>6} {me:>8.3f} {md:>16.3f} {'YES' if md<me else 'NO(worse!)':>10}")
json.dump(out, open("runs/cycle172_checkpoint_estimated/score.json", "w"), indent=1)
print("\nPRE-REG: P1 unbiased -> difficulty beats even, degrades w/ eta; P2 anti-corr (rho>0.5) -> difficulty WORSE than even (c101 reversal).")
print("DONE")
