"""Cycle 169 (HARDEN c82 — checkpoint PLACEMENT under HETEROGENEOUS step difficulty; tail/checkpoint pillar;
diversified off the over-verification/curse/benchmark-design threads). c82: "checkpoint THROUGHOUT, EVENLY" minimizes
error-propagation waste — but it assumed UNIFORM per-step error. Real long-horizon tasks have a FEW hard bottleneck
steps. Where should a FIXED checkpoint budget B go? Ties c154's heterogeneous-allocation theme to the CHAIN/tail.

MODEL (c82 propagation): L steps, per-step error rate p_i (heterogeneous). B checks placed at positions; an error at
step i is caught by the FIRST check at-or-after i, wasting (next_check - i) steps of downstream work built on the
corrupted state. Expected wasted work = sum_i p_i * (next_check_after_i - i). Minimize over check placement.
Policies: UNIFORM (even spacing, c82) | DIFFICULTY (equalize cumulative p_i per segment - denser where p high) |
ORACLE (DP: optimal placement minimizing weighted waste - naturally POSITION-aware: a check right after each hard step).

PRE-REG: P1 (control) uniform p -> all policies ~equal. P2 heterogeneous p -> DIFFICULTY beats UNIFORM, gain grows
with heterogeneity. P3 (non-obvious) ORACLE beats DIFFICULTY too - because it's POSITION-aware (a hard step EARLY in a
segment wastes more than late, so put a check RIGHT AFTER each hard step, not just 'denser in hard regions').
"""
import numpy as np, json, os
os.makedirs("runs/cycle169_heterogeneous_checkpoint", exist_ok=True)
rng = np.random.default_rng(0)

L = 50
B = 8                  # checkpoint budget
REPS = 400             # for random heterogeneous p draws

def waste(p, checks):
    """expected wasted work = sum_i p_i*(next_check_after_i - i); checks = sorted positions in 1..L (L always a check)."""
    checks = sorted(set(list(checks) + [L]))
    w = 0.0; ci = 0
    for i in range(1, L + 1):
        while checks[ci] < i: ci += 1
        w += p[i - 1] * (checks[ci] - i)
    return w

def uniform_checks(B):
    return [round(L * k / B) for k in range(1, B + 1)]

def difficulty_checks(p, B):
    # place checks at cumulative-p quantiles (equalize sum p_i per segment)
    cum = np.cumsum(p); total = cum[-1]
    targets = [total * k / B for k in range(1, B + 1)]
    return [int(np.searchsorted(cum, t) + 1) for t in targets]

def oracle_checks(p, B):
    # DP: partition 1..L into B segments (each ending at a check) minimizing sum of segment costs
    # seg_cost(a,b) = sum_{i=a+1..b} p_i*(b-i)   (0-indexed p)
    INF = float("inf")
    segcost = np.full((L + 1, L + 1), INF)
    for a in range(L):
        for b in range(a + 1, L + 1):
            segcost[a][b] = sum(p[i - 1] * (b - i) for i in range(a + 1, b + 1))
    dp = np.full((B + 1, L + 1), INF); dp[0][0] = 0
    for k in range(1, B + 1):
        for b in range(1, L + 1):
            for a in range(0, b):
                if dp[k - 1][a] + segcost[a][b] < dp[k][b]:
                    dp[k][b] = dp[k - 1][a] + segcost[a][b]
    return dp[B][L]   # we only need the optimal cost for L being the last check

def gen_p(mode):
    if mode == "uniform":   return np.full(L, 0.05)
    if mode == "mild":      return np.clip(rng.lognormal(np.log(0.04), 0.5, L), 0, 0.5)
    if mode == "spiky":     # mostly low, a few hard spikes
        p = np.full(L, 0.01); idx = rng.choice(L, 5, replace=False); p[idx] = rng.uniform(0.3, 0.6, 5); return p

out = {"L": L, "B": B, "REPS": REPS, "by_mode": {}}
print(f"=== HARDEN c82: checkpoint placement under heterogeneous difficulty (L={L}, B={B}) ===")
print(f"{'mode':>10} {'uniform':>10} {'difficulty':>11} {'oracle(DP)':>11} {'unif/oracle':>12}")
for mode in ["uniform", "mild", "spiky"]:
    wu, wd, wo = [], [], []
    reps = 1 if mode == "uniform" else REPS
    for _ in range(reps):
        p = gen_p(mode)
        wu.append(waste(p, uniform_checks(B)))
        wd.append(waste(p, difficulty_checks(p, B)))
        wo.append(oracle_checks(p, B))
    mu, md, mo = float(np.mean(wu)), float(np.mean(wd)), float(np.mean(wo))
    out["by_mode"][mode] = {"uniform": round(mu,3), "difficulty": round(md,3), "oracle": round(mo,3),
                            "uniform_over_oracle": round(mu/mo,2) if mo>0 else None,
                            "difficulty_over_oracle": round(md/mo,2) if mo>0 else None}
    print(f"{mode:>10} {mu:>10.3f} {md:>11.3f} {mo:>11.3f} {mu/mo if mo>0 else 0:>12.2f}")
json.dump(out, open("runs/cycle169_heterogeneous_checkpoint/score.json", "w"), indent=1)
print("--- difficulty/oracle ratio (is oracle's position-awareness extra over difficulty-weighting?) ---")
for m in ["mild","spiky"]: print(f"  {m}: difficulty/oracle = {out['by_mode'][m]['difficulty_over_oracle']}")
print("PRE-REG: P1 uniform-mode all equal; P2 heterogeneous difficulty<uniform; P3 oracle<difficulty (position-aware).")
print("DONE")
