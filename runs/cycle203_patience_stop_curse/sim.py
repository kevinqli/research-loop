"""Cycle 203 (#24 — the DEPLOYMENT-REALISTIC search pattern's winner's curse: an agent that keeps iterating until
improvements STALL (patience-stop) and ships its running-best. Diversify MODE off the real-agent/consolidation run;
extends the curse taxonomy c7 [fixed-K batch max] + c157 [first-over-τ]; free numpy. meth-28: c7=max-of-K curse
E[max_K]; c157=first-over-τ curse 1+σ² truncation).

The taxonomy has two idealizations: (a) FIXED-K — examine K candidates, ship argmax-observed (c7; curse grows with
K ∝ σ_v·√(2 ln K)); (b) FIRST-over-τ — ship the first observed ≥ τ (c157; curse ≈ constant truncation). But agents
actually search by PATIENCE: keep generating, track the running-best-observed, STOP when m consecutive candidates
fail to beat it, ship the running-best. This sim characterizes (c) PATIENCE-m's curse vs (a)/(b).

MODEL: candidates drawn sequentially, true q~N(MU,SIG_Q); observed s = q + N(0,SIG_V). Ship by OBSERVED; report =
observed; curse = reported_observed − true_q of shipped. Metrics per strategy: true_q shipped, curse, #examined.
HELD-OUT fix: a 2nd independent obs s' = q + fresh noise of the shipped → its curse should be ~0 for all three.

PRE-REG: (P1) PATIENCE-m INTERPOLATES — more patience m → more candidates effectively examined → bigger max-of-K_eff
curse + higher true quality (between c157-small and c7-largeK). (P2) its curse tracks the EFFECTIVE count it examines
(curse ≈ σ_v·√(2 ln K_examined)) — i.e. patience-stop is a max-of-K curse with a data-dependent K. (P3) the held-out
re-eval removes the curse for ALL three (curse→0) — the fix is strategy-invariant (the c170 axis-A point).
"""
import numpy as np, json, os
D = "runs/cycle203_patience_stop_curse"; os.makedirs(D, exist_ok=True)
rng = np.random.default_rng(0)
MU, SIG_Q, SIG_V, REPS = 1.0, 1.0, 1.0, 40000
MAXDRAW = 2000  # cap on sequential draws (patience search)

def draw_rep():
    q = rng.normal(MU, SIG_Q, MAXDRAW); s = q + rng.normal(0, SIG_V, MAXDRAW); return q, s

def fixed_K(q, s, K):
    i = s[:K].argmax(); return q[i], s[i], K

def first_tau(q, s, tau):
    idx = np.where(s >= tau)[0]
    if len(idx) == 0: i = s.argmax(); return q[i], s[i], MAXDRAW   # never crossed → examined all
    i = idx[0]; return q[i], s[i], i + 1

def patience(q, s, m):
    best_i, best_s, since = 0, s[0], 0
    for j in range(1, MAXDRAW):
        if s[j] > best_s: best_s, best_i, since = s[j], j, 0
        else:
            since += 1
            if since >= m: return q[best_i], s[best_i], j + 1
    return q[best_i], s[best_i], MAXDRAW

def heldout(q_true):  # independent 2nd observation of the shipped candidate
    return q_true + rng.normal(0, SIG_V)

STRATS = ([("fixedK_%d" % K, lambda q, s, K=K: fixed_K(q, s, K)) for K in [5, 20, 100]] +
          [("first_tau_%.1f" % t, lambda q, s, t=t: first_tau(q, s, t)) for t in [1.5, 2.0, 2.5]] +
          [("patience_%d" % m, lambda q, s, m=m: patience(q, s, m)) for m in [3, 10, 30]])

acc = {name: {"tq": [], "curse": [], "nex": [], "ho_curse": []} for name, _ in STRATS}
for _ in range(REPS):
    q, s = draw_rep()
    for name, fn in STRATS:
        tq, so, nex = fn(q, s)
        acc[name]["tq"].append(tq); acc[name]["curse"].append(so - tq); acc[name]["nex"].append(nex)
        acc[name]["ho_curse"].append(heldout(tq) - tq)

print(f"=== Patience-stop vs fixed-K vs first-τ winner's curse (mu={MU}, sig_q={SIG_Q}, sig_v={SIG_V}, REPS={REPS}) ===")
print(f"{'strategy':>14} {'true_q':>8} {'curse(report)':>14} {'#examined':>10} {'curse(held-out)':>16}")
out = {}
for name, _ in STRATS:
    a = acc[name]; row = {"true_q": float(np.mean(a["tq"])), "curse": float(np.mean(a["curse"])),
                          "n_examined": float(np.mean(a["nex"])), "curse_heldout": float(np.mean(a["ho_curse"]))}
    out[name] = row
    print(f"{name:>14} {row['true_q']:>8.3f} {row['curse']:>14.3f} {row['n_examined']:>10.1f} {row['curse_heldout']:>16.4f}")

# P2 check: does patience curse ≈ σ_v·√(2 ln K_examined)?
print("\nP2 — patience curse vs the max-of-K_examined prediction σ_v·√(2 ln K_ex):")
for m in [3, 10, 30]:
    r = out["patience_%d" % m]; pred = SIG_V * np.sqrt(2 * np.log(max(r["n_examined"], 1.01)))
    print(f"  patience_{m:>2}: examined≈{r['n_examined']:.1f}  curse={r['curse']:.3f}  vs √(2 ln K_ex) pred={pred:.3f}")
p3 = max(abs(out[n]["curse_heldout"]) for n, _ in STRATS) < 0.02
print(f"\nP3 — held-out re-eval removes the curse for ALL strategies (max |curse_heldout| < 0.02)? {p3} "
      f"(max={max(abs(out[n]['curse_heldout']) for n,_ in STRATS):.4f})")
print("PRE-REG: P1 patience interpolates (more m → more examined, higher true_q, bigger curse); P2 curse≈max-of-K_ex; P3 held-out fixes all.")
json.dump({"params": {"MU": MU, "SIG_Q": SIG_Q, "SIG_V": SIG_V, "REPS": REPS}, "strategies": out, "P3_heldout_fixes_all": bool(p3)},
          open(f"{D}/score.json", "w"), indent=1)
print("DONE")
