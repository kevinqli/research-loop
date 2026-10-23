"""Cycle 186 (#28 — do the loop's two big mechanisms unify into ONE 'optimal verification amount', or are they
TWO orthogonal axes? SYNTHESIS/CHALLENGE; diversify MODE off the c185 sim; meth-28: read c170/c177 frame first).

#28's premise: TOO-LITTLE rigor = winner's curse (axis A, estimation-under-selection) and TOO-MUCH rigor =
over-verification (axis B, detection threshold) might be ONE inverted-U that SDT's optimal criterion unifies.
c177 already showed SDT unifies axis B INTERNALLY (tau* tracks d'). The open question: does the accept/reject
STRINGENCY knob tau (axis B's knob) ALSO fix axis A (the curse)?

TOY (one integrated pipeline, ONE stringency knob tau): K candidates, true q~N(mu,sig_q); a noisy verifier scores
s=q+N(0,sig_v). POLICY: ACCEPT candidates with s>=tau; among accepted, SHIP argmax-s and REPORT its score s; if
none pass, REGENERATE a fresh batch (budget cost). Sweep tau (the SINGLE 'verification amount/stringency' knob):
  - true_ship  = true q of the shipped candidate           (higher tau -> more positive selection -> rises)
  - curse      = reported s_ship - true q_ship              (the over-claim, axis A)
  - holdout    = a 2nd INDEPENDENT score s'=q+fresh-noise of the shipped one, minus true q (the axis-A FIX)
  - budget     = E[regeneration batches] = 1 / P(>=1 accepted)  (the over-verification cost, axis B)

PRE-REG: P1 the curse INCREASES with tau (stricter stringency = stronger positive selection = WORSE over-claim,
the OPPOSITE of 'more verification -> more honest') -> so tau does NOT fix axis A. P2 the held-out re-eval gives
curse~0 at EVERY tau (the axis-A fix is tau-INDEPENDENT -> orthogonal). P3 therefore #28's 'single optimal
amount' is UNDERSPECIFIED: you must set BOTH tau=beta* (axis B) AND use a held-out (axis A); the SDT criterion
unifies axis B but does NOT absorb axis A -> the c170 TWO-AXIS map is the correct frame (#28's unification REFUTED
as a single scalar). If instead curse DECREASES with tau and ->0 at the budget-optimal tau* -> #28 CONFIRMED.
"""
import numpy as np, json, os
from math import erf, sqrt
D = "runs/cycle186_one_amount_or_two_axes"; os.makedirs(D, exist_ok=True)
rng = np.random.default_rng(0)
MU, SIG_Q, SIG_V, K, REPS = 1.0, 1.0, 1.0, 20, 60000
TAUS = [-1.0, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]

def run(tau):
    q = rng.normal(MU, SIG_Q, (REPS, K))
    s = q + rng.normal(0, SIG_V, (REPS, K))
    smask = np.where(s >= tau, s, -np.inf)
    idx = smask.argmax(1); r = np.arange(REPS)
    valid = smask[r, idx] > -np.inf                    # at least one accepted this batch
    qsh, ssh = q[r, idx][valid], s[r, idx][valid]
    holdout = qsh + rng.normal(0, SIG_V, qsh.shape)    # 2nd INDEPENDENT verifier score of the shipped one
    p_acc = valid.mean()
    return {"tau": tau, "P_accept_per_batch": float(p_acc),
            "budget_batches": float(1.0 / p_acc) if p_acc > 0 else float("inf"),
            "true_ship": float(qsh.mean()),
            "curse_reported": float((ssh - qsh).mean()),          # axis A: over-claim using the SELECTION score
            "curse_holdout": float((holdout - qsh).mean()),       # axis A FIX: independent re-eval
            "n_valid": int(valid.sum())}

rows = [run(t) for t in TAUS]
# objective a naive 'optimal amount' picker would use: true_ship minus a budget cost (lambda per extra batch)
LAM = 0.1
for x in rows:
    x["net_obj"] = x["true_ship"] - LAM * (x["budget_batches"] - 1.0)
tau_star = max(rows, key=lambda x: x["net_obj"])["tau"]

print(f"=== #28: ONE amount or TWO axes? (K={K}, mu={MU}, sig_q={SIG_Q}, sig_v={SIG_V}, REPS={REPS}) ===")
print(f"{'tau':>5} {'P_acc':>7} {'budget':>7} {'true_ship':>10} {'curse(rep)':>11} {'curse(holdout)':>15} {'net(λ=.1)':>10}")
for x in rows:
    print(f"{x['tau']:>5} {x['P_accept_per_batch']:>7.3f} {x['budget_batches']:>7.2f} {x['true_ship']:>10.3f} "
          f"{x['curse_reported']:>11.3f} {x['curse_holdout']:>15.4f} {x['net_obj']:>10.3f}")

curse_rep = [x["curse_reported"] for x in rows]
curse_ho = [abs(x["curse_holdout"]) for x in rows]
p1 = all(b >= a - 1e-3 for a, b in zip(curse_rep, curse_rep[1:]))   # curse_reported non-decreasing in tau
p2 = max(curse_ho) < 0.01                                            # held-out curse ~0 at every tau
curse_at_taustar = [x["curse_reported"] for x in rows if x["tau"] == tau_star][0]
print(f"\nP1 curse_reported INCREASES (non-decreasing) with tau? {p1}  -> tau does NOT fix axis A")
print(f"P2 |curse_holdout| < 0.01 at EVERY tau (max={max(curse_ho):.4f})? {p2}  -> the axis-A fix is tau-INDEPENDENT (orthogonal)")
print(f"P3 budget-optimal tau*={tau_star} (net obj, λ=0.1) STILL has reported curse = {curse_at_taustar:.3f} (NOT ~0)")
print(f"    -> #28 'single optimal amount' {'REFUTED' if (p1 and p2 and curse_at_taustar>0.05) else 'NOT refuted'}: "
      f"SDT criterion sets axis B (tau*), but axis A (curse) needs the INDEPENDENT held-out -> c170 two-axis map confirmed.")
json.dump({"params": {"MU": MU, "SIG_Q": SIG_Q, "SIG_V": SIG_V, "K": K, "REPS": REPS, "LAMBDA": LAM},
           "rows": rows, "tau_star": tau_star, "P1_curse_rises_with_tau": bool(p1),
           "P2_holdout_fixes_curse_all_tau": bool(p2), "curse_at_tau_star": curse_at_taustar},
          open(f"{D}/score.json", "w"), indent=1)
print("DONE")
