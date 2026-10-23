"""Cycle 159 (#25 — CAN a sequential-stop agent debias its OWN shipped result without a held-out? COMPOSE c157xc132xc151;
closes the c157(mechanism)->c158(grounding)->c159(the fix) centerpiece mini-arc). c157: stop at first observed>tau,
ship it -> inflated by the truncation curse (E[true|o1] = v/(v+sig^2)*o1, the Bayes shrinkage, c132). The debiasing
SHRINKAGE factor is v/(v+sig^2) [v = signal/prior variance, sig^2 = noise variance]. QUESTION: can the agent recover
that factor from its OWN run? From its observed sequence it estimates only the TOTAL Var(observed)=v+sig^2 — NOT the
v-vs-sig^2 SPLIT. So self-correction needs to KNOW v (or the split); otherwise it needs an independent 2nd draw (c151).

ARMS on the stopped candidate (true q~N(0,v), observed o = q + N(0,sig^2), stop at first o>tau):
  NAIVE        : ship o1 (the curse, c157).
  SHRINK_KNOWN : o1 * v/(v+sig^2)  [agent KNOWS v and the total -> the Bayes posterior mean, c132].
  SHRINK_WRONGV: o1 * v_wrong/(v+sig^2)  [agent has the total but GUESSES v wrong -> biased: v NOT recoverable from the total].
  HELDOUT_2ND  : o2 = a FRESH independent observation of the SAME stopped candidate [c151/c7: unbiased, PARAMETER-FREE].

PRE-REG: P1 NAIVE inflated (=c157). P2 SHRINK_KNOWN ~ true (debiases). P3 SHRINK_WRONGV biased (v not recoverable from
the total alone -> self-correction REQUIRES knowing v). P4 HELDOUT_2ND ~ true, parameter-free (the robust fix when v
unknown). => the sequential-stop curse is self-correctable IFF you know the signal variance; else the held-out (2nd
draw) is load-bearing (c132's informational limit, sequential version; confirms the thesis).
"""
import numpy as np, json, os
os.makedirs("runs/cycle159_seqstop_selfdebias", exist_ok=True)
rng = np.random.default_rng(0)

V = 1.0                     # true signal/prior variance (the agent may or may not know this)
SIGMA2 = 0.36               # noise variance (sigma_obs=0.6); TOTAL Var(observed) = V+SIGMA2 = 1.36
V_WRONG = 0.5               # the agent's wrong guess for v (still consistent with the same total)
TAU_VALUES = [0.5, 1.0, 1.5, 2.0]
REPS = 40000
N_MAX = 20000
shrink_known = V / (V + SIGMA2)
shrink_wrong = V_WRONG / (V + SIGMA2)

def run(tau, reps):
    o1s, trues, o2s = [], [], []
    sig = np.sqrt(SIGMA2); sv = np.sqrt(V)
    for _ in range(reps):
        for _ in range(N_MAX):
            q = sv * rng.standard_normal()
            o = q + sig * rng.standard_normal()
            if o > tau:
                o1s.append(o); trues.append(q); o2s.append(q + sig * rng.standard_normal()); break
    o1 = np.array(o1s); t = np.array(trues); o2 = np.array(o2s)
    def m(a): return float(a.mean())
    def ci(a): return float(1.96 * a.std(ddof=1) / np.sqrt(len(a)))
    return {"true": [m(t), ci(t)], "naive_o1": [m(o1), ci(o1)],
            "shrink_known": [m(o1 * shrink_known), ci(o1 * shrink_known)],
            "shrink_wrongv": [m(o1 * shrink_wrong), ci(o1 * shrink_wrong)],
            "heldout_o2": [m(o2), ci(o2)],
            "err_naive": m(o1) - m(t), "err_shrink_known": m(o1 * shrink_known) - m(t),
            "err_shrink_wrongv": m(o1 * shrink_wrong) - m(t), "err_heldout": m(o2) - m(t)}

out = {"V": V, "SIGMA2": SIGMA2, "total_var": V + SIGMA2, "shrink_known_factor": shrink_known,
       "V_WRONG": V_WRONG, "shrink_wrong_factor": shrink_wrong, "TAU_values": TAU_VALUES, "REPS": REPS,
       "separability_note": "Var(observed)=V+SIGMA2=1.36 is all the own-run sequence gives; (V,SIGMA2)=(1,0.36) and (0.5,0.86) have the SAME total but shrinkage 0.735 vs 0.368 -> v NOT recoverable from the total.",
       "by_tau": {}}
print(f"=== #25: self-debias the sequential-stop curse (V={V}, sig^2={SIGMA2}, total={V+SIGMA2}, REPS={REPS}) ===")
print(f"shrinkage factor v/(v+sig^2): KNOWN-v={shrink_known:.3f}  WRONG-v({V_WRONG})={shrink_wrong:.3f}")
print(f"{'tau':>5} {'true':>14} {'naive(o1)':>14} {'shrink_known':>14} {'shrink_wrongv':>14} {'heldout(o2)':>14}")
for tau in TAU_VALUES:
    r = run(tau, REPS); out["by_tau"][tau] = r
    print(f"{tau:>5} {r['true'][0]:>8.3f}+-{r['true'][1]:<5.3f} {r['naive_o1'][0]:>8.3f}+-{r['naive_o1'][1]:<5.3f}"
          f" {r['shrink_known'][0]:>8.3f}+-{r['shrink_known'][1]:<5.3f} {r['shrink_wrongv'][0]:>8.3f}+-{r['shrink_wrongv'][1]:<5.3f}"
          f" {r['heldout_o2'][0]:>8.3f}+-{r['heldout_o2'][1]:<5.3f}")
json.dump(out, open("runs/cycle159_seqstop_selfdebias/score.json", "w"), indent=1)
print("--- errors vs true (tau=1.5) ---")
r = out["by_tau"][1.5]
print(f"naive {r['err_naive']:+.3f} | shrink_known {r['err_shrink_known']:+.3f} | shrink_WRONGv {r['err_shrink_wrongv']:+.3f} | heldout {r['err_heldout']:+.3f}")
print("PRE-REG: P1 naive inflated; P2 shrink_known~true; P3 shrink_wrongv BIASED (v not recoverable from total); P4 heldout~true param-free.")
print("DONE")
