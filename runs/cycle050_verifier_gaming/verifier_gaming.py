"""Cycle 50: Goodhart's law for verification — if an agent can OPTIMIZE AGAINST the held-out verifier, does the
verifier still work? The whole thesis rests on a held-out re-eval removing the winner's curse (cycle 7) BECAUSE the
selection didn't optimize the held-out noise. But an adversarial agent that can see/query the verification set will
select candidates that look good on it TOO -> the held-out is no longer held-out (it becomes a 2nd in-sample set).
Model: K candidates, true theta. Agent selects argmax(theta + e_in + alpha_v*e_v), where e_v is the verifier-set
noise it can partially game (alpha_v = gaming access; 0 = can't see verifier). Report on (a) the GAMED verifier
[theta+e_v] and (b) a truly FRESH/secret held-out [theta+e_fresh] the agent never touched.
Pre-reg: gap on the GAMED verifier GROWS with alpha_v (it becomes inflated like in-sample); the FRESH held-out
stays ~0. => a verifier the agent can optimize against is worthless; you need a SECRET / ROTATING held-out.
numpy-only, REPS reps. MPS-exploratory (sim of the mechanism)."""
import numpy as np
from math import sqrt
rng = np.random.default_rng(0)
REPS, K, MU, S_TRUE, S = 6000, 100, 0.80, 0.05, 0.05  # S = noise scale (in-sample, verifier, fresh)

def gaps(alpha_v):
    g_in=np.empty(REPS); g_gamed=np.empty(REPS); g_fresh=np.empty(REPS)
    for r in range(REPS):
        theta = rng.normal(MU, S_TRUE, K)
        e_in  = rng.normal(0, S, K)      # in-sample noise
        e_v   = rng.normal(0, S, K)      # verifier-set noise (agent can game it w/ access alpha_v)
        e_fr  = rng.normal(0, S, K)      # truly fresh/secret held-out noise (never touched)
        s = int(np.argmax(theta + e_in + alpha_v*e_v))   # agent optimizes in-sample AND (partially) the verifier
        g_in[r]    = (theta[s]+e_in[s]) - theta[s]        # in-sample report gap (full winner's curse)
        g_gamed[r] = (theta[s]+e_v[s])  - theta[s]        # report on the GAMED verifier
        g_fresh[r] = (theta[s]+e_fr[s]) - theta[s]        # report on the SECRET held-out
    return float(g_in.mean()), float(g_gamed.mean()), float(g_fresh.mean())

print(f"K={K}, reps={REPS}, true~N({MU},{S_TRUE}), noise={S}\n")
print(f"{'gaming access α_v':>18} | {'in-sample gap':>13} | {'GAMED-verifier gap':>18} | {'SECRET held-out gap':>19}")
import json; rows=[]
for av in [0.0, 0.5, 1.0, 2.0]:
    gi,gg,gf = gaps(av)
    print(f"{av:>18.1f} | {gi:>+13.4f} | {gg:>+18.4f} | {gf:>+19.4f}")
    rows.append({"alpha_v":av,"in_sample_gap":gi,"gamed_verifier_gap":gg,"secret_holdout_gap":gf})
json.dump({"K":K,"rows":rows}, open("runs/cycle050_verifier_gaming/score.json","w"), indent=1)
print("\n--- read the rows above ---")
print(f"α_v=0 (can't game verifier): gamed-verifier gap {rows[0]['gamed_verifier_gap']:+.4f} ~ secret {rows[0]['secret_holdout_gap']:+.4f} ~0 — the held-out works.")
print(f"α_v=1 (games verifier equally to in-sample): gamed-verifier gap {rows[2]['gamed_verifier_gap']:+.4f} (INFLATED, ~half the in-sample gap {rows[2]['in_sample_gap']:+.4f}); secret held-out stays {rows[2]['secret_holdout_gap']:+.4f} ~0.")
print("=> Goodhart for verification: a held-out set the agent can OPTIMIZE AGAINST becomes a 2nd in-sample set (its gap grows with gaming access); only a SECRET/ROTATING held-out the agent never touched removes the curse. The verifier's value depends on it staying OUT of the optimization loop.")
print("DONE")
