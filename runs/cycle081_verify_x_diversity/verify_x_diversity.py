"""Cycle 81 (fleet design): does the OUTCOME value of verification INTERACT with candidate DIVERSITY?
Synthesizes c35 (interior search/verify split), c48 (redundancy worsens report-curse but has low outcome-regret), c51
(diversify-don't-replicate). The specific NEW claim: a held-out re-eval recovers MORE TRUE PERFORMANCE when candidates
are DIVERSE (low correlation ρ) — because diverse candidates have a real 'best' to find, so picking the in-sample
noise-winner costs real outcome; redundant candidates are ~interchangeable, so verification buys little outcome.
Setup: K candidates, true θ_i correlated at ρ (diversity = 1−ρ); in-sample metric = θ_i + noise; verification =
re-eval top-3 on an INDEPENDENT held-out, pick best by that. Outcome = true θ of finally-selected. Sweep ρ.
Pre-reg: outcome GAIN (verified − in-sample-selected) rises as ρ falls (diversity). numpy-only, REPS."""
import numpy as np, json
from math import sqrt
K,REPS,SIG_IN,SIG_HO = 100,4000,1.0,1.0   # candidates, reps, in-sample & held-out noise (sd, in θ units)
RHOS=[0.0,0.3,0.6,0.9,0.99]
def one(rho,rng):
    # true θ: common factor f + idiosyncratic, corr(θ_i,θ_j)=rho; θ ~ N(0,1)
    f=rng.normal(0,1); theta=sqrt(rho)*f+sqrt(1-rho)*rng.normal(0,1,K)
    insample=theta+rng.normal(0,SIG_IN,K)                    # noisy self-reported metric
    sel_IS=int(np.argmax(insample))                          # naive: pick best in-sample
    top3=np.argsort(insample)[-3:]                           # verify: re-eval top-3 on independent held-out
    ho=theta[top3]+rng.normal(0,SIG_HO,3)
    sel_V=int(top3[np.argmax(ho)])
    oracle=float(theta.max())
    return theta[sel_IS],theta[sel_V],oracle
out={}
print(f"K={K}, REPS={REPS}, in/held-out noise sd={SIG_IN}/{SIG_HO}. Outcome (TRUE θ of selected) vs candidate correlation ρ:\n")
print(f"{'ρ (corr)':>9} {'diversity':>9} | {'in-sample sel':>13} {'VERIFIED sel':>13} {'GAIN':>7} | {'oracle':>7} {'regret_IS':>9}")
for rho in RHOS:
    IS,V,OR=[],[],[]
    for r in range(REPS):
        a,b,o=one(rho,np.random.default_rng(8000+r+int(rho*1e6))); IS.append(a); V.append(b); OR.append(o)
    IS,V,OR=np.mean(IS),np.mean(V),np.mean(OR); gain=V-IS
    out[rho]={"in_sample":float(IS),"verified":float(V),"gain":float(gain),"oracle":float(OR),"regret_IS":float(OR-IS)}
    print(f"{rho:>9.2f} {1-rho:>9.2f} | {IS:>13.3f} {V:>13.3f} {gain:>+7.3f} | {OR:>7.3f} {OR-IS:>9.3f}")
json.dump(out,open("runs/cycle081_verify_x_diversity/score.json","w"),indent=1)
print("\n--- read the rows above ---")
g_div,g_red=out[0.0]["gain"],out[0.99]["gain"]
print(f"verification OUTCOME gain: diverse (ρ=0) {g_div:+.3f}  vs  redundant (ρ=0.99) {g_red:+.3f}.")
print(f"=> verification recovers {'MORE' if g_div>g_red else 'less'} true performance when candidates are DIVERSE (gain {g_div:.3f} vs {g_red:.3f}); redundant candidates have low regret so verification buys little OUTCOME.")
print("=> FLEET RULE: diversify the fleet (c48/51) AND — because you diversified — VERIFY MORE; the verify-fraction's OUTCOME payoff scales with diversity. (Redundant fleets still need verification for an HONEST report (c48), just not for outcome.)")
print("DONE")
