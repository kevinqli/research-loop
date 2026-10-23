"""Cycle 86: the missing DIMENSION of the verification thesis — not WHETHER to verify but HOW STRINGENTLY. Optimal
accept/reject threshold τ* under ASYMMETRIC error costs (false-positive = ship a fake, cost C_FP; false-negative =
discard a real, cost C_FN) and a FAKE BASE-RATE φ. Decision-theoretic core (Bayes threshold) — the contribution is the
SYNTHESIS: (a) τ* rises with both the cost ratio C_FP/C_FN and φ; (b) the WINNER'S CURSE (c7) raises φ in the accepted/
top pool at scale, so verification should get MORE stringent as K grows; (c) this RATIONALIZES c47 (adversarial 'refute'
framing = a high-recall/low-precision conservative filter that rejected reals too) — it's OPTIMAL exactly in the high-φ,
high-C_FP regime (research at scale: shipping a fabricated finding ≫ missing a real one). Permissive only when FN
dominates (exploration). numpy-only, REPS.  Pool: φ fake (true q≈0), 1−φ real (q~U(.3,.7)); verifier signal s=q+N(0,σ)."""
import numpy as np, json
REPS,SIG = 60000,0.25
def best_tau(rho,phi):   # rho=C_FP/C_FN; minimize expected cost over τ; return τ*, and precision/recall at τ*
    rng=np.random.default_rng(int(rho*1000)+int(phi*1000))
    fake=rng.random(REPS)<phi
    q=np.where(fake,0.0,rng.uniform(.3,.7,REPS)); s=q+rng.normal(0,SIG,REPS)
    taus=np.linspace(-.2,1.0,121); best=(1e9,None,None,None)
    for t in taus:
        acc=s>t
        fp=np.mean(acc&fake); fn=np.mean((~acc)&(~fake))     # rates over the pool
        cost=rho*fp+1.0*fn                                    # C_FN=1, C_FP=rho
        if cost<best[0]:
            tp=np.sum(acc&~fake); prec=tp/max(np.sum(acc),1); rec=tp/max(np.sum(~fake),1)
            best=(cost,round(float(t),3),round(float(prec),3),round(float(rec),3))
    return best
RHOS=[0.5,1,3,10,30]; PHIS=[0.2,0.5,0.8]
out={}
print(f"σ={SIG}, REPS={REPS}. Optimal accept threshold τ* (stringency) by cost-ratio C_FP/C_FN × fake base-rate φ:\n")
print(f"{'C_FP/C_FN':>10} | " + " ".join(f"φ={p}".rjust(10) for p in PHIS) + "   (τ* / precision@τ* / recall@τ*)")
for rho in RHOS:
    cells=[]
    for phi in PHIS:
        c,t,pr,rc=best_tau(rho,phi); out[f"rho{rho}_phi{phi}"]={"tau":t,"prec":pr,"rec":rc}
        cells.append(f"{t:.2f}/{pr:.2f}/{rc:.2f}")
    print(f"{rho:>10} | " + " ".join(c.rjust(10) for c in cells))
json.dump(out,open("runs/cycle086_stringency/score.json","w"),indent=1)
print("\n--- read the rows above ---")
lo=out["rho0.5_phi0.2"]["tau"]; hi=out["rho30_phi0.8"]["tau"]
print(f"τ* (stringency) ranges from {lo:+.2f} (cheap fakes + rare fakes: PERMISSIVE) to {hi:+.2f} (costly fakes + common fakes: STRINGENT).")
print(f"At high C_FP × high φ (research at scale): τ* high → accept few, precision↑/recall↓ — i.e. the c47 conservative filter (rejects reals too) is OPTIMAL there, not a bug.")
print("=> STRINGENCY dimension: verify more STRINGENTLY as (i) shipping-a-fake gets costlier and (ii) the fake base-rate rises. The winner's curse (c7) RAISES the fake rate in the top pool at scale → so a parallel-agent lab should tighten verification as it scales K. Be permissive only when missing-a-real dominates (exploration).")
print("DONE")
