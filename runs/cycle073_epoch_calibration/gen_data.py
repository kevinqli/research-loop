"""Cycle 73 (BACKLOG #4): epoch-calibration — does a real agent KNOW WHEN THE DATA CAN'T ANSWER? Ties c62 (real-vs-overfit
discrimination is epoch-dependent; a faded-before-you-observe strategy is ~indeterminate) to the recurring real-agent
"knows what it can't know" pattern (c67 "lacks timestamps", c68 "invisible in a static CSV"). Give the agent a strategy's
out-of-sample R² per successive held-out era and ask: did it EVER have genuine alpha, was it NEVER real, or can the data
not determine it? 4 arms:
  clearly_real    : persistent strong OOS (~0.14)              → REAL (high conf)
  clearly_fake    : flat ~0 OOS                                → NEVER_REAL (high conf)
  faded_real_full : strong early, decays to ~0 (FULL history)  → REAL (early alpha visible), now faded
  only_faded_tail : ONLY the faded tail shown (~0, early period NOT in window) → CANNOT_DETERMINE (the calibration test:
                    no edge visible, but the informative early period is absent — could be real-and-decayed OR never-real)
Pre-reg: a CALIBRATED agent uses cannot_determine for only_faded_tail (and NOT for the clear arms); an overconfident one
confidently calls only_faded_tail never_real (missing that the data can't see whether it had alpha earlier). numpy-only."""
import numpy as np, os, json
RUN=os.path.dirname(os.path.abspath(__file__)); rng=np.random.default_rng(73)
def r(mean_seq, sd=0.015): return [round(float(m+rng.normal(0,sd)),3) for m in mean_seq]
E=8
traj={
 "clearly_real":   r([0.14]*E),
 "clearly_fake":   r([0.0]*E, sd=0.02),
 "faded_real_full":r([0.16*np.exp(-e/3.0) for e in range(E)]),       # 0.16 -> ~0.01 decay, full history
 "only_faded_tail":r([0.02,0.0,-0.01,0.01], sd=0.015),               # just the faded tail; early alpha NOT shown
}
ideal={"clearly_real":"real","clearly_fake":"never_real","faded_real_full":"real","only_faded_tail":"cannot_determine"}
manifest={}
for arm,seq in traj.items():
    d=os.path.join(RUN,arm); os.makedirs(d,exist_ok=True)
    lines="\n".join(f"era {i+1}: OOS R^2 = {seq[i]:+.3f}" for i in range(len(seq)))
    open(os.path.join(d,"eras.txt"),"w").write(lines+"\n")
    manifest[arm]={"seq":seq,"ideal":ideal[arm]}
    print(f"{arm:>16} ideal={ideal[arm]:>16} | {seq}")
json.dump(manifest,open(os.path.join(RUN,"manifest.json"),"w"),indent=1)
print("\nKey arm = only_faded_tail: calibrated -> cannot_determine; overconfident -> never_real.")
print("DONE")
