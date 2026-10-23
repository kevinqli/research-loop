"""Cycle 82 (long-horizon TAIL × verification — the under-developed intersection). All prior verification work is
SINGLE-STEP selection; the tail problem is MULTI-STEP error COMPOUNDING: in a research chain, a step builds on the
previous one, so an uncaught error CORRUPTS everything downstream until a verification catches it. Question for a
long-horizon agent with a fixed verification budget B: WHERE to checkpoint? Model: L-step chain, each step independently
INTRODUCES an error w.p. p; a corrupted step stays corrupted (propagates) until a verification re-grounds it (recall 1);
verification at a step cleans the chain from there. Metric = expected #CORRUPTED (wasted) steps. Compare placements of B
checkpoints: none / final-only / back-loaded / front-loaded / EVEN-spaced.
Pre-reg: (1) unverified wasted work grows ~linearly→quadratically-in-cost with L; (2) 'verify only the final output' is
nearly as bad as none (everything's already corrupted — the check just tells you it's wrong); (3) EVEN spacing minimizes
wasted work (caps the propagation run-length); back-loaded ≫ front-loaded ≫ even. numpy-only, REPS."""
import numpy as np, json
L,P,REPS = 30,0.06,20000
def corrupted_steps(verify_set):
    """verify_set: set of step indices (0-based) that are checkpointed (cleaned AFTER the step's work)."""
    rng=corrupted_steps.rng
    intro=rng.random(L)<P                       # step introduces an error
    corrupt=np.zeros(L,bool); bad=False
    for k in range(L):
        if intro[k]: bad=True
        corrupt[k]=bad
        if k in verify_set: bad=False           # checkpoint re-grounds: downstream starts clean
    return corrupt.sum()
def schedule(kind,B):
    if B<=0: return set()
    if kind=="final": return {L-1}
    if kind=="back": return set(range(L-B,L))
    if kind=="front": return set(range(B))
    if kind=="even": return set(np.linspace(0,L-1,B+2)[1:-1].round().astype(int).tolist())
    return set()
out={}
for B in [0,1,3,6]:
    out[B]={}
    for kind in (["none"] if B==0 else (["final"] if B==1 else ["back","front","even"])):
        tot=0; rng=np.random.default_rng(900+B*7+hash(kind)%50); corrupted_steps.rng=rng
        vs=schedule("final" if (B==1) else ("none" if B==0 else kind),B)
        for r in range(REPS): tot+=corrupted_steps(vs)
        out[B][kind]=round(tot/REPS,3)
print(f"L={L} steps, per-step error prob p={P}, REPS={REPS}. Expected #CORRUPTED (wasted) steps by checkpoint placement:\n")
print(f"  B=0 (no verification): {out[0]['none']:.2f} wasted steps  ({100*out[0]['none']/L:.0f}% of the chain)")
print(f"  B=1 final-output-only : {out[1]['final']:.2f}  (catches that output is wrong, but everything's already corrupted)")
for B in [3,6]:
    print(f"  B={B}:  back-loaded {out[B]['back']:.2f}   front-loaded {out[B]['front']:.2f}   EVEN-spaced {out[B]['even']:.2f}")
# compounding vs chain length (no verification)
comp={}
for Lx in [5,10,20,40,80]:
    rng=np.random.default_rng(123); corrupted_steps.rng=rng
    g=L; L=Lx; tot=sum(corrupted_steps(set()) for _ in range(5000)); comp[Lx]=round(tot/5000,2); L=g
json.dump({"by_budget":out,"compounding_vs_L":comp,"L":L,"p":P},open("runs/cycle082_chain_checkpoints/score.json","w"),indent=1)
print(f"\n  COMPOUNDING (no verification) — wasted steps vs chain length L: {comp}")
print("\n--- read the rows above ---")
e6,b6,f6=out[6]['even'],out[6]['back'],out[6]['front']
print(f"At B=6: EVEN {e6:.2f} wasted vs back-loaded {b6:.2f} vs front-loaded {f6:.2f}; final-only(B=1) {out[1]['final']:.2f} ~ none {out[0]['none']:.2f}.")
print("=> LONG-HORIZON RULE: CHECKPOINT THROUGHOUT (even-spaced) to cap error-propagation run-length — NOT just the final output (which is nearly as wasteful as no verification: the error already corrupted everything). Wasted work grows with chain length; verification's job in the tail is to BOUND propagation, not certify the end.")
print("DONE")
