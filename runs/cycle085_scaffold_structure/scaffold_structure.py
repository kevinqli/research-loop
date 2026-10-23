"""Cycle 85 (scaffold pillar — least-mined; diversify from verification toys). Is scaffold STRUCTURE a propagation-
bounding lever? Same N units of work, per-unit error prob p, errors propagate downstream until re-grounded (c82 model).
Compare structures:
  LINEAR        : one length-N chain (c82) — an error corrupts all downstream → wasted ≈ N − 1/p.
  TREE(B)       : B INDEPENDENT subtask-chains of length N/B, each verified at its end before integration — an error is
                  ISOLATED to its subtask (doesn't cross to siblings) and the subtask is a SHORT chain (less compounding).
  TREE+integration-cost: each subtask boundary costs c_int (coordination/merge overhead) — gives an interior optimum B*.
Pre-reg: tree wasted-work ≈ (N − B/p) decreasing in B (toward N·p at full parallelism) — decomposition bounds
propagation STRUCTURALLY, equivalent to temporal checkpointing (c82): a subtask boundary IS a checkpoint. With
integration cost there's an interior optimum B* (decompose, but not so finely that merge overhead dominates — echoes
c83/c35). NUANCE to state: structure helps WASTED-WORK/recovery+isolation, not the raw odds the FINAL answer is right
(that needs all work correct either way). numpy-only, REPS."""
import numpy as np, json
N,P,REPS = 60,0.05,20000
def chain_wasted(L,rng):
    bad=False; w=0
    for s in range(L):
        if rng.random()<P: bad=True
        w+=bad
    return w   # corrupted steps in a length-L chain, no internal checkpoint
def tree_wasted(B,rng):
    m=N//B; tot=0
    for _ in range(B): tot+=chain_wasted(m,rng)   # each subtask an independent short chain, verified at end (isolated)
    return tot
BS=[1,2,3,4,5,6,10,12,20,30,60]
res={}
for B in BS:
    rng=np.random.default_rng(300+B)
    w=np.mean([tree_wasted(B,rng) for _ in range(REPS)]) if B>1 else np.mean([chain_wasted(N,rng) for _ in range(REPS)])
    res[B]={"wasted":round(float(w),2)}
print(f"N={N} work units, per-unit error p={P}, REPS={REPS}. Wasted (corrupted) work vs decomposition B (B=1 linear):\n")
print(f"{'B (subtasks)':>12} {'subtask len':>11} {'wasted work':>12}")
for B in BS: print(f"{B:>12} {N//B:>11} {res[B]['wasted']:>12.2f}")
# with integration cost: total = wasted + c_int*B ; find B* for a few c_int
print(f"\n  with integration cost c_int per subtask boundary → interior optimum B*:")
opt={}
for c_int in [0.3,1.0,3.0]:
    tot={B:res[B]['wasted']+c_int*(B-1) for B in BS}
    Bstar=min(BS,key=lambda B:tot[B]); opt[c_int]={"Bstar":Bstar,"total":round(tot[Bstar],2)}
    print(f"    c_int={c_int:>4}: B*={Bstar:>2} (total {tot[Bstar]:.2f}) | "+" ".join(f'{B}:{tot[B]:.0f}' for B in [1,2,5,12,60]))
json.dump({"wasted_by_B":res,"opt_by_cint":opt,"N":N,"p":P},open("runs/cycle085_scaffold_structure/score.json","w"),indent=1)
print("\n--- read the rows above ---")
print(f"LINEAR (B=1) wasted {res[1]['wasted']:.1f} (≈ N−1/p = {N-1/P:.0f}); fully decomposed (B={N}) wasted {res[N]['wasted']:.1f} (≈ N·p = {N*P:.0f}). Decomposition cuts wasted work ~{res[1]['wasted']/max(res[N]['wasted'],.1):.0f}×.")
print("=> SCAFFOLD STRUCTURE is a propagation-bounding lever: decomposing into INDEPENDENT subtasks isolates errors (don't cross siblings) AND shortens each chain (less compounding) — STRUCTURALLY equivalent to temporal checkpointing (c82); a subtask boundary IS a checkpoint. With integration cost there's an interior-optimum granularity B* (echoes c83/c35). NUANCE: this helps wasted-work/recovery+isolation, NOT the raw odds the final answer is correct (needs all work right either way). 3rd way to fight long-horizon error: checkpoint (temporal) | decompose (structural) | both.")
print("DONE")
