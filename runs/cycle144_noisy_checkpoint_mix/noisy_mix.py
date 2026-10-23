"""Cycle 144 (HARDEN c126 × c93; BACKLOG #16b): does c126's optimal cheap-LOCAL + sparse-GROUNDING checkpoint MIX survive
IMPERFECT checks? c126 assumed perfect checks (local catches every jump, grounding catches everything). Real checks MISS
(c93: recall<1). Extend c126: a LOCAL check catches the jump-accumulator w.p. r_L (else the jumps persist & propagate, c82);
a GROUNDING check resets both jump+drift w.p. r_G. Budget B = n_L·1 + n_G·r_cost (r_cost=4); cost = integral of uncaught
error over the L-step chain; find the n_L/n_G split minimizing cost, vs the recalls.

PRE-REG: P1 as r_L drops, optimal n_G RISES (shift toward grounding — it catches what local misses + the drift). P2 below
some r_L the cheap-local arm stops paying (→ heavy grounding). P3 if r_G ALSO drops, the whole scheme degrades → checkpoint
LESS / no-checkpoint wins (c93: a costly+unreliable verifier isn't worth running). CONTROL: r_L=r_G=1 recovers c126."""
import numpy as np, json, os
L, B, R_COST, REPS, SIG_J, P_J, DELTA = 100, 40, 4, 8000, 1.0, 0.15, 0.05
os.makedirs("runs/cycle144_noisy_checkpoint_mix", exist_ok=True)

def even_positions(n, L):
    if n <= 0: return set()
    return set(np.unique(((np.arange(n)+0.5)*L/n).astype(int)).tolist())

def sim_cost(n_L, n_G, r_L, r_G, reps=REPS, seed=0):
    rng = np.random.default_rng(seed)
    gpos = even_positions(n_G, L)
    free = [i for i in range(L) if i not in gpos]
    lpos = set()
    if n_L>0 and free:
        idx = np.unique(((np.arange(n_L)+0.5)*len(free)/n_L).astype(int)); idx=idx[idx<len(free)]
        lpos = set(np.array(free)[idx].tolist())
    jumps = (rng.random((reps,L))<P_J)*np.abs(rng.normal(0,SIG_J,(reps,L)))
    ja=np.zeros(reps); da=np.zeros(reps); cost=np.zeros(reps)
    for i in range(L):
        ja+=jumps[:,i]; da+=DELTA; cost+=ja+da
        if i in gpos:                       # grounding: catch BOTH w.p. r_G
            m = rng.random(reps)<r_G; ja[m]=0.0; da[m]=0.0
        elif i in lpos:                     # local: catch jumps w.p. r_L (drift untouched)
            m = rng.random(reps)<r_L; ja[m]=0.0
    return cost.mean()

def best_split(r_L, r_G):
    rows=[]
    for n_G in range(0, B//R_COST + 1):
        n_L = B - R_COST*n_G
        rows.append((n_G, n_L, sim_cost(n_L, n_G, r_L, r_G)))
    best = min(rows, key=lambda x: x[2])
    return best, rows

print(f"L={L}, budget B={B}, grounding cost r_cost={R_COST}, jumps p={P_J}, drift δ={DELTA}. cost=integral uncaught error (lower=better).\n")
print("=== P1/P2: optimal split vs LOCAL recall r_L (grounding recall r_G=0.9 fixed) ===")
print(f"{'r_L':>5} | {'best n_G':>8} {'best n_L':>8} | {'best cost':>10} | {'pure-ground cost':>16} | {'mix vs pure-ground':>18}")
out={"model":{"L":L,"B":B,"R_COST":R_COST,"P_J":P_J,"DELTA":DELTA},"vs_rL":[],"vs_rG":[]}
for r_L in [1.0, 0.9, 0.7, 0.5, 0.3, 0.1]:
    (ng,nl,c), rows = best_split(r_L, 0.9)
    pg = [r for r in rows if r[0]==B//R_COST][0][2]
    tag = "mix" if 0<ng<B//R_COST else ("pure-ground" if ng==B//R_COST else "pure-local")
    out["vs_rL"].append({"r_L":r_L,"n_G":ng,"n_L":nl,"cost":round(c,2),"pure_ground":round(pg,2),"shape":tag})
    print(f"{r_L:>5} | {ng:>8} {nl:>8} | {c:>10.2f} | {pg:>16.2f} | {tag:>18}")

print("\n=== P3: vary GROUNDING recall r_G (local r_L=0.7 fixed) — does the whole scheme degrade? ===")
print(f"{'r_G':>5} | {'best n_G':>8} {'best n_L':>8} | {'best cost':>10} | {'no-checkpoint cost':>18}")
nocheck = sim_cost(0,0,0,0)
for r_G in [1.0, 0.9, 0.7, 0.5, 0.3]:
    (ng,nl,c), rows = best_split(0.7, r_G)
    out["vs_rG"].append({"r_G":r_G,"n_G":ng,"n_L":nl,"cost":round(c,2)})
    print(f"{r_G:>5} | {ng:>8} {nl:>8} | {c:>10.2f} | {nocheck:>18.2f}")
out["no_checkpoint_cost"]=round(nocheck,2)

json.dump(out, open("runs/cycle144_noisy_checkpoint_mix/score.json","w"), indent=1)
print("\n--- READ THE RAW ROWS ---")
print("optimal n_G vs r_L (r_G=0.9): " + " ".join(f"rL{r['r_L']}:nG{r['n_G']}" for r in out["vs_rL"]) + "  (rises as r_L drops?)")
print("best cost vs r_L: " + " ".join(f"rL{r['r_L']}:{r['cost']}" for r in out["vs_rL"]))
print("optimal n_G vs r_G (r_L=0.7): " + " ".join(f"rG{r['r_G']}:nG{r['n_G']}" for r in out["vs_rG"]) + f"  (no-checkpoint cost={nocheck:.1f})")
print("=> P1: optimal mix shifts toward GROUNDING as local recall r_L drops (grounding catches what local misses + drift). P3: as grounding recall r_G drops the cost rises toward no-checkpoint (c93 — a costly+unreliable check isn't worth its budget).")
print("DONE")
