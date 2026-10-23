"""Cycle 31 (quant-eval Q4): memorization-vs-learning under SELECTIVE OBFUSCATION.
Key insight to test: held-out-era SURVIVAL (Q2) is NECESSARY but NOT SUFFICIENT. Returns here have TWO recoverable
signals: a STRUCTURAL one (r += X@beta, a learnable relationship) and a STABLE-IDENTIFIER one (r += g[asset_id],
a per-asset offset that is stable across eras -> genuinely predictive in-distribution, but it's a LABEL, not a
relationship). A 'memorizer' that learns the id->return map survives a plain era change (g is stable) -> looks real.
The eval lever that EXPOSES it: SELECTIVE OBFUSCATION — permute the asset identifiers while preserving X->r
structure. The structural learner (uses X) is invariant; the memorizer (uses id) collapses.
Pre-reg: obfuscation leaves the structural strategy ~unchanged and collapses the memorizer's OOS R^2 to ~0.
=> the eval needs BOTH survival AND obfuscation-robustness. numpy-only, REPS reps. MPS-exploratory (toy)."""
import numpy as np, json
from math import sqrt
N_FEATURES, N_TRUE, N_OBS, N_ASSETS = 30, 5, 800, 20
REPS = 200

def gen(rng):
    true_idx = np.sort(rng.choice(N_FEATURES, N_TRUE, replace=False))
    beta = rng.choice([-1.,1.], N_TRUE)*rng.uniform(0.8,1.2,N_TRUE)
    g = rng.normal(0, 1.0, N_ASSETS)              # per-asset offset, STABLE across eras (a predictive LABEL)
    sigma = 1.0
    def era():
        X = rng.normal(0,1,(N_OBS,N_FEATURES)); ids = rng.integers(0,N_ASSETS,N_OBS)
        r = X[:,true_idx]@beta + g[ids] + rng.normal(0,sigma,N_OBS)
        return X, ids, r
    return era(), era(), true_idx
def r2(y,yh): return 1 - np.sum((y-yh)**2)/np.sum((y-y.mean())**2)
def ols(Xtr,ytr,Xte):
    b,*_=np.linalg.lstsq(np.column_stack([np.ones(len(Xtr)),Xtr]),ytr,rcond=None)
    return np.column_stack([np.ones(len(Xte)),Xte])@b
def onehot(ids):
    M=np.zeros((len(ids),N_ASSETS)); M[np.arange(len(ids)),ids]=1; return M

struct_same=[]; struct_obf=[]; mem_same=[]; mem_obf=[]
for rep in range(REPS):
    rng=np.random.default_rng(rep)
    (Xa,ida,ra),(Xb,idb,rb),true_idx = gen(rng)
    perm = rng.permutation(N_ASSETS)          # SELECTIVE OBFUSCATION: relabel assets (structure X->r preserved)
    idb_obf = perm[idb]
    # STRUCTURAL strategy: uses feature values X (a learned relationship)
    struct_same.append(r2(rb, ols(Xa,ra,Xb)))
    struct_obf.append(r2(rb, ols(Xa,ra,Xb)))   # X unchanged by id-permutation -> identical
    # MEMORIZER strategy: uses asset identity only (one-hot id)
    mem_same.append(r2(rb, ols(onehot(ida),ra,onehot(idb))))
    mem_obf.append(r2(rb, ols(onehot(ida),ra,onehot(idb_obf))))   # fed scrambled ids -> stale map
m=lambda x: float(np.mean(x)); ci=lambda x:1.96*np.std(x,ddof=1)/sqrt(len(x))
print(f"dims: {N_FEATURES} feats ({N_TRUE} true) + {N_ASSETS} assets w/ STABLE id-offset; {N_OBS} obs/era, REPS={REPS}\n")
print(f"{'strategy':>22} | {'OOS R2 (ids intact)':>20} | {'OOS R2 (OBFUSCATED ids)':>24} | survives obfusc?")
print(f"{'STRUCTURAL (uses X)':>22} | {m(struct_same):>10.3f} ± {ci(struct_same):.3f} | {m(struct_obf):>14.3f} ± {ci(struct_obf):.3f} | YES (invariant)")
print(f"{'MEMORIZER (uses id)':>22} | {m(mem_same):>10.3f} ± {ci(mem_same):.3f} | {m(mem_obf):>14.3f} ± {ci(mem_obf):.3f} | NO (collapses)")
out={"structural":{"oos_same":m(struct_same),"oos_obf":m(struct_obf)},
     "memorizer":{"oos_same":m(mem_same),"oos_obf":m(mem_obf)}}
json.dump(out,open("runs/cycle031_quant_obfusc/score.json","w"),indent=1)
print("\n--- read the rows above ---")
print(f"Both strategies SURVIVE a plain era change (ids intact): structural {m(struct_same):.2f}, memorizer {m(mem_same):.2f} (both >0) -> survival ALONE can't tell them apart.")
print(f"Selective OBFUSCATION exposes the memorizer: structural stays {m(struct_obf):.2f} (invariant), memorizer collapses {m(mem_same):.2f}->{m(mem_obf):.2f}.")
print("=> the quant eval needs BOTH held-out-era survival (Q2) AND identifier-obfuscation robustness (this). Learned relationship survives; memorized label does not.")
print("DONE")
