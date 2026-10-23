"""Cycle 126 ADVERSARIAL VERIFY of P3 (a MIX beats pure-grounding under budget). Two attacks:
(1) cost-ratio r-sensitivity: is the interior-mix-wins result an artifact of r=4? sweep r in {2,4,8}.
(2) fresh-seed stability of the argmin + the mix-vs-grounding margin.
Also report the explicit MARGIN (mix vs pure-grounding) with non-overlapping-CI check -- the decision-relevant number."""
import numpy as np
L, B = 100, 40
SIG_J = 1.0

def even_positions(n, L):
    if n <= 0: return np.array([], dtype=int)
    return np.unique(((np.arange(n) + 0.5) * L / n).astype(int))

def sim_cost(n_L, n_G, p_J, delta, reps, seed_off):
    rng = np.random.default_rng(seed_off)
    gpos = set(even_positions(n_G, L).tolist())
    free = [i for i in range(L) if i not in gpos]
    lpos = set()
    if n_L > 0 and free:
        idx = np.unique(((np.arange(n_L) + 0.5) * len(free) / n_L).astype(int)); idx = idx[idx < len(free)]
        lpos = set(np.array(free)[idx].tolist())
    jumps = (rng.random((reps, L)) < p_J) * np.abs(rng.normal(0, SIG_J, (reps, L)))
    ja = np.zeros(reps); da = np.zeros(reps); cost = np.zeros(reps)
    for i in range(L):
        ja += jumps[:, i]; da += delta; cost += ja + da
        if i in gpos: ja[:] = 0.0; da[:] = 0.0
        elif i in lpos: ja[:] = 0.0
    return cost

def best_alloc(p_J, delta, r, reps, seed):
    rows = []
    for n_G in range(0, B // r + 1):
        c = sim_cost(B - r * n_G, n_G, p_J, delta, reps, seed)
        rows.append((n_G, B - r*n_G, c.mean(), c))
    best = min(rows, key=lambda x: x[2])
    pure_g = rows[-1]          # n_G = B//r, n_L=0  (pure grounding)
    pure_l = rows[0]           # n_G = 0 (pure local)
    return rows, best, pure_g, pure_l

def ci(x, seed=7, Bb=2000):
    rng = np.random.default_rng(seed); n=len(x)
    m = np.array([x[rng.integers(0,n,n)].mean() for _ in range(Bb)])
    return x.mean(), np.percentile(m,2.5), np.percentile(m,97.5)

print("=== ATTACK 1: cost-ratio r-sensitivity (p_J=0.15, delta=0.05) ===")
print(f"{'r':>3} | {'n_G*':>5} {'n_L*':>5} | {'mix cost':>9} | {'pure-ground':>11} | {'mix beats grounding?':>22}")
for r in [2, 4, 8]:
    rows, best, pg, pl = best_alloc(0.15, 0.05, r, reps=20000, seed=0)
    mm, mlo, mhi = ci(best[3]); gm, glo, ghi = ci(pg[3])
    interior = 0 < best[0] < B//r
    beats = mm < gm and mhi < glo     # mix strictly better, non-overlapping CI
    tag = f"{'YES' if beats else 'no'} ({(1-mm/gm)*100:.1f}% better, {'non-overlap' if mhi<glo else 'OVERLAP'})"
    print(f"{r:>3} | {best[0]:>5} {best[1]:>5} | {mm:>9.2f} | {gm:>11.2f} | {tag:>22}  {'[interior]' if interior else '[CORNER]'}")

print("\n=== ATTACK 2: fresh-seed stability of argmin (p_J=0.15, delta=0.05, r=4) ===")
print(f"{'seed':>5} | {'n_G*':>5} | {'mix cost':>9} | {'pure-ground cost':>16}")
ngs = []
for seed in [0, 1, 2, 12345, 99]:
    rows, best, pg, pl = best_alloc(0.15, 0.05, 4, reps=20000, seed=seed)
    ngs.append(best[0])
    print(f"{seed:>5} | {best[0]:>5} | {best[2]:>9.2f} | {pg[2]:>16.2f}")
print(f"  argmin n_G* across seeds = {ngs}  (range {min(ngs)}-{max(ngs)}; stable iff tight)")
print("\n=> P3 robust iff mix beats pure-grounding at ALL r (non-overlapping CI) AND argmin stable across seeds.")
print("DONE")
