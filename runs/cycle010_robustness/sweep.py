"""Robustness of the winner's-curse + cheap-top-3 recipe across rho = s_val/s_true (validation noise / true spread).
For each rho at K=100: optimism gap of naive top-1, and the smallest m whose verify-top-m true_perf captures >=90%
of the achievable regret-cut. Tests whether 'top ~3' generalizes. 3000 reps. numpy only."""
import numpy as np
rng=np.random.default_rng(0); REPS=3000; K=100; S_TRUE=0.05; MU=0.80
print(f"K={K}, reps={REPS}, s_true={S_TRUE}\n  rho=s_val/s_true | naive_gap | gap/s_true | m for >=90% regret-cut | true@bestm")
for rho in [0.5,1.0,2.0,4.0]:
    s_val=rho*S_TRUE
    gaps=np.empty(REPS); true_by_m={m:np.empty(REPS) for m in range(1,31)}; naive_true=np.empty(REPS); best_true=np.empty(REPS)
    for r in range(REPS):
        th=rng.normal(MU,S_TRUE,K); val=th+rng.normal(0,s_val,K); order=np.argsort(-val)
        s=order[0]; gaps[r]=val[s]-th[s]; naive_true[r]=th[s]; best_true[r]=th.max()
        for m in true_by_m:
            cand=order[:m]; held=th[cand]+rng.normal(0,s_val,m); true_by_m[m][r]=th[cand[int(np.argmax(held))]]
    nt=naive_true.mean(); achievable={m:true_by_m[m].mean()-nt for m in true_by_m}; maxcut=max(achievable.values())
    m90=next((m for m in range(1,31) if achievable[m]>=0.9*maxcut), 30)
    bestm=max(achievable,key=achievable.get)
    print(f"  {rho:>14.1f} | {gaps.mean():+.4f}  | {gaps.mean()/S_TRUE:>6.1f}x  | {m90:>20} | {nt+achievable[bestm]:.4f} (m={bestm})")
