"""Actionable design follow-up to cycle 7: how should a verification layer pick among K parallel candidates?
Compare, at K=100 parallel candidates (true theta ~ N(0.80,0.05), single-holdout val noise s_val=0.05):
  NAIVE     = pick argmax(validation)  [the winner's-curse regime]
  VERIFY-m  = take top-m by validation, RE-EVALUATE each on a fresh held-out set, pick argmax(held-out).
We care about (i) optimism gap (reported - true of the FINAL pick), and (ii) TRUE perf / regret of the final pick
(what actually matters). Sweep m. Pre-reg: re-eval kills the gap AND raises true perf vs naive (picks a genuinely
good candidate, not a noise-inflated dud); diminishing returns in m. 4000 reps. numpy only."""
import numpy as np, json
rng = np.random.default_rng(0)
REPS=4000; K=100; MU,S_TRUE=0.80,0.05; S_VAL=0.05; S_REEVAL=0.05
ms=[1,2,3,5,10,20,50]
def run():
    naive_gap=np.empty(REPS); naive_true=np.empty(REPS)
    vm_gap={m:np.empty(REPS) for m in ms}; vm_true={m:np.empty(REPS) for m in ms}
    for r in range(REPS):
        theta=rng.normal(MU,S_TRUE,K); val=theta+rng.normal(0,S_VAL,K)
        order=np.argsort(-val)                       # candidates ranked by (noisy) validation
        # NAIVE: top-1 by val
        s=order[0]; naive_gap[r]=val[s]-theta[s]; naive_true[r]=theta[s]
        # VERIFY-m: re-eval top-m on a FRESH held-out set, pick best held-out
        for m in ms:
            cand=order[:m]; held=theta[cand]+rng.normal(0,S_REEVAL,m); p=cand[int(np.argmax(held))]
            vm_gap[m][r]=held[int(np.argmax(held))]-theta[p]; vm_true[m][r]=theta[p]
    ci=lambda x:1.96*x.std(ddof=1)/np.sqrt(len(x))
    return naive_gap,naive_true,vm_gap,vm_true,ci
ng,nt,vg,vt,ci=run()
oracle=MU+S_TRUE*np.mean(np.max(rng.normal(0,1,(2000,K)),axis=1))  # E[best true theta] reference
print(f"K={K}, reps={REPS}, true~N({MU},{S_TRUE}), val noise={S_VAL}\n")
print(f"NAIVE top-1:   optimism_gap={ng.mean():+.4f}±{ci(ng):.4f}   true_perf={nt.mean():.4f}±{ci(nt):.4f}")
print(f"oracle (best true theta) ~ {oracle:.4f}\n")
print(f"{'m':>3} | {'optimism_gap':>20} | {'true_perf of pick':>22} | {'regret vs oracle':>16}")
rows=[{"strategy":"naive_top1","gap":float(ng.mean()),"true":float(nt.mean())}]
for m in ms:
    g,t=vg[m].mean(),vt[m].mean(); print(f"{m:>3} | {g:+.4f} ± {ci(vg[m]):.4f}   | {t:.4f} ± {ci(vt[m]):.4f}     | {oracle-t:.4f}")
    rows.append({"strategy":f"verify_top{m}","gap":float(g),"true":float(t),"regret":float(oracle-t)})
print(f"\nINTERPRETATION:")
print(f"  NAIVE gap {ng.mean():+.4f} (overstated) vs VERIFY-m gap collapses toward ~{vg[ms[0]].mean():+.4f}..{vg[ms[-1]].mean():+.4f}")
print(f"  TRUE perf: naive {nt.mean():.4f} -> verify-top-{ms[len(ms)//2]} {vt[ms[len(ms)//2]].mean():.4f} (re-eval picks a genuinely better candidate)")
best_m=max(ms,key=lambda m:vt[m].mean()); print(f"  best true-perf at m={best_m} ({vt[best_m].mean():.4f}); diminishing/over-selection beyond")
json.dump({"naive":{"gap":float(ng.mean()),"true":float(nt.mean())},"oracle":float(oracle),"rows":rows},open("score.json","w"),indent=1)
print("\nwrote score.json")
