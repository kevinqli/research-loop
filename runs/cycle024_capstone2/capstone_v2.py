"""Cycle 24 (consolidation): refreshed capstone figure — the FULL current story in 6 panels, all from saved
score.json (no recompute). Supersedes cycle-13's 4-panel figure (which predated the conditioning-arc completion,
the self-audit, and the scaffold work). Panels:
(a) verification is ADAPTIVE (cycle3)  (b) ...but the mechanical signature BREAKS in the multi-covariate case,
only the causal-ROLE policy recovers truth (cycle18)  (c) the winner's curse: hallucinated progress grows with K,
held-out re-eval removes it (cycle7)  (d) scaffold: branching's true gain vs its ~2x-inflated self-report (cycle22)
(e) the loop AUDITS ITSELF: 3/3 findings survive a held-out re-test (cycle19)  (f) the long-horizon tail =
speed-vs-ceiling (cycle12). MPS-exploratory (plotting only). Each panel guarded; visually verify vs data."""
import json, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
R = "runs/"
fig, ax = plt.subplots(2, 3, figsize=(19, 10))
fig.suptitle("What makes autonomous research trustworthy — verification (filter→gate), the winner's curse, "
             "scaffolds, and a loop that audits itself", fontsize=13, fontweight="bold")

# (a) verification adaptive (cycle 3)
try:
    d = json.load(open(R+"cycle003_verifeval_v2/score.json")); conds=["A","B_naive","B_sound","B_collider"]
    labs=["none","within-cell-t","always-cond","collider-aware"]; m=[d[c][0] for c in conds]; e=[d[c][1] for c in conds]
    a=ax[0,0]; a.bar(range(4),m,0.6,yerr=e,capsize=4,color=["gray","C0","C1","C2"])
    a.set_xticks(range(4)); a.set_xticklabels(labs,fontsize=8,rotation=10); a.set_ylabel("accuracy (20 seeds)"); a.set_ylim(0.5,1.0)
    a.set_title("(a) verification is ADAPTIVE, not maximal\ncollider-aware 0.88 > always-condition 0.76")
    for i,v in enumerate(m): a.annotate(f"{v:.2f}",(i,v),ha="center",va="bottom",fontsize=8)
except Exception as ex: ax[0,0].text(.5,.5,f"(a) {ex}",ha="center",wrap=True)

# (b) signature breaks in multi-covariate; only role recovers truth (cycle 18)
try:
    d=json.load(open(R+"cycle018_multicovariate/score.json")); tau=d["tau"]
    order=["marginal","shrink_grow","condition_all","condition_C(role)"]; labs=["marginal","shrink-vs-grow","condition-all","role (pre/post)"]
    bT=[d["policies"][k]["bT"] for k in order]; err=[d["policies"][k]["abs_err"] for k in order]
    a=ax[0,1]; cols=["gray","C3","C1","C2"]; a.bar(range(4),bT,0.6,color=cols)
    a.axhline(tau,color="k",ls="--",lw=1.2,label=f"true effect τ={tau}")
    a.set_xticks(range(4)); a.set_xticklabels(labs,fontsize=8,rotation=10); a.set_ylabel("estimated total effect")
    a.set_title("(b) ...but the data SIGNATURE breaks (3 covariates)\nonly the causal-ROLE policy recovers τ"); a.legend(fontsize=8)
    for i,v in enumerate(bT): a.annotate(f"{v:.2f}",(i,v),ha="center",va="bottom" if v>0 else "top",fontsize=8)
except Exception as ex: ax[0,1].text(.5,.5,f"(b) {ex}",ha="center",wrap=True)

# (c) winner's curse vs K (cycle 7)
try:
    rows=json.load(open(R+"cycle007_winners_curse/score.json")); K=[r["K"] for r in rows]
    gh=[r["gap_holdout"] for r in rows]; gc=[r["gap_CV"] for r in rows]; rb=[r["reeval_bias"] for r in rows]
    a=ax[0,2]; a.plot(K,gh,"o-",color="C3",label="single-holdout select"); a.plot(K,gc,"s-",color="C1",label="CV select"); a.plot(K,rb,"^--",color="C2",label="held-out RE-EVAL")
    a.set_xscale("log"); a.axhline(0,color="k",lw=0.5); a.set_xlabel("K (parallel candidates)"); a.set_ylabel("optimism gap (reported − true)")
    a.set_title("(c) 'HALLUCINATED PROGRESS' grows with parallelism\nheld-out re-eval ≈0 removes it"); a.legend(fontsize=8)
except Exception as ex: ax[0,2].text(.5,.5,f"(c) {ex}",ha="center",wrap=True)

# (d) scaffold branch true vs reported gain (cycle 22)
try:
    d=json.load(open(R+"cycle022_scaffold/score.json")); br=d["branch"]
    K=[r["K"] for r in br]; tg=[r["true_gain"] for r in br]; rg=[r["reported_gain"] for r in br]
    a=ax[1,0]; a.plot(K,tg,"o-",color="C2",label="TRUE gain"); a.plot(K,rg,"s--",color="C3",label="self-REPORTED gain (~2× inflated)")
    a.set_xscale("log"); a.set_xlabel("K (branch factor)"); a.set_ylabel("gain over linear agent")
    a.set_title("(d) BRANCHING helps (true) but self-report\n~2× inflated (winner's curse); per-token K*≈2 (linear cost)"); a.legend(fontsize=8)
except Exception as ex: ax[1,0].text(.5,.5,f"(d) {ex}",ha="center",wrap=True)

# (e) loop self-audit: survival re-test (cycle 19) — orig vs holdout on the y=x diagonal
try:
    s=json.load(open(R+"cycle019_survival/survival_summary.json"))["findings"]
    pts=[("collider-aware acc",s["A_collider_aware_best"]["orig"]["B_collider"],s["A_collider_aware_best"]["holdout"]["B_collider"]),
         ("winner-curse gap@K100",s["B_winners_curse"]["orig"]["gap_K100"],s["B_winners_curse"]["holdout"]["gap_K100"]),
         ("mediator Bc acc",s["C_mediator_limit"]["orig"]["mediator_Bc"],s["C_mediator_limit"]["holdout"]["mediator_Bc"])]
    a=ax[1,1]; a.plot([0,1],[0,1],"k--",lw=0.8,label="y=x (perfect replication)")
    for nm,o,h in pts: a.scatter(o,h,s=70,zorder=3); a.annotate(nm,(o,h),fontsize=7,xytext=(5,-8),textcoords="offset points")
    a.set_xlim(-0.02,1); a.set_ylim(-0.02,1); a.set_xlabel("original value"); a.set_ylabel("held-out re-test value")
    a.set_title("(e) the loop AUDITS ITSELF: 3/3 findings\nSURVIVE a held-out re-test (on the diagonal)"); a.legend(fontsize=8)
except Exception as ex: ax[1,1].text(.5,.5,f"(e) {ex}",ha="center",wrap=True)

# (f) tail speed-vs-ceiling (cycle 12 logged values)
try:
    T=[1,2,4,8,16,32]; agent=[0.5928,0.7618,0.8743,0.8867,0.8949,0.8900]; human=[0.6577,0.6966,0.8082,0.8995,0.9875,0.9963]
    a=ax[1,2]; a.plot(T,agent,"o-",color="C0",label="agent (fast, capped)"); a.plot(T,human,"s-",color="C3",label="human (slow, priors→ceiling)")
    a.set_xscale("log"); a.set_xlabel("budget T (~hours)"); a.set_ylabel("best-so-far")
    a.set_title("(f) the long-horizon TAIL = SPEED-vs-CEILING\nagent wins short (T=2-4), higher ceiling wins long"); a.legend(fontsize=8)
    a.axvspan(2,4,alpha=0.08,color="C0")
except Exception as ex: ax[1,2].text(.5,.5,f"(f) {ex}",ha="center",wrap=True)

plt.tight_layout(rect=[0,0,1,0.95]); out=R+"cycle024_capstone2/capstone_v2.png"; plt.savefig(out,dpi=120)
print(f"wrote {out}; DONE")
