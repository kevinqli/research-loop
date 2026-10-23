"""Capstone figure for the agentic-research loop. 4 panels from the loop's own saved results:
(a) verification is ADAPTIVE not maximal (verifeval-v2, cycle3); (b) winner's curse grows with parallel breadth K,
held-out re-eval ~0 (cycle7); (c) cheap verifier: re-eval top-m true-perf (cycle8); (d) RE-Bench tail crossover =
speed-vs-ceiling (cycle12). Reads score.json files. MPS-exploratory (plotting only)."""
import json, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
R="runs/"
fig,ax=plt.subplots(2,2,figsize=(15,10))
fig.suptitle("Self-improving agentic-research loop — what makes autonomous research trustworthy (3 pillars + a recipe)", fontsize=12, fontweight="bold")

# (a) verification-eval: adaptive wins
try:
    d=json.load(open(R+"cycle003_verifeval_v2/score.json")); conds=["A","B_naive","B_sound","B_collider"]
    labs=["none","within-cell-t","always-condition","collider-aware"]; m=[d[c][0] for c in conds]; e=[d[c][1] for c in conds]
    a=ax[0,0]; b=a.bar(range(4),m,0.6,yerr=e,capsize=4,color=["gray","C0","C1","C2"])
    a.set_xticks(range(4)); a.set_xticklabels(labs,fontsize=8); a.set_ylabel("accuracy (20 seeds)"); a.set_ylim(0.5,1.0)
    a.set_title("(a) VERIFICATION is ADAPTIVE, not maximal\ncollider-aware 0.88 > always-condition 0.76 (over-corrects)")
    for i,v in enumerate(m): a.annotate(f"{v:.2f}",(i,v),ha="center",va="bottom",fontsize=8)
except Exception as ex: ax[0,0].text(.5,.5,f"(a) {ex}",ha="center",wrap=True)

# (b) winner's curse vs K
try:
    rows=json.load(open(R+"cycle007_winners_curse/score.json")); K=[r["K"] for r in rows]
    gh=[r["gap_holdout"] for r in rows]; gc=[r["gap_CV"] for r in rows]; rb=[r["reeval_bias"] for r in rows]
    a=ax[0,1]; a.plot(K,gh,"o-",color="C3",label="single-holdout select"); a.plot(K,gc,"s-",color="C1",label="CV select"); a.plot(K,rb,"^--",color="C2",label="held-out RE-EVAL")
    a.set_xscale("log"); a.axhline(0,color="k",lw=0.5); a.set_xlabel("K (parallel candidates)"); a.set_ylabel("optimism gap (reported − true)")
    a.set_title("(b) 'HALLUCINATED PROGRESS' grows with parallelism\nheld-out re-eval ≈0 removes it"); a.legend(fontsize=8)
except Exception as ex: ax[0,1].text(.5,.5,f"(b) {ex}",ha="center",wrap=True)

# (c) verify-top-m recipe
try:
    d=json.load(open(R+"cycle008_verify_topm/score.json")); naive=d["naive"]["true"]; orc=d["oracle"]
    ms=[]; tr=[]
    for r in d["rows"]:
        if r["strategy"].startswith("verify_top"): ms.append(int(r["strategy"].replace("verify_top",""))); tr.append(r["true"])
    a=ax[1,0]; a.plot(ms,tr,"o-",color="C2"); a.axhline(naive,color="gray",ls="--",label=f"naive top-1 ({naive:.3f})"); a.axhline(orc,color="k",ls=":",label=f"oracle ({orc:.3f})")
    a.set_xscale("log"); a.set_xlabel("m (top-m re-evaluated on held-out)"); a.set_ylabel("true perf of final pick")
    a.set_title("(c) CHEAP VERIFIER: re-eval top ~3 captures ~96%\n(peak ~m=10; m>10 over-selects)"); a.legend(fontsize=8)
except Exception as ex: ax[1,0].text(.5,.5,f"(c) {ex}",ha="center",wrap=True)

# (d) tail crossover (cycle12 values, from logged output)
try:
    T=[1,2,4,8,16,32]; agent=[0.5928,0.7618,0.8743,0.8867,0.8949,0.8900]; human=[0.6577,0.6966,0.8082,0.8995,0.9875,0.9963]
    a=ax[1,1]; a.plot(T,agent,"o-",color="C0",label="agent (fast, capped)"); a.plot(T,human,"s-",color="C3",label="human (slow, priors→higher ceiling)")
    a.set_xscale("log"); a.set_xlabel("budget T (~hours)"); a.set_ylabel("best-so-far")
    a.set_title("(d) RE-Bench TAIL crossover = SPEED-vs-CEILING\nagent wins short (T=2-4), higher ceiling wins long"); a.legend(fontsize=8)
    a.axvspan(2,4,alpha=0.08,color="C0")
except Exception as ex: ax[1,1].text(.5,.5,f"(d) {ex}",ha="center",wrap=True)

plt.tight_layout(rect=[0,0,1,0.96]); out=R+"cycle013_capstone/capstone_figure.png"; plt.savefig(out,dpi=120); print(f"wrote {out}; DONE")
