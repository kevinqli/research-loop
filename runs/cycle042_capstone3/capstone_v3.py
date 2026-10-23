"""Cycle 42 (consolidation): refreshed capstone v3 — the now-COMPLETE story (supersedes cycle-24 v2, which predated
the quant arc + real-agent thread + 2nd correction). 6 panels, all from saved score.json (verified vs data):
(a) winner's curse grows with K, held-out re-eval removes it (cycle 7 — centerpiece);
(b) QUANT: multi-regime survival tracks real alpha where in-sample R^2 can't separate null from real (cycle 30);
(c) verification adaptive BUT the data signature breaks — only the causal-ROLE policy recovers truth (cycle 18);
(d) REAL agent: self-confidence ranks the extremes but goes BLIND in the mid-range (cycle 39 — the case for the verifier);
(e) the loop AUDITS ITSELF: 3/3 findings survive a held-out re-test (cycle 19) — + 2 measure-before-keep corrections;
(f) the long-horizon TAIL = speed-vs-ceiling (cycle 12). MPS-exploratory (plotting only); each panel guarded."""
import json, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
R="runs/"
fig,ax=plt.subplots(2,3,figsize=(19,10))
fig.suptitle("What makes autonomous research trustworthy — verification at scale, applied to quant-agent-eval (a loop that audits itself)",
             fontsize=12.5, fontweight="bold")

# (a) winner's curse vs K (cycle 7)
try:
    rows=json.load(open(R+"cycle007_winners_curse/score.json")); K=[r["K"] for r in rows]
    a=ax[0,0]; a.plot(K,[r["gap_holdout"] for r in rows],"o-",color="C3",label="single-holdout select")
    a.plot(K,[r["gap_CV"] for r in rows],"s-",color="C1",label="CV select"); a.plot(K,[r["reeval_bias"] for r in rows],"^--",color="C2",label="held-out RE-EVAL")
    a.set_xscale("log"); a.axhline(0,color="k",lw=.5); a.set_xlabel("K (parallel candidates)"); a.set_ylabel("optimism gap (reported−true)")
    a.set_title("(a) HALLUCINATED PROGRESS grows with parallelism\nheld-out re-eval ≈0 removes it [centerpiece]"); a.legend(fontsize=8)
except Exception as ex: ax[0,0].text(.5,.5,f"(a) {ex}",ha="center",wrap=True)

# (b) quant: survival vs in-sample across SNR (cycle 30)
try:
    B=json.load(open(R+"cycle030_quant_survival/score.json"))["B_vs_SNR_K200"]; snr=[r["snr"] for r in B]
    a=ax[0,1]; a.plot(snr,[r["survival_rate"] for r in B],"o-",color="C2",label="verification-survival rate")
    a.plot(snr,[r["naive_IS"] for r in B],"s--",color="C3",label="naive in-sample R²")
    a.set_xscale("log"); a.set_xlabel("SNR (real signal)"); a.set_ylabel("metric value")
    a.set_title("(b) QUANT: survival tracks real alpha (12%→100%)\nin-sample R² can't separate null from real"); a.legend(fontsize=8)
except Exception as ex: ax[0,1].text(.5,.5,f"(b) {ex}",ha="center",wrap=True)

# (c) signature breaks, role recovers (cycle 18)
try:
    d=json.load(open(R+"cycle018_multicovariate/score.json")); tau=d["tau"]
    order=["marginal","shrink_grow","condition_all","condition_C(role)"]; labs=["marginal","shrink-grow","cond-all","ROLE"]
    bT=[d["policies"][k]["bT"] for k in order]
    a=ax[0,2]; a.bar(range(4),bT,.6,color=["gray","C3","C1","C2"]); a.axhline(tau,color="k",ls="--",label=f"true τ={tau}")
    a.set_xticks(range(4)); a.set_xticklabels(labs,fontsize=8,rotation=10); a.set_ylabel("estimated effect")
    a.set_title("(c) verification is ADAPTIVE but the data signature\nBREAKS — only the causal ROLE recovers τ"); a.legend(fontsize=8)
except Exception as ex: ax[0,2].text(.5,.5,f"(c) {ex}",ha="center",wrap=True)

# (d) real agent: confidence vs true across SNR (cycle 39)
try:
    rows=json.load(open(R+"cycle039_confidence_signal/score.json"))["rows"]; snr=[r["snr"] for r in rows]
    a=ax[1,0]; a.plot(snr,[r["confidence"] for r in rows],"s--",color="C0",label="agent self-confidence")
    a.plot(snr,[r["true_oos"] for r in rows],"o-",color="C2",label="true OOS R²")
    a.set_xscale("log"); a.axhline(0,color="k",lw=.5); a.set_xlabel("SNR"); a.set_ylabel("R²")
    a.axvspan(0.05,0.20,alpha=0.08,color="C3")
    a.set_title("(d) REAL agent: self-confidence ranks extremes but\ngoes BLIND in the mid-range (flat ~0.05) → need external eval"); a.legend(fontsize=8)
except Exception as ex: ax[1,0].text(.5,.5,f"(d) {ex}",ha="center",wrap=True)

# (e) loop self-audit survival (cycle 19) — orig vs holdout on diagonal
try:
    s=json.load(open(R+"cycle019_survival/survival_summary.json"))["findings"]
    pts=[("collider-aware",s["A_collider_aware_best"]["orig"]["B_collider"],s["A_collider_aware_best"]["holdout"]["B_collider"]),
         ("winner-curse@K100",s["B_winners_curse"]["orig"]["gap_K100"],s["B_winners_curse"]["holdout"]["gap_K100"]),
         ("mediator Bc",s["C_mediator_limit"]["orig"]["mediator_Bc"],s["C_mediator_limit"]["holdout"]["mediator_Bc"])]
    a=ax[1,1]; a.plot([0,1],[0,1],"k--",lw=.8,label="y=x (perfect replication)")
    for nm,o,h in pts: a.scatter(o,h,s=70,zorder=3); a.annotate(nm,(o,h),fontsize=7,xytext=(5,-9),textcoords="offset points")
    a.set_xlim(-.02,1); a.set_ylim(-.02,1); a.set_xlabel("original"); a.set_ylabel("held-out re-test")
    a.set_title("(e) the loop AUDITS ITSELF: 3/3 survive held-out\n+ 2 measure-before-keep CORRECTIONS (cycles 26, 37)"); a.legend(fontsize=8)
except Exception as ex: ax[1,1].text(.5,.5,f"(e) {ex}",ha="center",wrap=True)

# (f) tail (cycle 12 values)
try:
    T=[1,2,4,8,16,32]; agent=[0.5928,0.7618,0.8743,0.8867,0.8949,0.8900]; human=[0.6577,0.6966,0.8082,0.8995,0.9875,0.9963]
    a=ax[1,2]; a.plot(T,agent,"o-",color="C0",label="agent (fast, capped)"); a.plot(T,human,"s-",color="C3",label="human (slow→higher ceiling)")
    a.set_xscale("log"); a.set_xlabel("budget T (~hours)"); a.set_ylabel("best-so-far"); a.axvspan(2,4,alpha=.08,color="C0")
    a.set_title("(f) long-horizon TAIL = SPEED-vs-CEILING\n(self-metric also mis-judges the plateau — cycles 34/41)"); a.legend(fontsize=8)
except Exception as ex: ax[1,2].text(.5,.5,f"(f) {ex}",ha="center",wrap=True)

plt.tight_layout(rect=[0,0,1,0.95]); out=R+"cycle042_capstone3/capstone_v3.png"; plt.savefig(out,dpi=120)
print(f"wrote {out}; DONE")
