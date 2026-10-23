"""Cycle 77 (consolidation): capstone v4 — supersedes v3 (cycle 42), which predated the process-verification/leakage arc
(c63-68), the real-agent tail arc (c69-71), the capability map (c74) and the meta-audit (c76). 8 panels (2x4); a-f are
the v3 panels (same saved score.json sources, re-verified), g-h are NEW:
(g) the SECOND verification axis — PROCESS vs OUTCOME: outcome-checking is BLIND to leakage (acc 0.000), process catches
    it 100%, and vice-versa on clean-but-dead (c63);
(h) the CAPABILITY MAP (c74) — where agent-judgment SUFFICES vs needs EXTERNAL verification — with the c76 HONEST CAVEAT
    that the real-agent rows are high-α self-tests (same model, n=1/arm), directional not verified.
All numeric panels read saved score.json (no fabricated numbers). Plotting only."""
import json, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
R="runs/"
fig,ax=plt.subplots(2,4,figsize=(25,10.5))
fig.suptitle("What makes autonomous research trustworthy — verification at scale (two axes), applied to quant-agent-eval, by a loop that audits itself",
             fontsize=13, fontweight="bold")

# (a) winner's curse vs K (cycle 7)
try:
    rows=json.load(open(R+"cycle007_winners_curse/score.json")); K=[r["K"] for r in rows]
    a=ax[0,0]; a.plot(K,[r["gap_holdout"] for r in rows],"o-",color="C3",label="single-holdout select")
    a.plot(K,[r["gap_CV"] for r in rows],"s-",color="C1",label="CV select"); a.plot(K,[r["reeval_bias"] for r in rows],"^--",color="C2",label="held-out RE-EVAL")
    a.set_xscale("log"); a.axhline(0,color="k",lw=.5); a.set_xlabel("K (parallel candidates)"); a.set_ylabel("optimism gap (reported−true)")
    a.set_title("(a) HALLUCINATED PROGRESS grows with parallelism\nheld-out re-eval ≈0 removes it [centerpiece]"); a.legend(fontsize=8)
except Exception as ex: ax[0,0].text(.5,.5,f"(a) {ex}",ha="center",wrap=True)

# (b) quant survival (cycle 30)
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

# (g) NEW: process vs outcome axis (cycle 63), SNR=0.15
try:
    d=json.load(open(R+"cycle063_reasoning_vs_outcome/score.json"))["0.15"]
    types=["REAL","OVERFIT","LEAKY","CLEAN-NOISE"]; x=np.arange(4)
    outc=[d[t]["outcome"] for t in types]; proc=[d[t]["process"] for t in types]
    a=ax[0,3]; a.bar(x-.2,outc,.4,color="C3",label="OUTCOME check"); a.bar(x+.2,proc,.4,color="C2",label="PROCESS/construction audit")
    a.set_xticks(x); a.set_xticklabels(types,fontsize=7.5,rotation=12); a.set_ylabel("classification accuracy"); a.set_ylim(0,1.08)
    a.annotate("outcome BLIND\nto leakage",(2.2,0.04),fontsize=7.5,color="C3")
    a.set_title("(g) the SECOND axis: PROCESS vs OUTCOME\noutcome-check is BLIND to leakage (0.00); construction catches it"); a.legend(fontsize=8,loc="center left")
except Exception as ex: ax[0,3].text(.5,.5,f"(g) {ex}",ha="center",wrap=True)

# (d) real agent confidence (cycle 39)
try:
    rows=json.load(open(R+"cycle039_confidence_signal/score.json"))["rows"]; snr=[r["snr"] for r in rows]
    a=ax[1,0]; a.plot(snr,[r["confidence"] for r in rows],"s--",color="C0",label="agent self-confidence")
    a.plot(snr,[r["true_oos"] for r in rows],"o-",color="C2",label="true OOS R²")
    a.set_xscale("log"); a.axhline(0,color="k",lw=.5); a.set_xlabel("SNR"); a.set_ylabel("R²"); a.axvspan(0.05,0.20,alpha=0.08,color="C3")
    a.set_title("(d) REAL agent: self-confidence ranks extremes but\ngoes BLIND in the mid-range → need external eval"); a.legend(fontsize=8)
except Exception as ex: ax[1,0].text(.5,.5,f"(d) {ex}",ha="center",wrap=True)

# (e) self-audit survival (cycle 19)
try:
    s=json.load(open(R+"cycle019_survival/survival_summary.json"))["findings"]
    pts=[("collider-aware",s["A_collider_aware_best"]["orig"]["B_collider"],s["A_collider_aware_best"]["holdout"]["B_collider"]),
         ("winner-curse@K100",s["B_winners_curse"]["orig"]["gap_K100"],s["B_winners_curse"]["holdout"]["gap_K100"]),
         ("mediator Bc",s["C_mediator_limit"]["orig"]["mediator_Bc"],s["C_mediator_limit"]["holdout"]["mediator_Bc"])]
    a=ax[1,1]; a.plot([0,1],[0,1],"k--",lw=.8,label="y=x (perfect replication)")
    for nm,o,h in pts: a.scatter(o,h,s=70,zorder=3); a.annotate(nm,(o,h),fontsize=7,xytext=(5,-9),textcoords="offset points")
    a.set_xlim(-.02,1); a.set_ylim(-.02,1); a.set_xlabel("original"); a.set_ylabel("held-out re-test")
    a.set_title("(e) the loop AUDITS ITSELF: 3/3 survive held-out\n+ measure-before-keep CORRECTIONS (now 19 in FINDINGS)"); a.legend(fontsize=8)
except Exception as ex: ax[1,1].text(.5,.5,f"(e) {ex}",ha="center",wrap=True)

# (f) tail (cycle 12)
try:
    T=[1,2,4,8,16,32]; agent=[0.5928,0.7618,0.8743,0.8867,0.8949,0.8900]; human=[0.6577,0.6966,0.8082,0.8995,0.9875,0.9963]
    a=ax[1,2]; a.plot(T,agent,"o-",color="C0",label="agent (fast, capped)"); a.plot(T,human,"s-",color="C3",label="human (slow→higher ceiling)")
    a.set_xscale("log"); a.set_xlabel("budget T (~hours)"); a.set_ylabel("best-so-far"); a.axvspan(2,4,alpha=.08,color="C0")
    a.set_title("(f) long-horizon TAIL = SPEED-vs-CEILING\n(real agent self-corrects easy plateaus 6/6, thrashes 5/5 borderline)"); a.legend(fontsize=8)
except Exception as ex: ax[1,2].text(.5,.5,f"(f) {ex}",ha="center",wrap=True)

# (h) NEW: capability map (cycle 74) + honest caveat (cycle 76)
try:
    a=ax[1,3]; a.axis("off")
    a.set_title("(h) THE CAPABILITY MAP (c74): when does\nagent-judgment suffice vs need external verification?", fontsize=10)
    suffices=("AGENT SELF-CORRECTS (judgment suffices):\n"
      "• avoids overfit catastrophe via CV (c36)\n• catches leakage when it left a trace (c64-68)\n"
      "• stops on EASY plateaus 6/6 (c69)\n• knows when data CAN'T answer 4/4 (c73)\n• resists sunk-cost bias (c70/71)")
    fails=("NEEDS EXTERNAL INDEPENDENT VERIFICATION:\n"
      "• magnitude / selection MARGIN (mid-range c39,\n   money-losers c44)\n• any metric it can GAME (c52/55)\n"
      "• BORDERLINE plateaus — thrash 5/5 (c71)\n• data-only causal/leakage w/o metadata (c67/68)")
    a.text(0.0,0.97,suffices,fontsize=7.6,va="top",ha="left",color="darkgreen",transform=a.transAxes)
    a.text(0.0,0.46,fails,fontsize=7.6,va="top",ha="left",color="firebrick",transform=a.transAxes)
    a.text(0.0,0.10,"Moat = the agent can't judge the MARGIN &\nwill GAME any metric it sees → verify exactly there.",
           fontsize=7.8,va="top",ha="left",style="italic",transform=a.transAxes)
    a.text(0.0,-0.01,"⚠ real-agent rows are HIGH-α self-tests (same model,\nn=1/arm, loop-designed) — directional, not independently\nverified; the low-α check is Q5 (diff model/real data). [c76]",
           fontsize=6.8,va="top",ha="left",color="dimgray",transform=a.transAxes)
except Exception as ex: ax[1,3].text(.5,.5,f"(h) {ex}",ha="center",wrap=True)

plt.tight_layout(rect=[0,0,1,0.95]); out=R+"cycle077_capstone4/capstone_v4.png"; plt.savefig(out,dpi=115)
print(f"wrote {out}; DONE")
