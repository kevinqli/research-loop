"""verification-eval v1 (self-contained): compare THREE verification conditions on known-truth stats tasks —
  A        = naive marginal t-test (REAL iff |t|>2)
  B_naive  = A + cycle-1's confound check (mean of within-CELL t-stats; the rule that failed)
  B_sound  = A + a *statistically sound* confound check: weighted within-group POINT estimate (mean-diff) + its
             pooled SE -> REAL iff the CONFOUND-ADJUSTED effect is significant.
Tasks: clear real/null + confound traps (cond. on group -> effect vanishes) + a COLLIDER (cond. on group INDUCES a
spurious effect -> tests whether mechanical 'always condition' OVER-corrects). Truth set by construction. numpy only."""
import numpy as np
rng = np.random.default_rng(0)

def t_pe(y1, y0):  # returns (point_estimate mean-diff, t-stat)
    if len(y1) < 2 or len(y0) < 2: return 0.0, 0.0
    d = np.mean(y1) - np.mean(y0); se = np.sqrt(np.var(y1,ddof=1)/len(y1) + np.var(y0,ddof=1)/len(y0))
    return d, (d/se if se > 0 else 0.0)

tasks = {}; truth = {}
def reg(tid, treat, out, group, gt, note):
    tasks[tid] = dict(treat=np.asarray(treat,float), out=np.asarray(out,float),
                      group=(np.asarray(group,float) if group is not None else None)); truth[tid] = (gt, note)

# t0,t1 clear real ; t2 clear null ; t3 null+irrelevant group ; t4 underpowered real ; t5 confound trap
def basic(tid, n, eff, grp_irrel=False):
    tr = rng.integers(0,2,n); o = 50 + eff*tr + rng.normal(0,5,n)
    g = rng.integers(0,2,n) if grp_irrel else None
    reg(tid, tr, o, g, "REAL" if eff!=0 else "NULL", f"basic eff={eff}")
basic("t0",40,8.0); basic("t1",60,5.0); basic("t2",40,0.0); basic("t3",60,0.0,grp_irrel=True); basic("t4",16,4.0)
def confound(tid, n):  # group drives treatment AND outcome; treatment has NO direct effect -> NULL
    g = rng.integers(0,2,n); tr = (rng.random(n) < np.where(g==1,0.8,0.2)).astype(int)
    o = 50 + 10*g + rng.normal(0,3,n); reg(tid, tr, o, g, "NULL", "confound (marginal spurious)")
confound("t5",60); confound("t6",80)
def collider(tid, n):  # treatment & U both cause group (collider); outcome depends on U, NOT treatment -> marginal NULL is CORRECT; conditioning on group induces spurious assoc
    tr = rng.integers(0,2,n); U = rng.normal(0,1,n)
    g = ((tr + U) > 0.5).astype(int)              # collider of treatment and U
    o = 50 + 6*U + rng.normal(0,2,n)              # outcome <- U only
    reg(tid, tr, o, g, "NULL", "COLLIDER (conditioning induces spurious effect)")
collider("t8",120)

def analyze(tid):
    d = tasks[tid]; tr, o, g = d["treat"], d["out"], d["group"]
    dm, tm = t_pe(o[tr==1], o[tr==0])                       # marginal
    A = "REAL" if abs(tm) > 2 else "NULL"
    # B_naive: mean of within-CELL t-stats (cycle-1 rule)
    Bn = A
    if g is not None:
        tws = [t_pe(o[(g==gg)&(tr==1)], o[(g==gg)&(tr==0)])[1] for gg in np.unique(g)]
        tws = [abs(x) for x in tws if x != 0.0]
        if tws and abs(tm) > 2 and np.mean(tws) < 1.0: Bn = "NULL"
    # B_sound: weighted within-group POINT estimate + pooled SE -> significance of ADJUSTED effect
    Bs = A
    if g is not None:
        num = 0.0; wsum = 0.0; var = 0.0
        for gg in np.unique(g):
            m = g==gg; y1, y0 = o[m&(tr==1)], o[m&(tr==0)]
            if len(y1) >= 2 and len(y0) >= 2:
                dgg, _ = t_pe(y1, y0); w = m.sum()
                num += w*dgg; wsum += w
                var += (w**2)*(np.var(y1,ddof=1)/len(y1) + np.var(y0,ddof=1)/len(y0))
        if wsum > 0:
            d_adj = num/wsum; se_adj = np.sqrt(var)/wsum; t_adj = d_adj/se_adj if se_adj>0 else 0.0
            Bs = "REAL" if abs(t_adj) > 2 else "NULL"   # decide on the confound-ADJUSTED effect
    return A, Bn, Bs, abs(tm)

conds = {"A":0, "B_naive":0, "B_sound":0}; rows = []
for tid,(gt,note) in truth.items():
    A,Bn,Bs,tm = analyze(tid)
    for k,v in (("A",A),("B_naive",Bn),("B_sound",Bs)):
        if v == gt: conds[k]+=1
    rows.append((tid,gt,A,Bn,Bs,note))
n=len(truth)
print(f"n={n} tasks")
for k in conds: print(f"  acc[{k:8s}] = {conds[k]}/{n} = {conds[k]/n:.2f}")
print("--- per task (truth | A | B_naive | B_sound) ---")
for tid,gt,A,Bn,Bs,note in rows:
    flag = "" if Bs==gt else "  <-Bsound WRONG"
    print(f"  {tid}: {gt:4s} | {A:4s} | {Bn:4s} | {Bs:4s}  {note}{flag}")
import json; json.dump({"n":n,**{f"acc_{k}":conds[k]/n for k in conds}}, open("score.json","w"), indent=1)
print("\nwrote score.json")
