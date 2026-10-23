"""verification-eval v2: (a) replicate the cycle-2 ranking across SEEDS (CIs); (b) test a COLLIDER-AWARE rule.
Conditions:
  A           = naive marginal (REAL iff |t_marg|>2)
  B_naive     = A + within-CELL-t confound check (cycle-1 rule)
  B_sound     = decide on the confound-ADJUSTED effect (always condition on group)   <- over-corrects colliders
  B_collider  = condition ONLY if it SHRINKS the effect (|d_adj|<|d_marg| => confounder); if conditioning GROWS the
                effect (|d_adj|>|d_marg| => collider signature) keep the MARGINAL decision (don't condition).
8 task types incl. confound traps (t5,t6) and a collider (t8). Avg accuracy over SEEDS=20 with normal CIs +
per-task-type correctness. numpy only; deterministic per seed."""
import numpy as np

def t_pe(y1, y0):
    if len(y1) < 2 or len(y0) < 2: return 0.0, 0.0
    d = np.mean(y1) - np.mean(y0); se = np.sqrt(np.var(y1,ddof=1)/len(y1) + np.var(y0,ddof=1)/len(y0))
    return d, (d/se if se > 0 else 0.0)

def gen(seed):
    rng = np.random.default_rng(seed); T = {}
    def basic(tid,n,eff,grp=False):
        tr=rng.integers(0,2,n); o=50+eff*tr+rng.normal(0,5,n); g=rng.integers(0,2,n) if grp else None
        T[tid]=(tr.astype(float),o,(g.astype(float) if g is not None else None),"REAL" if eff!=0 else "NULL")
    def confound(tid,n):
        g=rng.integers(0,2,n); tr=(rng.random(n)<np.where(g==1,0.8,0.2)).astype(int); o=50+10*g+rng.normal(0,3,n)
        T[tid]=(tr.astype(float),o,g.astype(float),"NULL")
    def collider(tid,n):
        tr=rng.integers(0,2,n); U=rng.normal(0,1,n); g=((tr+U)>0.5).astype(int); o=50+6*U+rng.normal(0,2,n)
        T[tid]=(tr.astype(float),o,g.astype(float),"NULL")
    basic("t0",40,8.0); basic("t1",60,5.0); basic("t2",40,0.0); basic("t3",60,0.0,grp=True); basic("t4",16,4.0)
    confound("t5",60); confound("t6",80); collider("t8",120)
    return T

def adjusted(tr,o,g):
    num=wsum=var=0.0
    for gg in np.unique(g):
        m=g==gg; y1,y0=o[m&(tr==1)],o[m&(tr==0)]
        if len(y1)>=2 and len(y0)>=2:
            d,_=t_pe(y1,y0); w=m.sum(); num+=w*d; wsum+=w; var+=(w**2)*(np.var(y1,ddof=1)/len(y1)+np.var(y0,ddof=1)/len(y0))
    if wsum==0: return None
    d_adj=num/wsum; se=np.sqrt(var)/wsum; return d_adj,(d_adj/se if se>0 else 0.0)

def decide(tr,o,g):
    dm,tm=t_pe(o[tr==1],o[tr==0]); A="REAL" if abs(tm)>2 else "NULL"
    Bn=A
    if g is not None:
        tws=[abs(t_pe(o[(g==gg)&(tr==1)],o[(g==gg)&(tr==0)])[1]) for gg in np.unique(g)]; tws=[x for x in tws if x>0]
        if tws and abs(tm)>2 and np.mean(tws)<1.0: Bn="NULL"
    Bs=A; Bc=A
    if g is not None:
        adj=adjusted(tr,o,g)
        if adj is not None:
            d_adj,t_adj=adj
            Bs="REAL" if abs(t_adj)>2 else "NULL"                       # always trust adjusted
            if abs(d_adj) < abs(dm):                                    # conditioning SHRANK -> confounder -> trust adjusted
                Bc="REAL" if abs(t_adj)>2 else "NULL"
            else:                                                       # conditioning GREW -> collider signature -> keep marginal
                Bc=A
    return A,Bn,Bs,Bc

SEEDS=range(20,40); conds=["A","B_naive","B_sound","B_collider"]
acc={c:[] for c in conds}; pertask={c:{} for c in conds}
for s in SEEDS:
    T=gen(s); cc={c:0 for c in conds}
    for tid,(tr,o,g,gt) in T.items():
        A,Bn,Bs,Bc=decide(tr,o,g)
        for c,v in zip(conds,(A,Bn,Bs,Bc)):
            ok=(v==gt); cc[c]+=ok; pertask[c].setdefault(tid,[]).append(ok)
    for c in conds: acc[c].append(cc[c]/len(T))
print(f"n_tasks=8, seeds={len(list(SEEDS))}")
for c in conds:
    a=np.array(acc[c]); ci=1.96*a.std(ddof=1)/np.sqrt(len(a)); print(f"  acc[{c:11s}] = {a.mean():.3f} ± {ci:.3f}")
print("--- per-task correctness rate (across seeds), B_sound vs B_collider on the hard ones ---")
for tid in ["t3","t5","t6","t8"]:
    print(f"  {tid}: A={np.mean(pertask['A'][tid]):.2f} B_naive={np.mean(pertask['B_naive'][tid]):.2f} "
          f"B_sound={np.mean(pertask['B_sound'][tid]):.2f} B_collider={np.mean(pertask['B_collider'][tid]):.2f}")
import json; json.dump({c:[float(np.mean(acc[c])),float(1.96*np.std(acc[c],ddof=1)/np.sqrt(len(acc[c])))] for c in conds},
                       open("runs/cycle019_survival/vv2_score.json","w"),indent=1); print("\nwrote score.json (mean,ci per condition)")
