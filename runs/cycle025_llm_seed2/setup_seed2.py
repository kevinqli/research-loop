"""Set up the LLM-researcher comparison (verifeval-v3a). Regenerate the SAME seed-0 tasks the mechanical rules saw
(t0-t6, t8), write the agent-facing tasks.json (no ground truth), the hidden ground_truth.json, and mechanical.json
(A / B_sound always-condition / B_collider adaptive answers) so we can compare the LLM head-to-head. numpy only."""
import json, numpy as np, os
SEED = 2; rng = np.random.default_rng(SEED)
def t_pe(y1,y0):
    if len(y1)<2 or len(y0)<2: return 0.0,0.0
    d=np.mean(y1)-np.mean(y0); se=np.sqrt(np.var(y1,ddof=1)/len(y1)+np.var(y0,ddof=1)/len(y0)); return d,(d/se if se>0 else 0.0)
T={}; truth={}
def reg(tid,tr,o,g,gt): T[tid]=(np.asarray(tr,float),np.asarray(o,float),(np.asarray(g,float) if g is not None else None)); truth[tid]=gt
def basic(tid,n,eff,grp=False):
    tr=rng.integers(0,2,n); o=50+eff*tr+rng.normal(0,5,n); g=rng.integers(0,2,n) if grp else None; reg(tid,tr,o,g,"REAL" if eff!=0 else "NULL")
def confound(tid,n):
    g=rng.integers(0,2,n); tr=(rng.random(n)<np.where(g==1,0.8,0.2)).astype(int); o=50+10*g+rng.normal(0,3,n); reg(tid,tr,o,g,"NULL")
def collider(tid,n):
    tr=rng.integers(0,2,n); U=rng.normal(0,1,n); g=((tr+U)>0.5).astype(int); o=50+6*U+rng.normal(0,2,n); reg(tid,tr,o,g,"NULL")
basic("t0",40,8.0); basic("t1",60,5.0); basic("t2",40,0.0); basic("t3",60,0.0,grp=True); basic("t4",16,4.0); confound("t5",60); confound("t6",80); collider("t8",120)

def adjusted(tr,o,g):
    num=wsum=var=0.0
    for gg in np.unique(g):
        m=g==gg; y1,y0=o[m&(tr==1)],o[m&(tr==0)]
        if len(y1)>=2 and len(y0)>=2:
            d,_=t_pe(y1,y0); w=m.sum(); num+=w*d; wsum+=w; var+=(w**2)*(np.var(y1,ddof=1)/len(y1)+np.var(y0,ddof=1)/len(y0))
    return (None if wsum==0 else (num/wsum,(num/wsum)/(np.sqrt(var)/wsum) if var>0 else 0.0))
mech={}
for tid,(tr,o,g) in T.items():
    dm,tm=t_pe(o[tr==1],o[tr==0]); A="REAL" if abs(tm)>2 else "NULL"; Bs=Bc=A
    if g is not None:
        adj=adjusted(tr,o,g)
        if adj: d_adj,t_adj=adj; Bs="REAL" if abs(t_adj)>2 else "NULL"; Bc=("REAL" if abs(t_adj)>2 else "NULL") if abs(d_adj)<abs(dm) else A
    mech[tid]={"A":A,"B_sound":Bs,"B_collider":Bc}

os.makedirs("runs/cycle025_llm_seed2/researcher_view",exist_ok=True)
agent={tid:{"question":"Does `treatment` have a REAL causal effect on `outcome`, or NULL?","data":[{k:(round(float(v),2)) for k,v in zip(("treatment","outcome","group"),(tr[i],o[i],(g[i] if g is not None else None))) if v is not None} for i in range(len(tr))]} for tid,(tr,o,g) in T.items()}
json.dump(agent,open("runs/cycle025_llm_seed2/researcher_view/tasks.json","w"))
json.dump(truth,open("runs/cycle025_llm_seed2/ground_truth.json","w"),indent=1)
json.dump(mech,open("runs/cycle025_llm_seed2/mechanical.json","w"),indent=1)
print("truth:   "+", ".join(f"{k}:{v}" for k,v in truth.items()))
print("mech B_collider: "+", ".join(f"{k}:{mech[k]['B_collider']}" for k in truth))
print("wrote researcher_view/tasks.json, ground_truth.json, mechanical.json")
