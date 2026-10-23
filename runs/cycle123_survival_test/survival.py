"""Cycle 123: SURVIVAL-TEST of recent findings on a FRESH SEED (12345, vs each finding's original seed=0). The
finding-survival-rate discipline (c19/meth-9) re-runs a finding on independent seeds to check it replicates. Only
done once (c19); the 90+ findings since are un-survival-tested. Re-implement the CORE metric of 5 recent findings with
a fresh seed and compare to the LOGGED value — survive iff it replicates within ~CI. numpy-only."""
import numpy as np
RNG = np.random.default_rng(12345)            # FRESH seed (originals used 0)
results = []

# --- c90: leaderboard gap, N=100, sig_skill=0.3, sig_noise=sig_inst/sqrt(M)=1/sqrt(100)=0.1. LOGGED sim_gap≈0.0789 ---
N=100; SK=0.3; SN=0.1; R=40000
s=RNG.normal(0,SK,(R,N)); sc=s+RNG.normal(0,SN,(R,N)); r=np.arange(R); w=sc.argmax(1)
g90=float((sc[r,w]-s[r,w]).mean()); results.append(("c90 leaderboard gap N=100", 0.0789, g90))

# --- c107: correlated eval, rho=0.5, sig_skill=0.3, sig_noise=1. LOGGED gap≈1.634 ---
N=100; SK=0.3; SN=1.0; rho=0.5; R=60000
s=RNG.normal(0,SK,(R,N)); zc=RNG.normal(0,SN,(R,1)); zi=RNG.normal(0,SN,(R,N))
e=np.sqrt(rho)*zc+np.sqrt(1-rho)*zi; r=np.arange(R); w=(s+e).argmax(1)
g107=float(e[r,w].mean()); results.append(("c107 correlated-eval gap rho=0.5", 1.634, g107))

# --- c109: verification regress, p0=0.3, q=0.7. analytic is deterministic; re-derive + fresh MC at s=0,k=3 ---
P0,Q=0.3,0.7
def post(k,s): pf=s+(1-s)*(1-Q)**k; return P0/(P0+(1-P0)*pf)
a_s0k3=post(3,0.0); a_s2inf=P0/(P0+(1-P0)*0.2)
results.append(("c109 regress posterior s=0,k=3 (analytic)", 0.9407, round(a_s0k3,4)))
results.append(("c109 regress floor s=0.2 (analytic)", 0.6818, round(a_s2inf,4)))

# --- c119: generator oracle at K=100: reliable N(0.8,0.3) vs wild N(0.2,0.6). LOGGED rel 1.5515, wild 1.7053 ---
K=100; R=30000
rel=RNG.normal(0.8,0.3,(R,K)).max(1).mean(); wild=RNG.normal(0.2,0.6,(R,K)).max(1).mean()
results.append(("c119 reliable oracle K=100", 1.5515, round(float(rel),4)))
results.append(("c119 wild oracle K=100", 1.7053, round(float(wild),4)))

# --- c122: batch(re-eval top-3) vs secretary at sig=0.5, K=50. LOGGED batch 2.073, seq 1.134 ---
K=50; M=3; SG=0.5; R=20000; LEARN=int(round(K/np.e))
q=RNG.normal(0,1,(R,K)); obs=q+RNG.normal(0,SG,(R,K)); r=np.arange(R)
topm=np.argsort(-obs,1)[:,:M]; qh=np.take_along_axis(q,topm,1)+RNG.normal(0,SG,(R,M))
bsel=np.take_along_axis(topm,qh.argmax(1)[:,None],1)[:,0]; batch=float(q[r,bsel].mean())
thr=obs[:,:LEARN].max(1); o=q[:,-1].copy(); taken=np.zeros(R,bool)
for j in range(LEARN,K):
    pk=(~taken)&(obs[:,j]>thr); o=np.where(pk,q[:,j],o); taken=taken|pk
results.append(("c122 batch outcome sig=0.5", 2.073, round(batch,4)))
results.append(("c122 secretary outcome sig=0.5", 1.134, round(float(o.mean()),4)))

print(f"SURVIVAL TEST (fresh seed 12345). survive iff |fresh − logged|/|logged| < 5%.\n")
print(f"{'finding':>42} | {'logged':>8} | {'fresh':>8} | {'rel.err':>8} | verdict")
nsurv=0
for name, logged, fresh in results:
    rel = abs(fresh-logged)/max(abs(logged),1e-9)
    ok = rel < 0.05; nsurv += ok
    print(f"{name:>42} | {logged:>8} | {fresh:>8} | {rel:>7.2%} | {'SURVIVE' if ok else 'FAIL'}")
print(f"\n--- {nsurv}/{len(results)} survived a fresh-seed re-run (<5% rel.err) ---")
print("DONE")
