"""Cycle 40: verification as a STOPPING SIGNAL for the long-horizon tail (unifies tail cycle-12 + iterative
winner's curse cycle-34 + the verification pillar). Iterative hill-climb on a noisy objective f=-mean(theta^2)
(optimum 0). Each round: propose K, evaluate by noisy validation, move to best. The agent's SELF-metric (selection
value) climbs monotonically and crosses the ceiling (cycle 34) -> it NEVER plateaus -> an agent watching its own
metric never knows to stop. A HELD-OUT re-eval reveals the TRUE plateau. Compare stopping rules (patience W: stop if
no improvement for W rounds):
  NAIVE   stop on the self-reported metric   -> (pre-reg) never triggers; runs the full budget
  VERIFIED stop on a held-out re-eval         -> stops near the true plateau, ~same true perf, far fewer rounds
  ORACLE  stop on the TRUE objective (unobservable; reference)
Pre-reg: verified-stop uses far fewer rounds than naive for ~equal true perf; the naive self-metric never plateaus.
numpy-only, SEEDS seeds. MPS-exploratory (toy)."""
import numpy as np
from math import sqrt
D, K, TMAX, STEP, S_EVAL, W, SEEDS = 10, 20, 50, 0.3, 0.3, 5, 80
def f(th): return -np.mean(th**2, axis=-1)

def episode(seed):
    rng = np.random.default_rng(seed); th = rng.normal(0,1,D)
    rep=[float(f(th))]; held=[float(f(th)+rng.normal(0,S_EVAL))]; true=[float(f(th))]
    for _ in range(TMAX):
        c = th + STEP*rng.normal(0,1,(K,D)); val = f(c)+rng.normal(0,S_EVAL,K); b=int(np.argmax(val))
        th = c[b]
        rep.append(float(val[b]))                         # self-reported (selection) metric
        held.append(float(f(th)+rng.normal(0,S_EVAL)))    # held-out re-eval (fresh noise)
        true.append(float(f(th)))                         # true objective
    return np.array(rep), np.array(held), np.array(true)

def stop_round(series, W):
    # first round t where series hasn't improved over its running max for W consecutive rounds
    best=-1e9; since=0
    for t,v in enumerate(series):
        if v>best+1e-9: best=v; since=0
        else: since+=1
        if since>=W: return t
    return len(series)-1   # never plateaued -> ran to the end

naive_r=[]; ver_r=[]; orac_r=[]; naive_true=[]; ver_true=[]; full_true=[]
for s in range(SEEDS):
    rep,held,true = episode(s)
    nr=stop_round(rep,W); vr=stop_round(held,W); orr=stop_round(true,W)
    naive_r.append(nr); ver_r.append(vr); orac_r.append(orr)
    naive_true.append(true[nr]); ver_true.append(true[vr]); full_true.append(true[-1])
ci=lambda x:1.96*np.std(x,ddof=1)/sqrt(len(x)); m=lambda x:float(np.mean(x))
print(f"d={D},K={K},TMAX={TMAX},eval-noise={S_EVAL},patience W={W},SEEDS={SEEDS}\n")
print(f"{'stop rule':>26} | {'rounds used':>18} | {'true f at stop':>16} | {'% budget used':>13}")
for name,r,t in [("NAIVE (self-metric)",naive_r,naive_true),("VERIFIED (held-out)",ver_r,ver_true)]:
    print(f"{name:>26} | {m(r):>8.1f} ± {ci(r):.1f} | {m(t):>9.3f} ± {ci(t):.3f} | {100*m(r)/TMAX:>11.0f}%")
print(f"{'ORACLE (true plateau, ref)':>26} | {m(orac_r):>8.1f} ± {ci(orac_r):.1f} | {'(reference)':>16} | {100*m(orac_r)/TMAX:>11.0f}%")
print(f"{'run-to-budget (no stop)':>26} | {TMAX:>8.1f}{'':>9} | {m(full_true):>9.3f} ± {ci(full_true):.3f} | {'100%':>11}")
import json; json.dump({"naive_rounds":m(naive_r),"verified_rounds":m(ver_r),"oracle_rounds":m(orac_r),
  "naive_true":m(naive_true),"verified_true":m(ver_true),"full_true":m(full_true),"TMAX":TMAX},
  open("runs/cycle040_verify_stopping/score.json","w"),indent=1)
print("\n--- read the rows above ---")
print(f"PRE-REG REFUTED: (1) naive self-metric DOES trigger early ({m(naive_r):.0f}/{TMAX}), not 'never'. (2) verified (held-out) stop is NO better ({m(ver_r):.0f} rounds, true {m(ver_true):.3f}) than naive ({m(naive_r):.0f}, {m(naive_true):.3f}).")
print(f"(3) CRUCIALLY both early-stops get WORSE true f than run-to-budget ({m(ver_true):.3f}/{m(naive_true):.3f} vs {m(full_true):.3f}) — the true objective is STILL slowly improving (no genuine plateau).")
print("=> HONEST NEGATIVE: 'verification as a stopping signal' fails when progress is slow-but-real (the tail regime) — a plateau-based stop on EITHER metric fires prematurely. To show stopping value needs a setup with a GENUINE plateau (true f flatlines while the self-metric climbs); this objective doesn't have one.")
print("DONE")
