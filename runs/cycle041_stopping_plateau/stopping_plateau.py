"""Cycle 41: the PROPER verification-as-stopping test (fixes the cycle-40 design flaw — its objective had no genuine
plateau). Here the true objective RISES then FLATLINES at a ceiling (a genuine plateau at round P). The agent tracks
best-so-far by its NOISY self-metric (a ratchet = running max of noisy evals), which keeps climbing past the plateau
(winner's curse on the flat region). A held-out estimate (low-noise re-eval of the current solution) reveals the
flat true value. Stopping rule = patience W (stop if no improvement over the running max for W rounds).
  NAIVE   patience on the self-reported (ratchet) metric
  VERIFIED patience on the held-out estimate
Pre-reg: VERIFIED stops near the plateau onset P (~optimal); NAIVE's ratchet never plateaus -> runs to budget;
both reach the same true ceiling -> verified saves the post-plateau compute at equal true perf. numpy-only, SEEDS."""
import numpy as np
from math import sqrt
TMAX, P, K, S_EVAL, R_HELD, W, SEEDS = 50, 20, 20, 0.30, 8, 5, 200
def true_curve(t):  # rises -1 -> 0 over rounds 0..P, then FLAT 0 (genuine plateau / ceiling)
    return -1.0 + min(t, P)/P

def episode(seed):
    rng = np.random.default_rng(seed)
    true=np.array([true_curve(t) for t in range(TMAX+1)])
    # reported = ratchet: running max over rounds of (true + best-of-K selection noise)
    sel = true + np.array([np.max(rng.normal(0, S_EVAL, K)) for _ in range(TMAX+1)])
    reported = np.maximum.accumulate(sel)
    # held-out = low-noise estimate of the CURRENT true value (R averaged re-evals)
    held = true + rng.normal(0, S_EVAL/sqrt(R_HELD), TMAX+1)
    return true, reported, held

def stop_round(series, W):
    best=-1e9; since=0
    for t,v in enumerate(series):
        if v>best+1e-9: best=v; since=0
        else: since+=1
        if since>=W: return t
    return len(series)-1

nr=[]; vr=[]; nt=[]; vt=[]; ft=[]
for s in range(SEEDS):
    true,reported,held = episode(s)
    n=stop_round(reported,W); v=stop_round(held,W)
    nr.append(n); vr.append(v); nt.append(true[n]); vt.append(true[v]); ft.append(true[-1])
ci=lambda x:1.96*np.std(x,ddof=1)/sqrt(len(x)); m=lambda x:float(np.mean(x))
print(f"TMAX={TMAX}, plateau onset P={P}, K={K}, eval-noise={S_EVAL}, held re-evals R={R_HELD}, patience W={W}, SEEDS={SEEDS}\n")
print(f"{'stop rule':>26} | {'rounds used':>16} | {'true f at stop':>16} | {'% budget':>9}")
print(f"{'NAIVE (self-metric ratchet)':>26} | {m(nr):>8.1f} ± {ci(nr):.1f} | {m(nt):>9.3f} ± {ci(nt):.3f} | {100*m(nr)/TMAX:>7.0f}%")
print(f"{'VERIFIED (held-out)':>26} | {m(vr):>8.1f} ± {ci(vr):.1f} | {m(vt):>9.3f} ± {ci(vt):.3f} | {100*m(vr)/TMAX:>7.0f}%")
print(f"{'run-to-budget (no stop)':>26} | {TMAX:>8.1f}{'':>7} | {m(ft):>9.3f} ± {ci(ft):.3f} | {'100%':>9}")
import json; json.dump({"naive_rounds":m(nr),"verified_rounds":m(vr),"naive_true":m(nt),"verified_true":m(vt),
  "full_true":m(ft),"plateau_P":P,"TMAX":TMAX}, open("runs/cycle041_stopping_plateau/score.json","w"),indent=1)
print("\n--- read the rows above ---")
print(f"plateau onset P={P}. NAIVE ratchet stops PREMATURELY at {m(nr):.0f} (true {m(nt):.3f}, well below ceiling {m(ft):.2f}) — noise makes its running-max stall, a FALSE plateau signal -> quits early. VERIFIED (held-out) stops at {m(vr):.0f} (true {m(vt):.3f}, near ceiling).")
print(f"PRE-REG REFUTED: the ratchet does NOT over-run; it UNDER-runs (quits early). Verified used MORE rounds ({m(vr):.0f} vs {m(nr):.0f}) but reached a BETTER true value ({m(vt):.3f} vs {m(nt):.3f}).")
print("=> Corrected unified claim (with cycle 34): the self-metric is an unreliable stopping signal in BOTH directions — a running-max ratchet plateaus PREMATURELY (quit too early, here), a current-selection value climbs FOREVER (never quit, cycle 34). A low-noise HELD-OUT estimate tracks the true plateau either way. Verification's value for the tail = don't quit prematurely / don't thrash; know the true state.")
print("DONE")
