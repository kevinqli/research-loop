"""Cycle 102 (HARDEN/extend c82, long-horizon tail): does the checkpoint TYPE matter? c82 ('checkpoint throughout')
assumed IID per-step errors. Real long-horizon failure is often SYSTEMATIC DRIFT — a wrong early assumption makes ALL
downstream steps consistently-but-wrongly follow it; each step is LOCALLY CONSISTENT (the agent's own belief drifted
too), so a LOCAL checkpoint ('does this step follow from the last?') sees nothing. Only a GROUNDING checkpoint
('does this still match the original spec/target?') catches drift.

Model: 1-D state s should decay to target 0 (correct factor a_true=0.85 from s0=1). TWO error regimes:
 - IID: each step a small prob of a one-off random KICK (discontinuity).
 - DRIFT: at a random onset the agent's BELIEF flips to a diverging factor (a_drift=1.05) and stays — every step is
   exactly belief-consistent (locally fine) but the state diverges from target.
THREE checkpoint modes (every k steps): NONE; LOCAL (flag |s_t - belief*s_{t-1}| > tau -> correct the step, belief
UNCHANGED); GROUNDING (flag |s_t - correct_trajectory_t| > tau -> reset to correct traj + reset belief).
Metric: mean final |s_L| (deviation from target; lower=better) + drift-catch rate. numpy-only, REPS reps."""
import numpy as np, json, os
RNG = np.random.default_rng(0)
REPS, L, K = 20000, 40, 5
A_TRUE, A_DRIFT, S0 = 0.85, 1.05, 1.0
P_IID, KICK = 0.06, 0.6                  # iid: per-step kick prob + kick size
P_ONSET = 0.04                            # drift: per-step onset prob
TAU, OBSN = 0.35, 0.02                    # checkpoint threshold + small check observation noise
os.makedirs("runs/cycle102_drift_checkpoints", exist_ok=True)

def simulate(regime, mode):
    s = np.full(REPS, S0); belief = np.full(REPS, A_TRUE); drifted = np.zeros(REPS, bool)
    correct = np.full(REPS, S0)
    caught_drift = np.zeros(REPS, bool)
    for t in range(1, L+1):
        correct *= A_TRUE
        s_prev = s.copy()
        if regime == "drift":
            onset = (~drifted) & (RNG.random(REPS) < P_ONSET)
            belief = np.where(onset, A_DRIFT, belief); drifted = drifted | onset
            s = belief * s_prev                                   # agent follows its (maybe drifted) belief exactly
        else:  # iid
            s = A_TRUE * s_prev
            kick = RNG.random(REPS) < P_IID
            s = s + np.where(kick, RNG.normal(0, KICK, REPS), 0.0)
        if t % K == 0:
            noise = RNG.normal(0, OBSN, REPS)
            if mode == "local":
                flag = np.abs(s - belief*s_prev + noise) > TAU     # deviation from BELIEF-expected step
                s = np.where(flag, belief*s_prev, s)               # correct the step; belief UNCHANGED (key)
            elif mode == "grounding":
                flag = np.abs(s - correct + noise) > TAU           # deviation from the CORRECT trajectory
                caught_drift = caught_drift | (flag & drifted)
                s = np.where(flag, correct, s); belief = np.where(flag, A_TRUE, belief); drifted = drifted & ~flag
    return float(np.abs(s).mean()), float(caught_drift.mean()) if regime=="drift" else float('nan')

res = {"params": {"REPS":REPS,"L":L,"K":K,"A_TRUE":A_TRUE,"A_DRIFT":A_DRIFT,"P_IID":P_IID,"P_ONSET":P_ONSET,"TAU":TAU}, "results": {}}
print(f"L={L} steps, target=0, correct decay {A_TRUE}; checkpoint every k={K}; REPS={REPS}. Metric = mean final |s_L| (lower better).\n")
print(f"{'regime':>8} | {'NO check':>9} | {'LOCAL check':>12} | {'GROUNDING check':>15}")
for regime in ["iid", "drift"]:
    row = {}
    for mode in ["none","local","grounding"]:
        err, catch = simulate(regime, mode); row[mode] = {"final_err": round(err,4), "drift_catch": round(catch,3)}
    res["results"][regime] = row
    print(f"{regime:>8} | {row['none']['final_err']:>9.4f} | {row['local']['final_err']:>12.4f} | {row['grounding']['final_err']:>15.4f}")

json.dump(res, open("runs/cycle102_drift_checkpoints/score.json","w"), indent=1)
print("\n--- READ THE RAW ROWS (don't trust this summary) ---")
r = res["results"]
print(f"IID: local {r['iid']['local']['final_err']} ~ grounding {r['iid']['grounding']['final_err']} (both << none {r['iid']['none']['final_err']})? (P1)")
print(f"DRIFT: local {r['drift']['local']['final_err']} ~ none {r['drift']['none']['final_err']} (local MISSES drift)? grounding {r['drift']['grounding']['final_err']} << both? (P2)")
print(f"drift-catch rate: local {r['drift']['local']['drift_catch']} vs grounding {r['drift']['grounding']['drift_catch']} (grounding catches, local doesn't)")
print("DONE")
