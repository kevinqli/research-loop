"""Cycle 110 (compose c8 + c39 + winner's curse): WHERE to spend a verification budget — on the SUSPECT (low-
confidence) candidates, or the TOP (highest-confidence = what you'll ship)? Intuition says 'verify the suspect ones';
this tests whether that's BACKWARDS. The winner's curse means you SHIP the highest-confidence candidate — and the
dangerous fakes are the SNEAKY ones with HIGH confidence (they fooled the agent, c39 mid-range blindness), which is
exactly what wins selection. So verify what you'll SHIP (the top), not what you won't (the suspect bottom).

Model: K candidates; n_fake fakes (true q≈0), n_real reals (q~U(.4,.8)). Confidence: real ~N(.6,.15); fake mostly low
~N(.2,.15) but a fraction f_sneaky get HIGH confidence ~N(.6,.15) (indistinguishable from real). You SHIP the
highest-confidence surviving candidate. Verification budget B: perfectly check B candidates (a checked fake is dropped
+ you re-select). Strategies: VERIFY-TOP-B / VERIFY-BOTTOM-B / VERIFY-RANDOM-B (by confidence). Metric: true quality of
the SHIPPED candidate (higher=better). numpy-only, REPS reps."""
import numpy as np, json, os
RNG = np.random.default_rng(0)
REPS, K, N_REAL, B = 8000, 50, 15, 8
os.makedirs("runs/cycle110_verify_top_not_suspect", exist_ok=True)

def trial(f_sneaky, strategy):
    out = np.zeros(REPS)
    for rep in range(REPS):
        is_real = np.zeros(K, bool); is_real[:N_REAL] = True
        q = np.where(is_real, RNG.uniform(.4,.8,K), 0.0)              # true quality
        conf = np.where(is_real, RNG.normal(.6,.15,K), RNG.normal(.2,.15,K))
        # sneaky fakes: a fraction of fakes get HIGH confidence
        fakes = np.where(~is_real)[0]
        nsneak = int(round(f_sneaky*len(fakes)))
        sneak = RNG.choice(fakes, nsneak, replace=False)
        conf[sneak] = RNG.normal(.6,.15,nsneak)
        order = np.argsort(-conf)                                     # by confidence desc
        if strategy=="none": checked=set()
        elif strategy=="top": checked=set(order[:B].tolist())
        elif strategy=="bottom": checked=set(order[-B:].tolist())
        else: checked=set(RNG.choice(K,B,replace=False).tolist())
        # ship highest-confidence candidate that is NOT a checked-and-found-fake
        for idx in order:
            if idx in checked and not is_real[idx]:  # checked fake -> dropped
                continue
            out[rep]=q[idx]; break                   # ship this one (highest conf surviving)
    return float(out.mean())

res={"model":{"REPS":REPS,"K":K,"N_REAL":N_REAL,"B":B},"sweep":{}}
print(f"K={K} ({N_REAL} real / {K-N_REAL} fake), verify budget B={B}, REPS={REPS}. Metric=true quality of SHIPPED candidate.\n")
print(f"{'f_sneaky':>9} | {'no verify':>10} | {'VERIFY-BOTTOM':>14} | {'VERIFY-RANDOM':>14} | {'VERIFY-TOP':>11}")
for fs in [0.1, 0.3, 0.6]:
    nv=trial(fs,"none"); bot=trial(fs,"bottom"); rnd=trial(fs,"random"); top=trial(fs,"top")
    res["sweep"][fs]={"no_verify":round(nv,4),"bottom":round(bot,4),"random":round(rnd,4),"top":round(top,4)}
    print(f"{fs:>9} | {nv:>10.4f} | {bot:>14.4f} | {rnd:>14.4f} | {top:>11.4f}")

json.dump(res,open("runs/cycle110_verify_top_not_suspect/score.json","w"),indent=1)
print("\n--- READ THE RAW ROWS (don't trust this summary) ---")
s=res["sweep"]
print(f"P1 @f_sneaky=0.3: TOP {s[0.3]['top']} vs RANDOM {s[0.3]['random']} vs BOTTOM {s[0.3]['bottom']} vs none {s[0.3]['no_verify']} (top>>random>bottom~none?)")
print(f"P2 TOP-minus-BOTTOM gap vs sneaky: " + " ".join(f"{fs}:{round(s[fs]['top']-s[fs]['bottom'],3)}" for fs in [0.1,0.3,0.6]) + " (grows with sneaky-fakes?)")
print("=> verify the TOP (what you ship), NOT the suspect bottom; the winner's curse hides in HIGH confidence. Suspicion-triage is backwards.")
print("DONE")
