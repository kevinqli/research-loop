"""Cycle 94 (PIVOT to the least-mined SCAFFOLD pillar; NEW-mechanism per meth-22): a scaffold's optimal ASPECT RATIO
(breadth W vs depth D) under a FIXED compute budget B=W*D — and does it UNIFY the two verification roles onto the two
axes? BREADTH = explore W diverse approaches then SELECT best (c90 winner's-curse risk, set by eval-noise sigma);
DEPTH = refine one approach over D steps (c82 propagation risk, set by per-step error p). Fixed budget => trade one
for the other.

Model (per branch i): latent approach quality a_i ~ N(0, sig_a^2) [breadth explores these]; refinement progress =
delta * useful_steps, useful = min(first-error-step, D) under per-step error p [depth accumulates but CORRUPTS:
expected useful saturates at ~1/p regardless of D, c82]. true q_i = a_i + delta*useful_i. observed o_i = q_i +
N(0,sigma^2) [eval noise]. Scaffold picks argmax(o) over W branches; OUTCOME = true q of the selected branch.
Sweep W (D=B//W) for various (p, sigma). numpy-only, REPS reps. Predicts D*~1/p (=> W*~B*p), W* falls with sigma."""
import numpy as np, json, os
RNG = np.random.default_rng(0)
REPS, B = 8000, 240
SIG_A, DELTA = 1.0, 0.1
WS = [1,2,3,4,5,6,8,10,12,15,16,20,24,30,40,48,60,80,120,240]   # divisors of B=240
os.makedirs("runs/cycle094_scaffold_aspect_ratio", exist_ok=True)

def outcome(W, p, sigma):
    D = B//W
    a = RNG.normal(0, SIG_A, (REPS, W))
    g = RNG.geometric(p, (REPS, W))                              # first-error step (1-indexed)
    useful = np.clip(g-1, 0, D)
    q = a + DELTA*useful                                          # true quality
    o = q + RNG.normal(0, sigma, (REPS, W))                       # observed (eval noise)
    r = np.arange(REPS); win = o.argmax(axis=1)
    out = q[r, win]                                               # TRUE quality of selected
    wc_gap = float((o[r, win] - q[r, win]).mean())                # winner's-curse gap (observed - true), breadth risk
    corrupt_waste = float((D - useful).mean())                    # wasted depth steps (propagation), depth risk
    return float(out.mean()), 1.96*float(out.std()/np.sqrt(REPS)), wc_gap, corrupt_waste

def best_W(p, sigma):
    rows = [(W, *outcome(W, p, sigma)) for W in WS]
    star = max(rows, key=lambda x: x[1])
    return star[0], star[1], rows

out = {"model": {"REPS":REPS,"B":B,"SIG_A":SIG_A,"DELTA":DELTA}, "A_curve": {}, "B_vs_p": {}, "C_vs_sigma": {}}

# ===== PART A: outcome vs W at baseline (p=0.05, sigma=0.5) — show interior optimum + the two role-risks =====
p0, s0 = 0.05, 0.5
print(f"=== A. OUTCOME vs breadth W  (budget B={B}, D=B/W; p={p0}, eval-noise sigma={s0}) ===")
print(f"{'W':>4} {'D':>4} | {'true outcome':>12} {'95%CI':>7} | {'WC gap(breadth)':>15} | {'corrupt waste(depth)':>20}")
Wstar0, _, rows0 = best_W(p0, s0)
for W, m, ci, wc, cw in rows0:
    D=B//W; star = " *" if W==Wstar0 else ""
    out["A_curve"][W] = {"D":D,"outcome":round(m,4),"ci":round(ci,4),"wc_gap":round(wc,4),"corrupt_waste":round(cw,3)}
    print(f"{W:>4} {D:>4} | {m:>12.4f} {ci:>7.4f} | {wc:>15.4f} | {cw:>20.3f}{star}")
print(f"  -> interior optimum W*={Wstar0} (D*={B//Wstar0}); WC-gap GROWS with W (breadth=selection risk), corrupt-waste GROWS with D (depth=propagation risk)\n")

# ===== PART B: W* vs per-step error rate p (sigma fixed) — predict W* grows ~B*p (D*~1/p) =====
print(f"=== B. Optimal W* vs per-step error rate p  (sigma={s0})  [predict D*~1/p, so W*~B*p] ===")
print(f"{'p':>6} | {'W*':>4} {'D*':>4} | {'1/p':>6} {'B*p':>6} | {'outcome@W*':>11}")
for p in [0.02, 0.05, 0.10, 0.20, 0.40]:
    Ws, m, _ = best_W(p, s0); out["B_vs_p"][p] = {"Wstar":Ws,"Dstar":B//Ws,"outcome":round(m,4)}
    print(f"{p:>6} | {Ws:>4} {B//Ws:>4} | {1/p:>6.1f} {B*p:>6.1f} | {m:>11.4f}")

# ===== PART C: W* vs eval noise sigma (p fixed) — predict W* falls with sigma (noisy selection wastes breadth) =====
print(f"\n=== C. Optimal W* vs eval noise sigma  (p={p0})  [predict W* FALLS as sigma rises] ===")
print(f"{'sigma':>6} | {'W*':>4} {'D*':>4} | {'outcome@W*':>11}")
for sg in [0.1, 0.3, 0.6, 1.0, 2.0]:
    Ws, m, _ = best_W(p0, sg); out["C_vs_sigma"][sg] = {"Wstar":Ws,"Dstar":B//Ws,"outcome":round(m,4)}
    print(f"{sg:>6} | {Ws:>4} {B//Ws:>4} | {m:>11.4f}")

json.dump(out, open("runs/cycle094_scaffold_aspect_ratio/score.json","w"), indent=1)
print("\n--- READ THE RAW ROWS ABOVE (don't trust this summary) ---")
print(f"A: interior W*={Wstar0} of {len(WS)} options (P1: not 1 and not {B})")
print(f"B: W* vs p: " + " ".join(f"p{p}:W{out['B_vs_p'][p]['Wstar']}" for p in [0.02,0.05,0.10,0.20,0.40]) + "   (P2: W* grows with p, D*~1/p)")
print(f"C: W* vs sigma: " + " ".join(f"s{sg}:W{out['C_vs_sigma'][sg]['Wstar']}" for sg in [0.1,0.3,0.6,1.0,2.0]) + "   (P3: W* falls as sigma rises)")
print("DONE")
