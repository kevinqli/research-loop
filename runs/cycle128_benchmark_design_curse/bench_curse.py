"""Cycle 128 (NEW application; the recursion one MORE level up, c7->c90->HERE): the BENCHMARK-DESIGN selection curse.
A benchmark builder picks the eval CONFIG (held-out size, metric, #passes...) that best SEPARATES known-good from
known-bad CALIBRATION agents. Selecting the best-APPARENT config by its calibration discrimination is itself a best-of-V
selection by a noisy metric -> the selected config's discrimination is WINNER'S-CURSED (overfit to the calibration agent
SAMPLE), so its TRUE discrimination on FRESH agents is lower than it looked. This is c90 (leaderboard=selector over AGENTS)
one level up: the BUILDER is a selector over CONFIGS. Honest: same order-statistic math as c7/c90 — the value is the new
OBJECT (your eval design), the meta-Goodhart framing (you overfit your benchmark to its calibration set), and the FIX.

Model: agents are GOOD (skill +δ) or BAD (skill −δ). Config v observes skill + N(0, σ_v); lower σ_v = better config.
TRUE discrimination of config v = AUC = P(good_score>bad_score) = 0.5*(1+erf(δ/σ_v)) (closed form). The builder measures
each config's APPARENT AUC on a finite CALIBRATION set of A agents (noisy), picks argmax, DEPLOYS it. Gap = apparent
(calibration) AUC of the selected config − its TRUE AUC. numpy-only, bootstrap CI.

PRE-REG: P1 gap>0, grows with V, shrinks with A. P2 gap larger when σ_v's are SIMILAR (configs hard to tell apart).
P3 a SEPARATE held-out calibration set gives an unbiased estimate of the deployed config's true AUC (the fix).
CONTROL: V=1 (no selection) -> gap≈0."""
import numpy as np, json, os, math
RNG = np.random.default_rng(0)
REPS = 4000
DELTA = 1.0
os.makedirs("runs/cycle128_benchmark_design_curse", exist_ok=True)

def true_auc(sigma):                       # closed form: P(good>bad), diff~N(2δ,2σ²) -> Φ(δ/σ)
    return 0.5 * (1 + math.erf(DELTA / (sigma * math.sqrt(2)) * math.sqrt(2)))  # = 0.5(1+erf(δ/σ))

def cal_auc(sigmas, A, reps, rng):
    """Apparent AUC of each config on a fresh calibration set of A agents (A/2 good, A/2 bad). -> (reps, V)."""
    ng = nb = A // 2
    V = len(sigmas)
    out = np.empty((reps, V))
    for v, s in enumerate(sigmas):
        good = DELTA + rng.normal(0, s, (reps, ng))      # good agents, config-v scores
        bad = -DELTA + rng.normal(0, s, (reps, nb))
        # AUC = mean over all good-bad pairs of [good>bad]
        out[:, v] = (good[:, :, None] > bad[:, None, :]).mean(axis=(1, 2))
    return out

def boot_ci(x, Bb=2000, seed=1):
    rng = np.random.default_rng(seed); n = len(x)
    m = np.array([x[rng.integers(0, n, n)].mean() for _ in range(Bb)])
    return float(x.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))

def run(V, A, sig_lo, sig_hi, reps=REPS, seed=0, label=""):
    rng = np.random.default_rng(seed)
    sigmas = np.linspace(sig_lo, sig_hi, V) if V > 1 else np.array([(sig_lo + sig_hi) / 2])
    trues = np.array([true_auc(s) for s in sigmas])
    cal = cal_auc(sigmas, A, reps, rng)                  # (reps, V) apparent AUCs
    sel = cal.argmax(1)                                  # builder picks best-apparent config
    r = np.arange(reps)
    apparent_sel = cal[r, sel]                           # naive reported (calibration) AUC
    true_sel = trues[sel]                                # its TRUE AUC on fresh agents
    gap = apparent_sel - true_sel
    # FIX: a SEPARATE held-out calibration set re-estimates the SELECTED config's AUC
    cal2 = cal_auc(sigmas, A, reps, np.random.default_rng(seed + 777))
    heldout_sel = cal2[r, sel]
    gap_fix = heldout_sel - true_sel
    g_m, g_lo, g_hi = boot_ci(gap)
    f_m, f_lo, f_hi = boot_ci(gap_fix)
    return {"label": label, "V": V, "A": A, "sig_range": [sig_lo, sig_hi],
            "true_best_auc": round(float(trues.max()), 4),
            "apparent_selected": round(float(apparent_sel.mean()), 4),
            "true_selected": round(float(true_sel.mean()), 4),
            "gap": round(g_m, 4), "gap_ci": [round(g_lo, 4), round(g_hi, 4)],
            "gap_AFTER_heldout_fix": round(f_m, 4), "fix_ci": [round(f_lo, 4), round(f_hi, 4)]}

out = {"model": {"REPS": REPS, "DELTA": DELTA}, "P1_vs_V": [], "P1_vs_A": [], "P2_similarity": [], "P3_fix": []}
print("=== P1a: gap vs #configs V (A=40 calibration agents, σ∈[0.8,1.6]) ===")
print(f"{'V':>4} | {'apparent':>9} {'true':>7} {'GAP':>8} | 95% CI {'':>10} | gap after held-out fix")
for V in [1, 3, 10, 25, 50]:
    r = run(V, 40, 0.8, 1.6, label=f"V={V}"); out["P1_vs_V"].append(r)
    print(f"{V:>4} | {r['apparent_selected']:>9} {r['true_selected']:>7} {r['gap']:>8} | [{r['gap_ci'][0]},{r['gap_ci'][1]}] | {r['gap_AFTER_heldout_fix']:>+.4f} [{r['fix_ci'][0]},{r['fix_ci'][1]}]")

print("\n=== P1b: gap vs calibration-set size A (V=25, σ∈[0.8,1.6]) ===")
print(f"{'A':>4} | {'GAP':>8} | 95% CI")
for A in [20, 40, 100, 400]:
    r = run(25, A, 0.8, 1.6, label=f"A={A}"); out["P1_vs_A"].append(r)
    print(f"{A:>4} | {r['gap']:>8} | [{r['gap_ci'][0]},{r['gap_ci'][1]}]")

print("\n=== P2: gap vs config SIMILARITY (V=25, A=40); tight σ-range = configs alike ===")
print(f"{'σ-range':>14} | {'GAP':>8} | 95% CI | (wider range = one config clearly best)")
for lo, hi, tag in [(1.0, 1.05, "tight[1.0,1.05]"), (0.9, 1.3, "med[0.9,1.3]"), (0.6, 2.0, "wide[0.6,2.0]")]:
    r = run(25, 40, lo, hi, label=tag); out["P2_similarity"].append(r)
    print(f"{tag:>14} | {r['gap']:>8} | [{r['gap_ci'][0]},{r['gap_ci'][1]}]")

json.dump(out, open("runs/cycle128_benchmark_design_curse/score.json", "w"), indent=1)
print("\n--- READ THE RAW ROWS (don't trust this summary) ---")
v = out["P1_vs_V"]
print(f"P1a gap vs V: " + " ".join(f"V{r['V']}:{r['gap']:+.4f}" for r in v) + "  (V=1 control should be ~0)")
print(f"P1a FIX (held-out) vs V: " + " ".join(f"V{r['V']}:{r['gap_AFTER_heldout_fix']:+.4f}" for r in v) + "  (should be ~0 at all V)")
a = out["P1_vs_A"]; print(f"P1b gap vs A: " + " ".join(f"A{r['A']}:{r['gap']:+.4f}" for r in a) + "  (should shrink with A)")
p2 = out["P2_similarity"]; print(f"P2 gap vs similarity: " + " ".join(f"{r['label'].split('[')[0]}:{r['gap']:+.4f}" for r in p2) + "  (bigger when configs alike)")
print("=> benchmark-design selection curse: picking the best-APPARENT eval config inflates its discrimination; fix = held-out calibration agents.")
print("DONE")
