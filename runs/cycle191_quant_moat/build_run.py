"""Cycle 191 (REAL-AGENT, #37 — the QUANTITATIVE-moat test + FIRST APPLICATION of meth-29: set the pre-reg via
meth-27, not the naive weaker-fails default. Also measures meth-29 [measure-before-keep]. claude -p free).

c188: both tiers recover the QUALITATIVE combinability->strategy call (average-combinable, select-and-discount-tail)
= famous -> capability-FLAT. The c98/99 moat claim: the loop's add is the QUANTITATIVE STRUCTURE neither tier derives
— (i) a combinability THRESHOLD λ*≈0.25 below which SELECT always wins (grows with K), and (ii) the σ_v* crossover's
K-dependence FLIPS SIGN with λ (rises with K at low λ, falls at high λ; flip ~0.75) [c185/c187].

PRE-REG via meth-29/meth-27 (NOT the naive default): classify the targets.
  - Q1 QUALITATIVE ("more combinable -> ensemble favored"): FAMOUS (ensemble-vs-select) -> predict BOTH tiers get it.
  - Q-threshold (is there a SHARP λ*? where?) and Q-signflip (does the K-direction FLIP with λ?): these are LOOP-NOVEL
    derived structure with NO textbook home -> meth-27 predicts NEITHER tier derives them reliably (beyond recall):
    they hand-wave a smooth transition and give a UNIFORM-direction answer for K, MISSING the sign-flip.
PRE-REG OUTCOME: agents confirm the qualitative direction but do NOT nail (a) the threshold-existence/location or
(b) the K-dependence sign-flip -> CONFIRMS the quantitative moat (c98/99) AND validates meth-29's calibration. If an
agent DOES derive the sign-flip -> the moat shrinks AND meth-27/meth-29 need revision (real news, meth-29 working as
a tripwire). [Contrast-only naive prior: 'Opus derives it, Haiku doesn't' — NOT my prediction; meth-27 says neither.]
"""
import json, os, subprocess, time
D = "runs/cycle191_quant_moat"; os.makedirs(D, exist_ok=True)

SCEN = (
"Setup: you run K parallel agents; each returns one candidate. A candidate's value has two parts — a TAIL part "
"(you keep only the single BEST; averaging destroys it) and a COMBINABLE part (a numeric estimate whose error "
"AVERAGING reduces). Let lambda in [0,1] be the fraction of value that is combinable. A noisy verifier (noise sigma_v) "
"scores each candidate; you then either SELECT the top-scored candidate, or ENSEMBLE (average all K). Answer "
"precisely, with reasoning:\n"
"(1) As lambda rises from 0, is there a SHARP THRESHOLD lambda* below which SELECT beats ENSEMBLE at EVERY sigma_v "
"(no crossover), or is it a smooth transition? If a threshold, roughly where is lambda*, and does it move with K?\n"
"(2) Hold lambda fixed and INCREASE K (more parallel agents). Does the verifier-noise sigma_v* at which ENSEMBLE "
"overtakes SELECT go UP or DOWN with K — and does that DIRECTION depend on lambda? Be specific about the direction(s).")

MODELS = [("haiku", "claude-haiku-4-5-20251001"), ("opus", "claude-opus-4-8")]
REPS = 2
open(f"{D}/prompt.txt", "w").write(SCEN)
json.dump({"truth": {"Q1_threshold": "YES a sharp threshold lambda*~0.25 below which SELECT always wins (no crossover); lambda* GROWS with K",
                     "Q2_signflip": "the K-dependence of sigma_v* FLIPS SIGN with lambda: sigma_v* RISES with K at low lambda (tail-grab dominates), FALLS with K at high lambda (variance-reduction dominates); flip ~lambda 0.75"},
           "discriminator": "Q1: asserts a SHARP threshold (not smooth) + ballpark + grows-with-K; Q2: gets the SIGN-FLIP (direction depends on lambda), not a uniform direction",
           "prereg_meth29": "via meth-27: qualitative direction FAMOUS->both get it; threshold-existence + K-signflip are LOOP-NOVEL->NEITHER tier derives them (hand-wave smooth + uniform K-direction). NOT 'Opus yes Haiku no' (that's the naive contrast)."},
          open(f"{D}/truth.json", "w"), indent=1)

for tag, model in MODELS:
    for rep in range(REPS):
        t0 = time.time()
        try:
            r = subprocess.run(["claude", "-p", SCEN, "--model", model, "--dangerously-skip-permissions"],
                               capture_output=True, text=True, timeout=300)
            out = r.stdout.strip() or f"(empty; stderr {r.stderr.strip()[:200]})"
        except subprocess.TimeoutExpired: out = "(TIMEOUT)"
        except Exception as e: out = f"(ERROR {e})"
        open(f"{D}/out_{tag}_r{rep}.txt", "w").write(out)
        lo = out.lower()
        threshold = any(s in lo for s in ["threshold", "sharp", "critical", "lambda*", "λ*", "no crossover", "abrupt", "phase"])
        signflip = (("depend" in lo and ("lambda" in lo or "λ" in lo)) and ("up" in lo or "down" in lo or "rise" in lo or "fall" in lo or "increase" in lo or "decrease" in lo))
        print(f"{tag:>5} r{rep}: threshold-word~{threshold} signflip-hint~{signflip}  ({time.time()-t0:.0f}s)", flush=True)
print("DONE — READ out_*.txt: does either model (a) assert a SHARP threshold + grows-with-K, (b) get the K-direction SIGN-FLIP? (don't trust the scan).", flush=True)
