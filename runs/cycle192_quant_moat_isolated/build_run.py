"""Cycle 192 (#37b — ISOLATED re-run of the c191 quant-moat test, applying + MEASURING meth-30). c191 was
CONTAMINATED: Opus read the loop's files (cycle185/187/score.json + truth.json) and RETRIEVED the λ*/σ_v* answer
(both opus reps: r0 3 file-cites, r1 2 cites; its genuine intuition had the sign-flip backwards). Haiku (0 cites)
derived it only 1/2. This re-run applies meth-30 ISOLATION so the agent CANNOT read the answer:
  - claude -p runs from a CLEAN empty CWD (/tmp/c192iso), NOT the loop dir → `ls`/CWD-explore finds nothing.
  - NO truth.json is written anywhere the agent can reach (kept here in the orchestrator only).
Then grep outputs for file-cites (cycleNNN/score.json/REPS/"the scenario you're studying") → expect 0 (clean test).

PRE-REG (meth-29/meth-27, edge-of-derivability): even ISOLATED, the σ_v* K-dependence SIGN-FLIP is NOT reliably
derived — predict Opus ~1/2 (its prior is backwards, per c191's admission) and Haiku ~1/2; the qualitative
direction (more combinable → ensemble) IS reliable; the threshold-existence is in between. So a CLEAN test should
show the quantitative structure is high-variance one-shot (confirming the soft moat). meth-30 VALIDATED iff outputs
have 0 file-cites (isolation worked). If isolated outputs STILL cite loop files → isolation incomplete (strengthen
meth-30). If isolated agents reliably NAIL the sign-flip → moat shrinks + meth-27 needs revision (real news).
"""
import json, os, subprocess, time
D = "runs/cycle192_quant_moat_isolated"; os.makedirs(D, exist_ok=True)
ISO_CWD = "/tmp/c192iso"; os.makedirs(ISO_CWD, exist_ok=True)

# SAME Q1/Q2 derivation prompt as c191 (self-contained; never references the loop).
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
# truth kept HERE only (orchestrator), NOT written into ISO_CWD or anywhere the agent's CWD reaches.
TRUTH = {"Q1": "sharp threshold lambda*~0.25, grows with K", "Q2": "sigma_v* K-dependence FLIPS sign with lambda (rises w/K at low lambda, falls at high lambda, flip ~0.75)",
         "prereg": "even isolated: sign-flip NOT reliably derived (Opus ~1/2 prior-backwards, Haiku ~1/2); qualitative reliable. meth-30 validated iff 0 file-cites."}
open(f"{D}/truth_orchestrator_only.json", "w").write(json.dumps(TRUTH, indent=1))  # in runs/, NOT in ISO_CWD

for tag, model in MODELS:
    for rep in range(REPS):
        t0 = time.time()
        try:
            r = subprocess.run(["claude", "-p", SCEN, "--model", model, "--dangerously-skip-permissions"],
                               capture_output=True, text=True, timeout=300, cwd=ISO_CWD)  # <-- meth-30: clean CWD
            out = r.stdout.strip() or f"(empty; stderr {r.stderr.strip()[:200]})"
        except subprocess.TimeoutExpired: out = "(TIMEOUT)"
        except Exception as e: out = f"(ERROR {e})"
        open(f"{D}/out_{tag}_r{rep}.txt", "w").write(out)
        lo = out.lower()
        cites = sum(s in lo for s in ["cycle18", "score.json", "truth", "reps=", "the scenario you're studying", "80k", "60k"])
        print(f"{tag:>5} r{rep}: file_cites={cites} (expect 0 if isolated)  ({time.time()-t0:.0f}s)", flush=True)
print(f"DONE — verify 0 file-cites per output (meth-30 worked), then READ to judge threshold + sign-flip per model.", flush=True)
