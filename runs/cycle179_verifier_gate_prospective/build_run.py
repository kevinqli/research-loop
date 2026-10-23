"""Cycle 179 (#33 — the CLEAN prospective test of the operational novel-vs-famous criterion, committed in c178).
c178 found c177's d'-stringency is RECOVERABLE by both models because it maps to a SINGLE named framework (SDT) -> FAMOUS;
that reclassification was POST-HOC. This tests the OPPOSITE prediction on an unambiguous novel-COMPOSITION: c156's
VERIFIER-GATE ("a near-random selector negates parallel breadth's coverage" = coverage x selection-quality, NO single
textbook home). FRESH surface (BREADTH/DEPTH framing, reworded selector descriptions; NOT verbatim c156) to avoid
memorization; UNPROMPTED (no hint that the selector is the key consideration -> avoids the c178 confound). meth-24.

DESIGN: S1 and S3 are IDENTICAL except the SELECTOR RELIABILITY (the only varied fact). The discriminator = does the
model CONDITION its breadth/depth choice on selector quality? S1 (reliable selector) -> BREADTH; S3 (near-random
selector) -> DEPTH (the gate: breadth degenerates to pick-at-random, so refine instead). 3 reps/cell (vs c178's n=1).

PRE-REG: OPUS differentiates (S1->BREADTH, S3->DEPTH, catches the gate); HAIKU misses (BREADTH for BOTH, anchored on the
quality-variance, under-weighting selector reliability) = the c156 split, now PROSPECTIVE. If the SPLIT holds -> the
"novel-COMPOSITION -> Opus-only" operational criterion is VALIDATED (can be methodology-encoded). If BOTH differentiate
(both S3->DEPTH) -> the criterion is REFUTED, the moat-novelty thesis needs rethinking. Honest + decisive either way.
"""
import json, os, re, subprocess, time
D = "runs/cycle179_verifier_gate_prospective"
os.makedirs(D, exist_ok=True)

COMMON = ("You manage a FIXED compute budget for one hard task. Two ways to spend it (same total compute either way):\n"
"- BREADTH: launch ~20 short independent attempts, then keep the single best one.\n"
"- DEPTH: run just ONE attempt but iterate/refine it ~20x longer, and keep that one.\n"
"Decide which yields higher TRUE quality in the situation below. Reason briefly, then end with exactly "
"'CHOICE: BREADTH' or 'CHOICE: DEPTH'.\n\nSituation: ")

SCENARIOS = {
 "S1_reliable": ("the candidate approaches differ a LOT in their potential — a few are excellent, most are so-so, and "
   "you can't predict in advance which is which. When an attempt finishes, you can score it with an evaluation that is "
   "accurate and trustworthy: it reliably ranks finished attempts by their true quality."),
 "S3_random":   ("the candidate approaches differ a LOT in their potential — a few are excellent, most are so-so, and "
   "you can't predict in advance which is which. When an attempt finishes, the only score you can compute is almost "
   "uninformative: it ranks finished attempts only very slightly better than a coin flip (it very often rates a worse "
   "attempt above a better one)."),
}
TRUTH = {"S1_reliable": "BREADTH", "S3_random": "DEPTH"}
for k, s in SCENARIOS.items():
    open(f"{D}/prompt_{k}.txt", "w").write(COMMON + s)
json.dump({"truth": TRUTH, "discriminator": "does the model CONDITION breadth/depth on SELECTOR reliability? "
           "S1 reliable->BREADTH, S3 near-random->DEPTH (the verifier-gate). Opus expected to differentiate; Haiku to miss (BREADTH both).",
           "prereg": "split holds (Opus differentiates, Haiku misses) -> composition->Opus-only criterion VALIDATED; both differentiate -> REFUTED"},
          open(f"{D}/truth.json", "w"), indent=1)

MODELS = [("haiku", "claude-haiku-4-5-20251001"), ("opus", "claude-opus-4-8")]
REPS = 3
def parse(out):
    m = re.findall(r"CHOICE:\s*(BREADTH|DEPTH)", out, re.I)
    return m[-1].upper() if m else "?"

results = {}
for tag, model in MODELS:
    results[tag] = {}
    for k in SCENARIOS:
        prompt = COMMON + SCENARIOS[k]
        choices = []
        for rep in range(REPS):
            t0 = time.time()
            try:
                r = subprocess.run(["claude", "-p", prompt, "--model", model, "--dangerously-skip-permissions"],
                                   capture_output=True, text=True, timeout=150)
                out = r.stdout.strip() or f"(empty; stderr {r.stderr.strip()[:200]})"
            except subprocess.TimeoutExpired:
                out = "(TIMEOUT)"
            except Exception as e:
                out = f"(ERROR {e})"
            c = parse(out)
            choices.append(c)
            open(f"{D}/out_{tag}_{k}_r{rep}.txt", "w").write(out)
            print(f"{tag:>5} {k:>12} r{rep}: {c}  ({time.time()-t0:.0f}s)")
        results[tag][k] = choices
    print()

# tally + discriminator
summary = {}
for tag in results:
    s1 = results[tag]["S1_reliable"]; s3 = results[tag]["S3_random"]
    s1_breadth = s1.count("BREADTH"); s3_depth = s3.count("DEPTH")
    differentiates = (s1_breadth >= 2) and (s3_depth >= 2)   # majority-correct on BOTH = catches the gate
    summary[tag] = {"S1_reliable": s1, "S3_random": s3, "S1_breadth_n": s1_breadth, "S3_depth_n": s3_depth,
                    "differentiates_on_selector": differentiates}
    print(f"{tag}: S1 {s1} ({s1_breadth}/3 BREADTH=correct) | S3 {s3} ({s3_depth}/3 DEPTH=correct) "
          f"-> differentiates-on-selector(gate): {differentiates}")
json.dump({"truth": TRUTH, "REPS": REPS, "results": results, "summary": summary}, open(f"{D}/score.json", "w"), indent=1)
print("\nPRE-REG: Opus differentiates (S1 breadth, S3 depth); Haiku misses (breadth both). "
      "Split -> composition->Opus-only VALIDATED; both differentiate -> REFUTED.")
print("DONE")
