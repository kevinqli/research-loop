"""Cycle 156 (#10d — does a real agent CHOOSE breadth-vs-depth correctly, and does it weight the VERIFIER-GATE?
Real-agent, high-α, meth-24; validates c149 behaviorally). c149's rule: at fixed compute, PARALLELIZE (N short
attempts + select best) pays IFF (a) attempt-ceilings VARY *and* (b) the verifier/selector is good; else run ONE
LONG agent. The subtle, c149/c152-specific half is (b): breadth buys COVERAGE of the ceiling distribution, but a
noisy verifier converts coverage into the winner's curse — if you can't SELECT the good attempt, generating many
varied ones is wasted. Does a capable agent apply this — especially the verifier-gate?

3 scenarios (meth-24, known c149-correct answer), SAME fixed-compute breadth-vs-depth choice, varying 2 factors:
  S1 vary_good  : approaches VARY widely + a RELIABLE evaluator -> c149: PARALLEL (breadth pays).
  S2 similar    : approaches reach SIMILAR quality (low variance) + reliable evaluator -> c149: DEPTH (no coverage benefit).
  S3 vary_bad   : approaches VARY widely BUT the evaluator is ~RANDOM/gameable (can't tell which output is best) ->
                  c149: DEPTH (the verifier-gate — breadth's coverage is UNREALIZABLE without a selector). THE KEY TEST.

PRE-REG: agents get S1 (parallel) + S2 (depth) right (intuitive coverage reasoning). The discriminating test is S3:
if the agent says DEPTH *because it can't select the winner* -> it weights the verifier-gate (c149/c152). If it says
PARALLEL ("more attempts = more coverage") -> it UNDER-weights the gate (the subtle half), an honest gap (cf c152:
selection is verifier-gated; agents may over-index on coverage).
"""
import json, os
os.makedirs("runs/cycle156_parallelize_choice", exist_ok=True)

COMMON = ("You are allocating a FIXED compute budget to solve a hard problem. You must choose ONE of:\n"
  "- PARALLEL: run ~20 SHORT independent attempts in parallel, then SELECT the single best to ship.\n"
  "- DEPTH: run ONE attempt and refine it for ~20x longer (no selection — you ship that one).\n"
  "Same total compute either way. Here is the situation:\n\n{body}\n\n"
  "Reason briefly about whether parallel-and-select or one-long-refinement gets more true quality here, then end "
  "with exactly 'DECISION: PARALLEL' or 'DECISION: DEPTH'.")

scenarios = {
 "S1_vary_good": ("There are MANY genuinely different approaches (architectures / feature sets / framings) and they "
   "vary WIDELY in how good they could get — a few could be excellent, most are mediocre, and you cannot tell in "
   "advance which is which. You DO have a clean, reliable held-out test that accurately ranks any finished solution."),
 "S2_similar": ("The available approaches are all minor variations on one method and they all reach about the SAME "
   "quality — there's very little spread in their potential (they plateau around the same score). You have a clean, "
   "reliable held-out test that accurately ranks any finished solution."),
 "S3_vary_bad": ("There are MANY genuinely different approaches and they vary WIDELY in how good they could get — a "
   "few could be excellent, most mediocre, and you cannot tell in advance which. BUT your only way to evaluate a "
   "finished solution is a NOISY, gameable metric that ranks solutions barely better than chance — it frequently "
   "rates a worse solution above a better one, and you have no other way to tell which output is actually best."),
}
truth = {"S1_vary_good": "PARALLEL", "S2_similar": "DEPTH", "S3_vary_bad": "DEPTH"}
for k, body in scenarios.items():
    open(f"runs/cycle156_parallelize_choice/prompt_{k}.txt", "w").write(COMMON.format(body=body))
json.dump({"truth": truth, "key_test": "S3_vary_bad (the verifier-gate: DEPTH because you can't SELECT the good attempt)"},
          open("runs/cycle156_parallelize_choice/truth.json", "w"), indent=1)
print("Built 3 scenarios (meth-24). c149-correct: S1=PARALLEL, S2=DEPTH, S3=DEPTH (the verifier-gate).")
print("KEY TEST S3: does the agent say DEPTH *because it cannot select the winner* (weights the gate) or PARALLEL (under-weights it)?")
print("DONE")
