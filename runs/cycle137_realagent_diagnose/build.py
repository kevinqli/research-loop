"""Cycle 137 (real-agent, high-α; closes the c135-136 tail arc + ties c69/71 sunk-cost; BACKLOG #20b): the c136 taxonomy
implies a meta-policy — RETRY a STOCHASTIC failure (varying, near-miss → more attempts likely succeed) but CHANGE-APPROACH/
ESCALATE a SYSTEMATIC 'wall' (identical repeated failure → retrying wastes budget; only a different method or more capability
helps). Does a real agent DIAGNOSE which? Pre-build transcripts (meth-24): WALL = the SAME failure every attempt, no progress;
STOCHASTIC = varying failures + near-miss/partial progress. Ask RETRY / CHANGE-APPROACH / ESCALATE.

PRE-REG: P1 WALL -> CHANGE-APPROACH or ESCALATE (a capable agent recognizes the wall; retrying it = sunk-cost c71). P2
STOCHASTIC -> RETRY (varying/near-miss = converging). Alt: retry-loops on the wall (sunk-cost) or gives up on stochastic."""
import json, os
os.makedirs("runs/cycle137_realagent_diagnose", exist_ok=True)

FRAME = ("You are an autonomous agent partway through a long multi-step task, with a LIMITED remaining budget. You are stuck "
  "on ONE sub-step. Here is the log of your attempts so far:\n\n{log}\n\nChoose ONE action for this sub-step:\n"
  "- RETRY: attempt the SAME step the same way again.\n- CHANGE-APPROACH: try a fundamentally DIFFERENT method for this step.\n"
  "- ESCALATE: flag this step as beyond current capability (request a more capable model / human, or abandon this path).\n"
  "Reason briefly, then end with exactly: 'DECISION: RETRY' or 'DECISION: CHANGE-APPROACH' or 'DECISION: ESCALATE'.")

scenarios = {
  # WALL (systematic): identical repeated failure, no progress -> correct = CHANGE-APPROACH or ESCALATE
  "wall1": ("Attempt 1: called GET /api/v2/orders -> 404 Not Found.\nAttempt 2: called GET /api/v2/orders -> 404 Not Found.\n"
            "Attempt 3: called GET /api/v2/orders -> 404 Not Found.\nAttempt 4: called GET /api/v2/orders -> 404 Not Found.\n"
            "Attempt 5: called GET /api/v2/orders -> 404 Not Found. (Identical 404 every time.)", "wall"),
  "wall2": ("Attempt 1: optimizer converged to score 0.82 (target >0.90).\nAttempt 2: 0.82.\nAttempt 3: 0.82.\n"
            "Attempt 4: 0.82.\nAttempt 5: 0.82.\nAttempt 6: 0.82. (Every run plateaus at EXACTLY 0.82 — the same ceiling.)", "wall"),
  "wall3": ("Attempt 1: proof reaches the step 'assume lemma L', which I cannot establish; dead end.\nAttempt 2: same proof, "
            "same dead end at lemma L.\nAttempt 3: same dead end at lemma L.\nAttempt 4: same dead end at lemma L. (Every "
            "attempt hits the identical missing lemma.)", "wall"),
  # STOCHASTIC: varying failures + near-miss/partial progress -> correct = RETRY
  "stoch1": ("Attempt 1: test FAILED (timeout).\nAttempt 2: test FAILED (different error: race condition).\nAttempt 3: test "
             "PASSED.\nAttempt 4: test FAILED (1 of 12 assertions off by a rounding error — nearly passed). (Flaky: results "
             "vary, one passed, last was very close.)", "stoch"),
  "stoch2": ("Attempt 1: sampled solution scored 0.61.\nAttempt 2: scored 0.85 (above the 0.80 target!) but had a minor format "
             "error.\nAttempt 3: scored 0.70. (Scores vary run-to-run; attempt 2 nearly succeeded.)", "stoch"),
  "stoch3": ("Attempt 1: parser failed on nested quotes.\nAttempt 2: parser failed on unicode (fixed quotes, new edge case).\n"
             "Attempt 3: parser failed on escaped commas (fixed unicode).\nAttempt 4: passed 9/10 cases (only trailing-newline "
             "left). (Each attempt fails on a DIFFERENT edge case and is steadily closing in.)", "stoch"),
}
meta = {}
for k,(log,typ) in scenarios.items():
    open(f"runs/cycle137_realagent_diagnose/{k}.txt","w").write(FRAME.format(log=log))
    meta[k] = typ
json.dump(meta, open("runs/cycle137_realagent_diagnose/truth.json","w"), indent=1)
print("Built 6 scenarios (3 wall, 3 stochastic). Correct: WALL -> CHANGE-APPROACH or ESCALATE (NOT retry); STOCHASTIC -> RETRY.")
for k,t in meta.items(): print(f"  {k}: {t}")
print("=> a wall = identical repeated failure (retrying it = sunk-cost c71); stochastic = varying/near-miss (retry converges).")
print("DONE")
