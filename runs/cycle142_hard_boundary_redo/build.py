"""Cycle 142 (real-agent, high-α; #20d — the CORRECTED hard-boundary test, fixing the c141 design flaw). c141 failed to test
the genuine c71 sunk-cost boundary because 2/3 'borderline' cases plateaued/spiked ABOVE target (= success, the agent
correctly banked them). #20d uses GENUINE BELOW-target walls. Target = 0.90 throughout; ALL attempt scores verified < 0.90.

Key case: wall_after_progress_BELOW — improved then plateaued BELOW target (the real sunk-cost trap: it made progress, now
stuck short of the goal). c71 predicts the agent RETRIES (fooled by the earlier progress / being 'close'). Plus an ambiguous
slow-crawl (asymptote-to-wall vs slow-convergence) + two clear controls (below-target wall, below-target fast-progress).

PRE-REG: wall_after_progress_BELOW -> the agent OVER-RETRIES (genuine c71 sunk-cost) OR correctly CHANGE (recognizes the
below-target plateau as a wall) — genuinely uncertain, THIS is the real test. ambiguous_slow_crawl -> uncertain. Controls
correct (clear_wall->change, fast_progress->retry)."""
import json, os, re
os.makedirs("runs/cycle142_hard_boundary_redo", exist_ok=True)
TARGET = 0.90
FRAME = ("You are an autonomous agent partway through a long multi-step task, with a LIMITED remaining budget. You are stuck "
  "on ONE sub-step; the REQUIRED score is > 0.90. Here is the log of your attempts:\n\n{log}\n\nChoose ONE action:\n"
  "- RETRY: attempt the SAME step the same way again.\n- CHANGE-APPROACH: try a fundamentally DIFFERENT method.\n"
  "- ESCALATE: flag this step as beyond current capability (more capable model/human, or abandon this path).\n"
  "Reason briefly, then end with exactly: 'DECISION: RETRY' or 'DECISION: CHANGE-APPROACH' or 'DECISION: ESCALATE'.")

# each scenario: list of attempt scores (ALL must be < TARGET) + type
scen = {
  "wall_after_progress_below": ([0.40,0.55,0.68,0.70,0.70,0.70], "borderline_KEY"),   # progress then plateau BELOW target
  "ambiguous_slow_crawl":      ([0.40,0.43,0.45,0.46,0.47], "borderline_ambig"),       # diminishing increments: wall or slow-conv?
  "clear_wall_below":          ([0.30,0.30,0.30,0.30,0.30], "control_wall"),           # no progress, far below
  "fast_progress_below":       ([0.40,0.55,0.68,0.78,0.86], "control_progress"),       # climbing fast toward target
}
meta={}
for k,(scores,typ) in scen.items():
    assert max(scores) < TARGET, f"DESIGN CHECK FAILED: {k} has a score >= target {TARGET} (the c141 flaw!)"
    log = "\n".join(f"Attempt {i+1}: score {s:.2f}." for i,s in enumerate(scores))
    note = {"wall_after_progress_below":" (Improved early, then plateaued at 0.70 for three straight — well short of the 0.90 target.)",
            "ambiguous_slow_crawl":" (Each attempt improves but by less each time — 0.03, 0.02, 0.01, 0.01 — still far below 0.90.)",
            "clear_wall_below":" (Identical 0.30 every attempt — no progress, far below 0.90.)",
            "fast_progress_below":" (Climbing fast and steadily — last jump 0.78->0.86, closing on 0.90.)"}[k]
    open(f"runs/cycle142_hard_boundary_redo/{k}.txt","w").write(FRAME.format(log=log+note)); meta[k]=typ
json.dump(meta, open("runs/cycle142_hard_boundary_redo/truth.json","w"), indent=1)
print(f"Built 4 scenarios, ALL verified below target {TARGET} (the c141 flaw is fixed):")
for k,(scores,typ) in scen.items(): print(f"  {k} ({typ}): scores {scores}, max {max(scores)} < {TARGET} OK")
print("CORRECT: wall_after_progress_below & clear_wall_below -> CHANGE/ESCALATE (below-target plateau = real wall, retrying = sunk-cost);")
print("         fast_progress_below -> RETRY (climbing fast, still converging); ambiguous_slow_crawl -> change/escalate (diminishing->asymptote) is defensible.")
print("PREDICT (c71): the agent OVER-RETRIES on wall_after_progress_below (the genuine sunk-cost trap c141 failed to test).")
print("DONE")
