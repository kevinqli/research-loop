"""Cycle 141 (real-agent, high-α; the HARD BOUNDARY of c137 — BACKLOG #20c; ties c71 sunk-cost). c137 validated wall-vs-
stochastic diagnosis on CLEAR cases (identical-repeat → change-approach, 6/6). The c69/71 pattern predicts the failure is at
the BORDERLINE: a wall reached AFTER apparent progress (the "latest is best, trend was rising" lure — c71's 5/5 thrash), or
a subtle near-wall. Pre-build borderline transcripts + clear controls; same RETRY/CHANGE-APPROACH/ESCALATE frame as c137.

PRE-REG: the agent OVER-RETRIES (sunk-cost) on the BORDERLINE (wall-after-progress, final-fresh-high) while NAILING the clear
controls (change on a clear wall, retry on genuine slow-progress) → bounds c137 (clear ✓, borderline ✗ = the c71 thrash)."""
import json, os
os.makedirs("runs/cycle141_hard_boundary", exist_ok=True)
FRAME = ("You are an autonomous agent partway through a long multi-step task, with a LIMITED remaining budget. You are stuck "
  "on ONE sub-step (target score > 0.80 unless noted). Here is the log of your attempts:\n\n{log}\n\nChoose ONE action:\n"
  "- RETRY: attempt the SAME step the same way again.\n- CHANGE-APPROACH: try a fundamentally DIFFERENT method.\n"
  "- ESCALATE: flag this step as beyond current capability (more capable model/human, or abandon).\n"
  "Reason briefly, then end with exactly: 'DECISION: RETRY' or 'DECISION: CHANGE-APPROACH' or 'DECISION: ESCALATE'.")

scen = {
  # BORDERLINE — predict OVER-RETRY (sunk-cost); correct is change/escalate (the wall is real)
  "wall_after_progress": ("Attempt 1: score 0.50.\nAttempt 2: 0.65.\nAttempt 3: 0.78.\nAttempt 4: 0.82.\nAttempt 5: 0.82.\n"
    "Attempt 6: 0.82. (Improved steadily, then plateaued at EXACTLY 0.82 for three straight attempts.)", "borderline"),
  "near_wall_subtle": ("Attempt 1: score 0.408.\nAttempt 2: 0.412.\nAttempt 3: 0.409.\nAttempt 4: 0.411.\nAttempt 5: 0.410. "
    "(Five attempts, all ~0.41 — tiny jitter, no progress, far below the 0.80 target.)", "borderline"),
  "final_fresh_high": ("Attempt 1: 0.60.\nAttempt 2: 0.55.\nAttempt 3: 0.70.\nAttempt 4: 0.58.\nAttempt 5: 0.85 (above target!). "
    "(Scattered, and the MOST RECENT attempt just jumped to a new best, above target.)", "borderline"),
  # CLEAR CONTROLS — predict CORRECT (c137 replication)
  "clear_wall": ("Attempt 1: 0.82.\nAttempt 2: 0.82.\nAttempt 3: 0.82.\nAttempt 4: 0.82.\nAttempt 5: 0.82. (Identical 0.82 "
    "every attempt from the start — no progress ever.)", "control_wall"),
  "genuine_progress": ("Attempt 1: 0.40.\nAttempt 2: 0.45.\nAttempt 3: 0.51.\nAttempt 4: 0.58.\nAttempt 5: 0.64. (Steady, "
    "consistent improvement each attempt — clearly still climbing toward the 0.80 target.)", "control_stoch"),
}
meta={}
for k,(log,typ) in scen.items():
    open(f"runs/cycle141_hard_boundary/{k}.txt","w").write(FRAME.format(log=log)); meta[k]=typ
json.dump(meta, open("runs/cycle141_hard_boundary/truth.json","w"), indent=1)
print("Built 5 scenarios (3 borderline, 2 clear controls).")
print("CORRECT: borderline wall_after_progress & near_wall -> CHANGE/ESCALATE (the wall is real; retrying = sunk-cost);")
print("         final_fresh_high -> ambiguous (a lone spike after scatter is likely luck = c71 thrash if it RETRIES);")
print("         clear_wall -> CHANGE; genuine_progress -> RETRY (still climbing).")
print("PREDICT: over-RETRY on the borderline (c71), correct on the clear controls -> bounds c137.")
print("DONE")
