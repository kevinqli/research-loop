"""Cycle 173 (#29b — does a real agent place checkpoints by DIFFICULTY (c169) or default to even? Real-agent
validation of c169/c172; high-α, meth-24; diversified mode off 3 sim HARDENs). c169: with KNOWN hard steps, place a
check RIGHT AFTER each hard step (4.3x less wasted work than even); c82: under UNIFORM difficulty, even is optimal.
Does a capable agent ADAPT its checkpoint placement to the difficulty profile, or use one fixed heuristic (even/periodic)?

2 scenarios (meth-24, known correct), same 200-step task + 8-checkpoint budget; an uncaught error PROPAGATES downstream
until the next checkpoint:
  S1 HETEROGENEOUS: steps 40, 90, 140 are KNOWN hard/error-prone; rest routine -> c169-correct: a check RIGHT AFTER
     each hard step (~41/91/141, catch its error before it propagates) + spread the rest. NOT even, NOT all-at-end.
  S2 UNIFORM (control): all steps equally routine, no known bottlenecks -> c82-correct: EVEN spacing (~every 25).

PRE-REG: the discriminating test is ADAPTATION — does the agent place checks AFTER the hard steps in S1 AND evenly in
S2 (adapts), or use one fixed heuristic for both? Predict: capable model adapts (S1 after-hard, S2 even) + reasons about
PROPAGATION (catch the hard step's error before it spreads); weaker may default to even/periodic in BOTH (misses the
difficulty-aware placement = the c169 subtle half), or place AT the hard step (40) rather than AFTER (41) (the position
subtlety: you must check AFTER the step executes to catch its error).
"""
import json, os
os.makedirs("runs/cycle173_realagent_checkpoint", exist_ok=True)

COMMON = ("You are running a 200-step autonomous task. You can afford only 8 CHECKPOINTS (each re-verifies all work so "
  "far; if it finds an error it's fixed, otherwise the error PROPAGATES and corrupts every downstream step until the "
  "next checkpoint catches it). You must choose WHERE (which of steps 1-200) to place your 8 checkpoints.\n\n{body}\n\n"
  "Reason briefly about where errors are most costly to leave uncaught, then end with exactly 'POSITIONS: [s1, s2, ..., s8]' "
  "(8 step-numbers).")

scenarios = {
 "S1_heterogeneous": ("You KNOW that steps 40, 90, and 140 are HARD and error-prone (high chance of an error there); "
   "all other steps are routine and very reliable."),
 "S2_uniform": ("Every step is equally routine and reliable — there are no known hard or error-prone steps; the error "
   "risk is uniform across all 200 steps."),
}
truth = {"S1_heterogeneous": "checks RIGHT AFTER each hard step (~41/91/141) + spread the rest (NOT even, NOT all-end)",
         "S2_uniform": "EVEN spacing (~every 25 steps) - no bottleneck to target"}
for k, body in scenarios.items():
    open(f"runs/cycle173_realagent_checkpoint/prompt_{k}.txt", "w").write(COMMON.format(body=body))
json.dump({"truth": truth, "discriminating": "ADAPTATION: after-hard-steps for S1 vs even for S2; + the AFTER-not-AT position subtlety (check after the hard step executes)"},
          open("runs/cycle173_realagent_checkpoint/truth.json", "w"), indent=1)
print("Built 2 scenarios (meth-24). c169-correct: S1 checks AFTER hard steps 40/90/140; S2 EVEN (control).")
print("KEY: does the agent ADAPT placement to the difficulty profile (after-hard for S1, even for S2)?")
print("DONE")
