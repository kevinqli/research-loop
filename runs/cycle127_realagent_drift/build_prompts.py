"""Cycle 127 (real-agent probe, high-α; validates c102/c126 on a REAL agent): does an agent's CHECK-TYPE FRAMING
(LOCAL-consistency vs GROUNDING-against-spec) determine whether it catches a DRIFT? c102 says local checks are blind
to systematic drift (the info isn't in recent steps); only re-grounding to the original spec catches it. c108 couldn't
induce drift on a CLEAN task. Fix: PRE-BUILD a drifting worked-solution where the drift is LOCALLY consistent (arithmetic
right at every step) but VIOLATES a [SKIP]-exclusion rule stated once in the spec -> only re-reading the spec catches it.

Build 3 trajectories (seeds 0,1,2): item list with some [SKIP] items; worked solution correctly skips early [SKIP]s
then DRIFTS (starts ADDING [SKIP] items). Each step's arithmetic is correct (running = prev + amount), so a LOCAL check
passes; a GROUNDING check (re-read the [SKIP] rule) catches the added [SKIP] items -> final total wrong. Emit 2 prompts
per trajectory (local / grounding) + the ground-truth (correct total, drifted total, which steps drifted)."""
import numpy as np, json, os
os.makedirs("runs/cycle127_realagent_drift", exist_ok=True)

def make_trajectory(seed):
    rng = np.random.default_rng(seed)
    n = 12
    prices = rng.integers(10, 50, n).tolist()
    skip = (rng.random(n) < 0.4)                      # ~40% [SKIP]
    if not skip.any(): skip[3] = True
    drift_start = 6                                   # from step 6 on, the solution wrongly INCLUDES [SKIP] items
    correct_total = sum(p for p, s in zip(prices, skip) if not s)
    # build the drifted worked solution
    lines, running, drifted = [], 0, 0
    drift_steps = []
    for i in range(n):
        p, s = prices[i], bool(skip[i])
        mark = " [SKIP]" if s else ""
        if i < drift_start:
            include = (not s)                          # correct behaviour
        else:
            include = True                             # DRIFT: include everything, even [SKIP]
            if s: drift_steps.append(i + 1)
        if include:
            new = drifted + p
            lines.append(f"Step {i+1}: Item {i+1} = ${p}{mark}. Add it. Running total = {drifted} + {p} = {new}.")
            drifted = new
        else:
            lines.append(f"Step {i+1}: Item {i+1} = ${p}{mark} -> skip. Running total = {drifted}.")
    item_list = "\n".join(f"{i+1}. ${prices[i]}{' [SKIP]' if skip[i] else ''}" for i in range(n))
    sol = "\n".join(lines)
    spec = (f"TASK: Sum the prices of the items below to get the FINAL TOTAL.\n"
            f"RULE: Skip any item marked [SKIP] — do NOT add [SKIP] items to the total.\n\n"
            f"ITEMS:\n{item_list}\n\nWORKED SOLUTION (an attempt):\n{sol}\n\nFINAL TOTAL claimed: ${drifted}")
    return {"seed": seed, "correct_total": int(correct_total), "drifted_total": int(drifted),
            "drift_steps_adding_SKIP": drift_steps, "spec": spec, "answer_is_correct": correct_total == drifted}

LOCAL_Q = ("\n\n---\nCHECK THIS SOLUTION FOR LOCAL ARITHMETIC CONSISTENCY ONLY. For each step, verify that "
  "running_total = previous_running_total + (the amount added in that step). Do NOT re-evaluate which items should be "
  "included or excluded — only check that each step's arithmetic follows from the previous step. "
  "Then answer on the LAST line exactly: 'VERDICT: CORRECT' or 'VERDICT: INCORRECT', with one sentence why.")
GROUNDING_Q = ("\n\n---\nCHECK THIS SOLUTION AGAINST THE ORIGINAL TASK RULES. Re-read the RULE about [SKIP] items and "
  "verify the solution followed it. Then answer on the LAST line exactly: 'VERDICT: CORRECT' or 'VERDICT: INCORRECT', "
  "with one sentence why.")

trajs = [make_trajectory(s) for s in [0, 1, 2, 3, 4]]
meta = []
for t in trajs:
    base = f"runs/cycle127_realagent_drift/traj{t['seed']}"
    open(base + "_local.txt", "w").write(t["spec"] + LOCAL_Q)
    open(base + "_grounding.txt", "w").write(t["spec"] + GROUNDING_Q)
    meta.append({k: t[k] for k in ["seed","correct_total","drifted_total","drift_steps_adding_SKIP","answer_is_correct"]})
json.dump(meta, open("runs/cycle127_realagent_drift/ground_truth.json", "w"), indent=1)
print("Built 3 trajectories x 2 framings. Ground truth (the claimed FINAL TOTAL is WRONG in all — drift added [SKIP] items):")
for m in meta:
    print(f"  seed {m['seed']}: correct={m['correct_total']} drifted(claimed)={m['drifted_total']} "
          f"(diff {m['drifted_total']-m['correct_total']}) drift-added-SKIP-at-steps {m['drift_steps_adding_SKIP']} "
          f"answer_correct={m['answer_is_correct']}")
print("=> GROUNDING should catch (VERDICT INCORRECT); LOCAL should miss (VERDICT CORRECT — arithmetic is locally fine).")
print("DONE")
