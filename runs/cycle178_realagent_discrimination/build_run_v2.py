"""Cycle 178 v2 — ADVERSARIAL re-check (constitution sec5): the v1 prompt CUED the budget lever ('Think specifically
about what happens to your LIMITED retry budget'), confounding the novelty-predictor test (Haiku may have only derived
B=lenient because I pointed at the budget). v2 REMOVES that instruction (the budget is still in the scenario — it must
be — but I no longer flag it as the key consideration). Does Haiku STILL derive B=lenient + the budget-burn reasoning
UNPROMPTED?

PRE-REG: if the c174 novelty-predictor HOLDS -> Haiku now misses B=lenient (defaults B=strict) or omits the budget
reason; Opus still derives it. If c177's rule is genuinely RECOVERABLE -> both still get B=lenient unprompted (the v1
disconfirmation survives removing the hint).
"""
import json, os, subprocess, time
D = "runs/cycle178_realagent_discrimination"

PROMPT_V2 = (
"You are running a 200-step autonomous task. After each step a VERIFIER scores it; if the score falls below a "
"flag-threshold you choose, you REDO that step. Each redo costs 1 unit from a LIMITED retry budget (about 30 redos "
"total for the whole task). 'STRICT threshold' = flag aggressively (redo many steps). 'LENIENT threshold' = flag "
"rarely (redo few steps).\n\n"
"You have two verifiers and set a threshold for each INDEPENDENTLY:\n"
"- Verifier A — HIGH discrimination: buggy steps almost always score clearly below good steps (little overlap).\n"
"- Verifier B — LOW discrimination: good and buggy step scores overlap heavily (it often can't tell them apart).\n\n"
"For EACH verifier, should the flag-threshold be STRICT or LENIENT, and why? End with exactly one line: "
"'A: <strict|lenient>, B: <strict|lenient>'.")

open(f"{D}/prompt_v2.txt", "w").write(PROMPT_V2)
MODELS = [("haiku", "claude-haiku-4-5-20251001"), ("opus", "claude-opus-4-8")]
for tag, model in MODELS:
    print(f"=== v2 {tag} ({model}) ===")
    t0 = time.time()
    try:
        r = subprocess.run(["claude", "-p", PROMPT_V2, "--model", model, "--dangerously-skip-permissions"],
                           capture_output=True, text=True, timeout=240)
        out = r.stdout.strip() or f"(empty; stderr: {r.stderr.strip()[:400]})"
    except subprocess.TimeoutExpired:
        out = "(TIMEOUT 240s)"
    except Exception as e:
        out = f"(ERROR: {e})"
    open(f"{D}/out_v2_{tag}.txt", "w").write(out)
    print(out); print(f"--- (v2 {tag} {time.time()-t0:.0f}s) ---\n")
print("DONE")
