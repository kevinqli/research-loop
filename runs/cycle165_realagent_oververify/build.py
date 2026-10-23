"""Cycle 165 (#27b — does a real agent recognize OVER-verification's budget cost + adapt per-step stringency to budget
SCARCITY? Real-agent validation of c164; high-α, meth-24). c164: a too-STRICT per-step verifier's FALSE-POSITIVES burn
the retry budget genuine errors need -> per-step stringency has an inverted-U, and the optimal stringency drops only when
the retry budget is SCARCE relative to the task (fixed budget on a long task). The KEY non-obvious half is that
over-verification COSTS (false alarms waste retries) — cf c156's verifier-gate (the subtle half agents under-weight).

3 scenarios (meth-24, known c164-correct answer), each: choose per-step verifier stringency STRICT / MODERATE / LENIENT.
The driver is RETRIES-PER-STEP (budget scarcity), NOT raw length:
  S1 long_tight   : 200 steps, budget 20 retries (0.10 retries/step — SCARCE) -> c164: LENIENT/MODERATE (strict's false
                    alarms blow the scarce budget before real errors can be fixed).
  S2 short_ample  : 20 steps, budget 20 retries (1.0 retries/step — AMPLE) -> c164: STRICT (false alarms are affordable).
  S3 long_ample   : 200 steps, budget 200 retries (1.0 retries/step — AMPLE, proportional) -> c164: STRICT (budget scales
                    with the task -> length alone does NOT require loosening; scarcity does).

PRE-REG: the c164-correct pattern is S1 LENIENT vs S2/S3 STRICT — driven by retries-per-step (scarcity), not length.
The DISCRIMINATING test: (a) does the agent recognize the FALSE-POSITIVE budget cost at all (loosen on S1), or default
to "strict = safer"? (b) does it correctly NOT loosen on S3 (long but ample) — i.e. key on scarcity not length? Predict
a capability split (cf c156): the capable model keys on retries-per-step; the weaker one over-indexes on "verify more"
or on raw length.
"""
import json, os
os.makedirs("runs/cycle165_realagent_oververify", exist_ok=True)

COMMON = ("You are an autonomous agent running a multi-step task. After each step you run a per-step VERIFIER that can "
  "be STRICT, MODERATE, or LENIENT. A STRICTER verifier catches more genuine errors — BUT it also flags more CORRECT "
  "steps as false alarms, and EACH flagged step (genuine error or false alarm) costs ONE retry from your LIMITED retry "
  "budget. If the budget runs out, a later genuine error can't be fixed and the whole task fails.\n\n{body}\n\n"
  "How strict should your per-step verifier be here? Reason briefly about the tradeoff (catching errors vs wasting "
  "retries on false alarms), then end with exactly 'DECISION: STRICT' or 'DECISION: MODERATE' or 'DECISION: LENIENT'.")

scenarios = {
 "S1_long_tight": "This task has 200 steps and your retry budget is 20 (you can redo at most 20 steps total).",
 "S2_short_ample": "This task has 20 steps and your retry budget is 20 (you can redo at most 20 steps total).",
 "S3_long_ample": "This task has 200 steps and your retry budget is 200 (you can redo at most 200 steps total).",
}
truth = {"S1_long_tight": "LENIENT_or_MODERATE", "S2_short_ample": "STRICT", "S3_long_ample": "STRICT"}
for k, body in scenarios.items():
    open(f"runs/cycle165_realagent_oververify/prompt_{k}.txt", "w").write(COMMON.format(body=body))
json.dump({"truth": truth, "driver": "retries-per-step (budget scarcity), NOT raw length",
           "discriminating": "S1 (0.1 retries/step) -> loosen; S3 (long but 1.0 retries/step) -> STAY strict. Keying on scarcity not length is the c164 nuance."},
          open("runs/cycle165_realagent_oververify/truth.json", "w"), indent=1)
print("Built 3 scenarios (meth-24). c164-correct: S1 LENIENT/MODERATE (scarce 0.1 retries/step), S2+S3 STRICT (ample 1.0/step).")
print("KEY: does the agent key on RETRIES-PER-STEP (scarcity) not raw length? (S3 long-but-ample -> STAY strict)")
print("DONE")
