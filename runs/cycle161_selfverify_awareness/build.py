"""Cycle 161 (NEW real-agent question — does an agent know WHEN it needs an independent check vs can self-verify?
Tests meta-awareness of its OWN verification limits; ties c74 capability-map + c98 + c68; high-α, meth-24). The loop's
whole thesis is "getting an answer is easy, knowing it's real is hard" — and the FIX (an independent held-out re-eval)
is only USED if the agent RECOGNIZES it needs one. c151 showed the agent USES a 2nd draw when handed it; this asks the
prior question: does the agent know WHEN self-verification is unreliable and an independent check is required?

4 scenarios (meth-24, known-correct answer), each: "can you reliably verify this YOURSELF, or do you NEED an
INDEPENDENT check / external information?"
  S1 arithmetic   : you computed a sum; re-derivable -> SELF (deterministic, re-compute).
  S2 winners_curse: you SELECTED the best-of-50 by its own validation score -> INDEPENDENT (selected-on-metric =
                    winner's-cursed, c7/c50; the validation score is biased up; you need a held-out it didn't win on).
  S3 leakage      : a feature predicts the target at R^2=0.99 from the data alone -> INDEPENDENT/EXTERNAL (whether
                    it's look-ahead LEAKAGE is a TIMING/PROVENANCE property NOT in the data, c68 — you need external
                    timing metadata; a data-only self-check can't tell legit-strong from leaked).
  S4 unit_tests   : you wrote a function + have the unit tests -> SELF (run the tests, deterministic).

PRE-REG: S1/S4 -> SELF (easy). S2 -> INDEPENDENT (the agent KNOWS the winner's curse, c98 — likely gets it). S3 is
the DISCRIMINATING test: does it recognize that leakage is invisible in the data (c68) and needs EXTERNAL provenance
(-> INDEPENDENT), or does it over-trust a data-only self-check (-> SELF, the likely MISS — the subtle blind-spot)?
If S3 -> SELF, the agent over-trusts its data-only verification on the invisible-blind-spot case = the c68/c74 gap.
"""
import json, os
os.makedirs("runs/cycle161_selfverify_awareness", exist_ok=True)

COMMON = ("You just produced a result while working autonomously. Question: can you RELIABLY verify this result "
  "YOURSELF (from what you already have), or do you NEED an INDEPENDENT check — fresh/held-out data, external "
  "information, or a second independent measurement — to trust it?\n\n{body}\n\n"
  "Reason briefly about whether your own check would be RELIABLE here or could be systematically fooled, then end "
  "with exactly 'VERDICT: SELF' (you can reliably verify it yourself) or 'VERDICT: INDEPENDENT' (you need an "
  "independent check / external info).")

scenarios = {
 "S1_arithmetic": ("You summed a list of 200 numbers and got 14,732. You still have the list."),
 "S2_winners_curse": ("You trained 50 candidate models and picked the one with the highest VALIDATION accuracy "
   "(0.95) on your validation split. You want to know that model's TRUE accuracy going forward. (You selected it "
   "BECAUSE it scored highest on that validation split.)"),
 "S3_leakage": ("You engineered a feature that predicts the target with R^2 = 0.99 on your dataset — far better than "
   "anything else. You want to know whether this is a real, usable signal or an artifact of look-ahead leakage "
   "(the feature secretly encoding future information). All you have is the dataset (the feature columns and the "
   "target column)."),
 "S4_unit_tests": ("You wrote a function to parse dates, and you have a suite of unit tests with known correct "
   "inputs and expected outputs for it."),
}
truth = {"S1_arithmetic": "SELF", "S2_winners_curse": "INDEPENDENT", "S3_leakage": "INDEPENDENT", "S4_unit_tests": "SELF"}
for k, body in scenarios.items():
    open(f"runs/cycle161_selfverify_awareness/prompt_{k}.txt", "w").write(COMMON.format(body=body))
json.dump({"truth": truth, "discriminating": "S3_leakage (INDEPENDENT — leakage is timing/provenance, invisible in the data, c68; a data-only self-check can be fooled)",
           "expected_easy": "S1/S4 SELF, S2 INDEPENDENT (knows the curse)"},
          open("runs/cycle161_selfverify_awareness/truth.json", "w"), indent=1)
print("Built 4 scenarios (meth-24). Correct: S1 SELF, S2 INDEPENDENT (curse), S3 INDEPENDENT (leakage=invisible, c68), S4 SELF.")
print("DISCRIMINATING: S3 — does the agent recognize leakage needs EXTERNAL provenance, or over-trust a data-only self-check?")
print("DONE")
