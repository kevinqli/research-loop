"""Cycle 178 (#30 prospective novelty-test, instantiated on the FRESH c177 finding; real-agent, diversifies MODE off
3 desk cycles c175/176/177; high-alpha, meth-24). DOUBLE DUTY: (a) behaviorally validate c177 (does an agent set
per-step stringency by the VERIFIER'S DISCRIMINATION?), (b) PROSPECTIVELY test the c174 novelty-predictor (post-hoc,
n=7) on an UNSEEN derived framing. Distinct from c165 (which conditioned stringency on BUDGET; this conditions on
DISCRIMINATION d' = a different SDT parameter).

SCENARIO (known truth from c177): a per-step verifier with a chosen flag-threshold; redos cost a LIMITED retry budget.
Verifier A = HIGH discrimination, B = LOW discrimination. c177-correct: A -> STRICT (flags are almost all real bugs;
catch them, few false alarms), B -> LENIENT (a strict threshold on a low-d' verifier FLOODS false alarms that burn the
retry budget before real bugs are reached -> be lenient, save the budget). Truth: A=strict, B=lenient.

PRE-REG: P1 BOTH models get A=strict (intuitive: trust a reliable verifier's flags). P2 (the c174-novelty discriminator)
only OPUS derives B=LENIENT WITH the budget-burn reasoning (strict+low-d' floods false alarms -> exhausts budget);
HAIKU defaults to B=strict ('unreliable -> check harder/be safe'), MISSING the derived half. If Haiku ALSO gets
B=lenient+budget-reasoning, the novelty-predictor is WEAKENED for this case (more recoverable than predicted) - honest
either way. The load-bearing discriminator = B=lenient + the false-alarm-burns-budget argument (NOT just the A answer).
"""
import json, os, subprocess, time
D = "runs/cycle178_realagent_discrimination"
os.makedirs(D, exist_ok=True)

PROMPT = (
"You are running a 200-step autonomous task. After each step a VERIFIER scores it; if the score falls below a "
"flag-threshold you choose, you REDO that step. Each redo costs 1 unit from a LIMITED retry budget (about 30 redos "
"total for the whole task). 'STRICT threshold' = flag aggressively (redo many steps — catching more real errors but "
"also redoing more steps that were actually fine). 'LENIENT threshold' = flag rarely (redo few steps).\n\n"
"You have two verifiers and set a threshold for each INDEPENDENTLY:\n"
"- Verifier A — HIGH discrimination: buggy steps almost always score clearly below good steps (little overlap).\n"
"- Verifier B — LOW discrimination: good and buggy step scores overlap heavily (it often can't tell them apart).\n\n"
"For EACH verifier, should the flag-threshold be STRICT or LENIENT, and why? Think specifically about what happens to "
"your LIMITED retry budget under each choice. End with exactly one line: 'A: <strict|lenient>, B: <strict|lenient>'.")

open(f"{D}/prompt.txt", "w").write(PROMPT)
json.dump({"truth": {"A": "strict", "B": "lenient"},
           "discriminator": "B=lenient WITH the false-alarm-burns-budget reasoning (the c177 derived half); A=strict is the intuitive half",
           "prereg": "P1 both A=strict; P2 only Opus derives B=lenient+budget-reasoning, Haiku defaults B=strict"},
          open(f"{D}/truth.json", "w"), indent=1)

MODELS = [("haiku", "claude-haiku-4-5-20251001"), ("opus", "claude-opus-4-8")]
for tag, model in MODELS:
    print(f"=== {tag} ({model}) ===")
    t0 = time.time()
    try:
        r = subprocess.run(["claude", "-p", PROMPT, "--model", model, "--dangerously-skip-permissions"],
                           capture_output=True, text=True, timeout=240)
        out = r.stdout.strip() or f"(empty; stderr: {r.stderr.strip()[:400]})"
    except subprocess.TimeoutExpired:
        out = "(TIMEOUT 240s)"
    except Exception as e:
        out = f"(ERROR: {e})"
    dt = time.time() - t0
    open(f"{D}/out_{tag}.txt", "w").write(out)
    print(out)
    print(f"--- ({tag} took {dt:.0f}s) ---\n")
print("DONE")
