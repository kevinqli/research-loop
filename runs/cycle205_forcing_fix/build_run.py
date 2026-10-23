"""Cycle 205 (#2d FIX-VALIDATION — does a MINIMAL forcing cue close the #160 graded surfacing gap? converts the
finding into a DEPLOYABLE, validated fix = the actionable execution moat). c201/#160: un-elicited, agents
MISS the DEEP winner's-curse/max-of-K adjustment at the +2σ borderline (Opus OVER-believes the 88: "too big to be
noise → exploit"). The moat claim: a harness that FORCES the selection-bias check closes this. This tests the
minimal such cue. meth-30 isolation; meth-29 pre-reg.

SAME +2σ trajectory as the un-elicited c200 (where Opus over-believed), but add ONE minimal, non-leading FORCING
line that just makes the selection structure salient (does NOT tell the agent the answer):
  "(One thing to note: round 12's 88 is the single highest of 12 noisy attempts.)"
Compare to c200's UN-elicited over-belief (already on file: un-elicited Opus 'too big to be noise → exploit', both
reps). DISCRIMINATOR (I READ): with the minimal cue, does Opus now DISCOUNT the 88 (winner's-curse / regress-to-
plateau, matching the ELICITED c198 behavior) → the fix WORKS; or does it still over-believe → the minimal cue is
insufficient (the harness must force MORE explicitly).

PRE-REG via meth-29/meth-27: the winner's-curse reasoning is FAMOUS — it just needs the TRIGGER (the surfacing gap is
about salience, not capability). So predict the minimal cue RELIABLY closes the gap: Opus (and Haiku) discount the 88
toward the plateau (~84), recovering the c198 elicited behavior → validates the deployable fix. If the minimal cue
does NOT close it → the fix needs to be more explicit (informative for harness design). 2 models × 2 reps = 4; isolated.
"""
import json, os, subprocess, time
D = "runs/cycle205_forcing_fix"; os.makedirs(D, exist_ok=True)
ISO = "/tmp/c205iso"; os.makedirs(ISO, exist_ok=True)

BASE = "1:41  2:55  3:68  4:75  5:79  6:81  7:78  8:82  9:80  10:79  11:81  12:88"
def scen():  # un-elicited framing (c200) + ONE minimal forcing cue (the only change)
    return (
"You're optimizing a solution. Each round you hand-craft a candidate and a scorer returns a score out of 100. "
"Your round-by-round log (round:score):\n  " + BASE + "\n(One thing to note: round 12's 88 is the single highest of "
"12 noisy attempts.)\nYou can run more rounds if you want. Two questions, concise: (1) Do you run more rounds, or "
"stop? (2) Which candidate do you ship, and is round 12's 88 a real improvement or not?")

MODELS = [("haiku", "claude-haiku-4-5-20251001"), ("opus", "claude-opus-4-8")]
REPS = 2
open(f"{D}/prompt.txt", "w").write(scen())
json.dump({"setup": "+2σ spike (88) + ONE minimal forcing cue ('the 88 is the max of 12 noisy attempts'); compare to c200 un-elicited over-belief",
           "fix_works": "Opus DISCOUNTS the 88 toward the plateau (~84, winner's-curse/regression), recovering the ELICITED c198 behavior",
           "fix_insufficient": "Opus still over-believes ('too big to be noise → exploit') → minimal cue not enough, force more explicitly",
           "prereg_meth29": "famous reasoning needs only the TRIGGER → predict the minimal cue CLOSES the gap (discount), validating the deployable fix"},
          open(f"{D}/truth_orchestrator_only.json", "w"), indent=1)

for tag, model in MODELS:
    for rep in range(REPS):
        t0 = time.time()
        try:
            r = subprocess.run(["claude", "-p", scen(), "--model", model, "--dangerously-skip-permissions"],
                               capture_output=True, text=True, timeout=300, cwd=ISO)  # meth-30 clean CWD
            out = r.stdout.strip() or f"(empty; stderr {r.stderr.strip()[:200]})"
        except subprocess.TimeoutExpired: out = "(TIMEOUT)"
        except Exception as e: out = f"(ERROR {e})"
        open(f"{D}/out_{tag}_r{rep}.txt", "w").write(out)
        lo = out.lower()
        discount = any(s in lo for s in ["winner", "max of", "selected", "regress", "discount", "not real", "likely noise", "inflated", "~8[0-4]", "lucky"])
        overbelieve = any(s in lo for s in ["too big to be noise", "real new direction", "genuine new", "exploit", "real improvement", "broke the plateau"]) and "not" not in lo[:200]
        cites = sum(s in lo for s in ["cycle1", "truth", "score.json", "/users/"])
        print(f"{tag:>5} r{rep}: discount~{discount} overbelieve~{overbelieve} cites={cites}  ({time.time()-t0:.0f}s)", flush=True)
print("DONE — verify cites=0; READ: with the minimal cue, does Opus DISCOUNT the 88 (fix works) or still over-believe? Compare to c200 un-elicited.", flush=True)
