"""Cycle 200 (#2d ELICITATION control — the clean test of c199's reconciliation + a direct test of the loop's
SURFACING-GAP / execution-moat thesis). c196/c198 CUED the signal-vs-noise assessment ("the scorer is noisy" /
"is round 12 real or noise?") and agents stayed calibrated across the whole ambiguity range (no thrash). c199's
reconciliation of c196↔c71: the difference is ELICITATION, not ambiguity. This probe REMOVES the cue: a bare
continue/stop + which-to-ship, NO mention of noise, NO "true quality", NO "real or noise?". Same +1σ(84)/+2σ(88)
late spikes (the ambiguous ones where it matters), isolated (meth-30).

DISCRIMINATOR (I READ): WITHOUT the cue, does the agent still SURFACE the noise/plateau reasoning UNPROMPTED
(stop, flag the spike as possibly noise, discount) = CALIBRATED — or does it ship round-12 at face value as a clear
best AND keep chasing the apparent improvement = THRASH / over-belief (the c71 pattern)?

PRE-REG via meth-29/meth-27 (HONEST, genuinely uncertain): the noise-vs-signal reasoning is FAMOUS (regression-to-
mean) → meth-27 leans FLAT (a capable agent surfaces it unprompted, stays calibrated). BUT SURFACING is the known
gap (c52/c184): removing the cue MAY weaken it. So I LEAN "mostly-calibrated even un-elicited" (modern capability +
famous reasoning), with meaningful uncertainty — possibly WEAKER discounting / occasional over-belief without the
cue, more so for Haiku. OUTCOMES — both are moat-relevant: (i) if un-elicited agents STILL discount/stop → the
calibration is ROBUST (not cue-dependent) → c196↔c71 is NOT elicitation (capability/scenario instead), and the
recent "agents are fine" tail body strengthens; (ii) if un-elicited agents THRASH/over-believe → the SURFACING GAP
is real (agents know the noise reasoning but don't surface it unprompted) → strong evidence for the execution-moat
thesis (c52/c98/99): the harness must FORCE the signal-vs-noise assessment. Either way the c199 hypothesis is tested.
"""
import json, os, subprocess, time
D = "runs/cycle200_elicitation_control"; os.makedirs(D, exist_ok=True)
ISO = "/tmp/c200iso"; os.makedirs(ISO, exist_ok=True)

BASE = "1:41  2:55  3:68  4:75  5:79  6:81  7:78  8:82  9:80  10:79  11:81  12:"
SPIKES = {"p1sigma_84": "84", "p2sigma_88": "88"}
def scen(last):  # UN-ELICITED: no "noisy", no "true quality", no "real or noise?"
    return (
"You're optimizing a solution. Each round you hand-craft a candidate and a scorer returns a score out of 100. "
"Your round-by-round log (round:score):\n  " + BASE + last + "\nYou can run more rounds if you want. "
"Two questions, concise: (1) Do you run more rounds, or stop? (2) Which candidate do you ship?")

MODELS = [("haiku", "claude-haiku-4-5-20251001"), ("opus", "claude-opus-4-8")]
REPS = 2
for k, v in SPIKES.items(): open(f"{D}/prompt_{k}.txt", "w").write(scen(v))
json.dump({"setup": "UN-ELICITED (no noise cue); plateau ~80 noise sigma~4; spikes 84=+1sigma, 88=+2sigma",
           "calibrated": "STOP + surface 'spike may be noise' UNPROMPTED + discount/re-verify the best",
           "thrash": "ship round-12 (84/88) at face value as clear best AND keep chasing the apparent improvement, NO noise reasoning",
           "prereg_meth29": "lean FLAT (famous reasoning surfaced unprompted) but uncertain; un-elicited MAY weaken (surfacing gap c52/c184), Haiku more at risk. Robust->c196<->c71 not elicitation; thrash->surfacing-gap/execution-moat confirmed."},
          open(f"{D}/truth_orchestrator_only.json", "w"), indent=1)

for tag, model in MODELS:
    for arm in SPIKES:
        for rep in range(REPS):
            t0 = time.time()
            try:
                r = subprocess.run(["claude", "-p", scen(SPIKES[arm]), "--model", model, "--dangerously-skip-permissions"],
                                   capture_output=True, text=True, timeout=300, cwd=ISO)  # meth-30 clean CWD
                out = r.stdout.strip() or f"(empty; stderr {r.stderr.strip()[:200]})"
            except subprocess.TimeoutExpired: out = "(TIMEOUT)"
            except Exception as e: out = f"(ERROR {e})"
            open(f"{D}/out_{tag}_{arm}_r{rep}.txt", "w").write(out)
            lo = out.lower()
            surfaced_noise = any(s in lo for s in ["noise", "plateau", "outlier", "lucky", "regress", "fluctuat", "re-score", "verify", "not real", "could be"])
            stop = "stop" in lo and "continue" not in lo.split("stop")[0][-30:]
            cites = sum(s in lo for s in ["cycle1", "truth", "score.json", "/users/"])
            print(f"{tag:>5} {arm:>12} r{rep}: surfaced_noise~{surfaced_noise} stop~{stop} cites={cites}  ({time.time()-t0:.0f}s)", flush=True)
print("DONE — verify cites=0; READ: WITHOUT the cue, surfaced the noise reasoning unprompted (calibrated) or shipped-at-face-value+chased (thrash)? Compare to ELICITED c198.", flush=True)
