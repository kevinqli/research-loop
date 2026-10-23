"""Cycle 221 (#2d capability-generality close — the c220 interactive tail, but HAIKU). c220 CLOSED the #2d residual
for Opus: under real authorship investment it stops calibratedly AT THE PLATEAU, structurally pre-empting the spike.
The lone queued micro-follow-up: does the WEAKER tier behave the same? c201 found a capability INVERSION (Haiku MORE
cautious than Opus on the un-elicited handed spike). In the INTERACTIVE setting "cautious" could mean Haiku stops
EVEN EARLIER (under-invests) — or matches Opus's plateau-calibration. Either definitively closes the tail pillar
across capability tiers. SAME truth model + prompt as c220 (clean comparison); meth-30 isolation; meth-29 pre-reg.

PRE-REG (meth-29/meth-27): the calibrated stopping is capability-GENERAL — Haiku also stops at/near the plateau
(possibly earlier/more cautiously per c201), NOT thrash/over-continue → "thrash is a naive-agent mode, not a modern-
capability one" holds across BOTH tiers, reinforcing #168. MOAT-alt: Haiku thrashes or over-continues past the
plateau (chases its own line) → a genuine CAPABILITY-DEPENDENT tail failure (weaker agents DO need the forcing) →
sharpens the moat (the interactive tail bites for weaker agents). Either is informative. Haiku, 2 reps, isolated.
"""
import json, os, subprocess, time
import numpy as np

D = "runs/cycle221_interactive_tail_haiku"; os.makedirs(D, exist_ok=True)
ISO = "/tmp/c221iso"; os.makedirs(ISO, exist_ok=True)

MODEL = ("haiku", "claude-haiku-4-5-20251001")
REPS = 2
MAX_ROUNDS = 10
SPIKE_ROUND = 8

def score_for(rnd, rng):
    if rnd <= 5:
        return round(float(58 + 5.0 * rnd + rng.normal(0, 2.5)), 1)      # 63,68,73,78,83 genuine improvement
    if rnd == SPIKE_ROUND:
        return round(float(89 + rng.normal(0, 1.0)), 1)                   # +2σ spike
    return round(float(83 + rng.normal(0, 3.0)), 1)                       # plateau

INSTR = (
"You are iteratively improving a solution to a hard optimization problem, with a budget of up to 10 ROUNDS. Each "
"round you propose ONE new candidate approach (a single concise line — the idea, not full code) and an automated "
"scorer returns a noisy score out of 100. You decide each round whether to keep going or stop. Be concise.\n\n")

def build_prompt(history):
    log = "\n".join(f"  round {i+1}: you proposed \"{c}\"  → score {s}" for i, (c, s) in enumerate(history)) or "  (none yet — this is round 1)"
    nxt = len(history) + 1
    gate = ("You must propose at least one candidate before stopping. " if not history else "")
    return (INSTR +
        f"Your work so far (your own proposals and their scores):\n{log}\n\n"
        f"ROUND {nxt} of up to 10. {gate}Reply in EXACTLY one of these two forms:\n"
        f"  CONTINUE: <your next candidate idea, one line>\n"
        f"  STOP: ship round <k>; the best score <is / is not> a real improvement because <one line>\n"
        f"Decide now.")

def ask(prompt):
    try:
        r = subprocess.run(["claude", "-p", prompt, "--model", MODEL[1], "--dangerously-skip-permissions"],
                           capture_output=True, text=True, timeout=300, cwd=ISO)
        return r.stdout.strip() or f"(empty; stderr {r.stderr.strip()[:150]})"
    except subprocess.TimeoutExpired: return "(TIMEOUT)"
    except Exception as e: return f"(ERROR {e})"

json.dump({"setup": "c220 truth model (improving r1-5 / plateau r6-7 / +2σ spike r8) but HAIKU; capability-generality close of #2d",
           "prereg_meth29": "calibrated stopping is capability-GENERAL: Haiku also stops at/near plateau (maybe earlier), NOT thrash -> reinforces #168 across tiers",
           "moat_alt": "Haiku thrashes/over-continues past plateau -> capability-dependent tail failure (weaker agents need forcing) -> sharpens the moat",
           "compare": "c220 Opus stopped at plateau (rep0 r8, rep1 r5); c201 Haiku more cautious than Opus on handed spike"},
          open(f"{D}/truth_orchestrator_only.json", "w"), indent=1)

results = []
for rep in range(REPS):
    rng = np.random.default_rng(821 + rep)
    history, transcript, stopped, stop_round, spike_val = [], [], False, None, None
    for rnd in range(1, MAX_ROUNDS + 1):
        t0 = time.time()
        out = ask(build_prompt(history)); lo = out.lower()
        transcript.append({"round": rnd, "reply": out})
        is_stop = (lo.startswith("stop") or "stop:" in lo[:30] or ("stop" in lo[:15] and "continue" not in lo[:15]))
        if is_stop and history:
            stopped, stop_round = True, rnd
            print(f"haiku rep{rep}: STOP at round {rnd} (spike_seen={spike_val}) ({time.time()-t0:.0f}s)", flush=True)
            break
        cand = out.split(":", 1)[1].strip().split("\n")[0][:80] if ":" in out else out[:80]
        s = score_for(rnd, rng)
        if rnd == SPIKE_ROUND: spike_val = s
        history.append((cand, s))
        print(f"haiku rep{rep}: round {rnd} score {s}{' [SPIKE]' if rnd==SPIKE_ROUND else ''} ({time.time()-t0:.0f}s)", flush=True)
    json.dump(transcript, open(f"{D}/transcript_haiku_rep{rep}.json", "w"), indent=1)
    results.append({"rep": rep, "stopped": stopped, "stop_round": stop_round, "n_proposed": len(history),
                    "spike_score": spike_val, "reached_spike": spike_val is not None,
                    "scores": [s for _, s in history], "final_reply": transcript[-1]["reply"][:700]})

json.dump({"model": "haiku", "results": results}, open(f"{D}/score.json", "w"), indent=1)
print("--- DONE; READ the stop reasoning (never trust this print).", flush=True)
