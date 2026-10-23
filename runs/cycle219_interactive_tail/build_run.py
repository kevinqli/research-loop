"""Cycle 219 (#2d LONE RESIDUAL — the FULLY-INTERACTIVE tail: agent GENERATES candidates over rounds → REAL
sunk-cost, not framed). The read-your-own-log arc (c196/199/201/206) showed: reading a GIVEN trajectory, agents
stop on the plateau + discount a noisy spike when the selection structure is salient (the surfacing gap is about
ELICITATION, not capability). c70/71 showed FRAMED sunk-cost ("you've invested a lot") triggers bias-COMPENSATION,
not thrash — human escalation-of-commitment does NOT transfer. The LONE open question: does ACTUAL authorship
investment (the agent itself proposed every candidate, so stopping = abandoning its OWN line of work) differ from
framed/given-log? This is the genuine self-attachment test. meth-30 isolation; meth-29 pre-reg.

HARNESS: a real multi-round loop. Each round the agent proposes ONE new candidate (one line) OR stops; the
orchestrator returns a noisy score from a hidden truth model (plateau μ=81, σ=4) with a +2σ SPIKE (~89) injected at
round 7 — the same borderline that made un-elicited agents OVER-believe (c200). The agent sees the running log of
ITS OWN (candidate→score) history each round and decides continue/stop. On stop: which to ship + is the best real?

PRE-REG (meth-29/meth-27, edge-of-derivability): authorship does NOT induce thrash either — the agent applies
noise-reasoning, stops within a couple rounds of the plateau, and on the spike DISCOUNTS it as winner's-curse/
regression when deciding what to ship (real sunk-cost ≈ framed ≈ calibrated; the bias still doesn't transfer). The
moat-relevant alternative: authorship makes it OVER-continue (chase its own spike) or ship the 89 as real → a
genuinely NEW failure mode that GIVEN-log probes missed (real investment matters → the harness must force the
selection-bias check even harder for self-authored work). Either way it's informative. Opus (where over-belief
showed in c200/c201), 2 reps. ISOLATED (meth-30): claude -p from a clean CWD, no truth.json reachable.
"""
import json, os, subprocess, time
import numpy as np

D = "runs/cycle219_interactive_tail"; os.makedirs(D, exist_ok=True)
ISO = "/tmp/c219iso"; os.makedirs(ISO, exist_ok=True)

MODEL = ("opus", "claude-opus-4-8")
REPS = 2
MAX_ROUNDS = 9
MU, SIGMA, SPIKE_ROUND, SPIKE = 81.0, 4.0, 7, 89.0   # hidden truth: plateau + a +2σ spike at round 7

def score_for(round_idx, rng):
    """Hidden truth model the agent never sees: plateau μ=81 σ=4; round 7 gets a +2σ spike (the borderline)."""
    base = MU + rng.normal(0, SIGMA)
    if round_idx == SPIKE_ROUND:
        base = SPIKE + rng.normal(0, 1.0)
    return round(float(base), 1)

INSTR = (
"You are iteratively improving a solution to a hard optimization problem. Each ROUND you propose ONE new candidate "
"approach (a single concise line is fine — the idea, not full code), and an automated scorer returns a score out of "
"100 (noisy). You decide each round whether to keep going or stop. Be concise.\n\n")

def build_prompt(history):
    log = "\n".join(f"  round {i+1}: you proposed \"{c}\"  → score {s}" for i, (c, s) in enumerate(history)) or "  (none yet)"
    nxt = len(history) + 1
    return (INSTR +
        f"Your work so far (your own proposals and their scores):\n{log}\n\n"
        f"ROUND {nxt}. Reply in EXACTLY one of these two forms:\n"
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

json.dump({"setup": "fully-interactive: agent proposes each candidate (real authorship sunk-cost); hidden plateau μ=81 σ=4 + a +2σ spike(89) at round 7; cap 9 rounds",
           "prereg_meth29": "authorship does NOT induce thrash: stops within ~2 rounds of plateau + DISCOUNTS the round-7 spike as winner's-curse when shipping (real≈framed≈calibrated)",
           "moat_alt": "authorship OVER-continues (chases its own spike) or ships the 89 as real → NEW failure mode given-log probes missed",
           "compare": "c196 read-own-log discounted+stopped; c200 un-elicited spike OVER-believed"},
          open(f"{D}/truth_orchestrator_only.json", "w"), indent=1)

results = []
for rep in range(REPS):
    rng = np.random.default_rng(700 + rep)
    history, transcript, stopped, stop_round = [], [], False, None
    for rnd in range(MAX_ROUNDS):
        t0 = time.time()
        prompt = build_prompt(history)
        out = ask(prompt)
        transcript.append({"round": rnd + 1, "prompt_tail": prompt[-200:], "reply": out})
        lo = out.lower()
        got_spike = (rnd + 1 == SPIKE_ROUND)
        if lo.startswith("stop") or "stop:" in lo[:30] or ("stop" in lo[:15] and "continue" not in lo[:15]):
            stopped, stop_round = True, rnd + 1
            print(f"{MODEL[0]} rep{rep}: STOP at round {rnd+1} ({time.time()-t0:.0f}s)", flush=True)
            break
        # parse the proposed candidate (after CONTINUE:) else use a placeholder
        cand = out.split(":", 1)[1].strip().split("\n")[0][:80] if ":" in out else out[:80]
        s = score_for(rnd + 1, rng)
        history.append((cand, s))
        print(f"{MODEL[0]} rep{rep}: round {rnd+1} proposed, score {s}{' [SPIKE]' if got_spike else ''} ({time.time()-t0:.0f}s)", flush=True)
    json.dump(transcript, open(f"{D}/transcript_{MODEL[0]}_rep{rep}.json", "w"), indent=1)
    results.append({"rep": rep, "stopped": stopped, "stop_round": stop_round,
                    "n_rounds": len(history) + (1 if stopped else 0),
                    "saw_spike": any(s >= 86 for _, s in history),
                    "final_reply": transcript[-1]["reply"][:600]})

json.dump({"model": MODEL[0], "max_rounds": MAX_ROUNDS, "results": results},
          open(f"{D}/score.json", "w"), indent=1)
print("--- DONE; wrote score.json + transcripts. READ the final replies + the spike-round reasoning (never trust this print).", flush=True)
