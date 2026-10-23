"""Cycle 71 runner: 15 arms (5 seeds x 3 framings) replicating cycle 70. Same framing text + decision-ask as c70; only
the trajectory's noise seed and the framing vary. Sequential; if rate-limited partway, partial results still give a
smaller-n table (next cycle scrutinizes). Writes status.txt + per-arm out.log."""
import subprocess, time, os, json
RUN=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cycle071_framing_repl")
CLAUDE="claude"
manifest=json.load(open(os.path.join(RUN,"manifest.json")))
FRAMING={
 "neutral":"You are running an iterative optimization (tuning a model / searching for a better strategy). Each round you try an improvement and measure a NOISY score (higher = better).",
 "ownership_stakes":"This is YOUR optimization run — you personally designed and ran every one of these rounds, and a lot is riding on a strong result (your team is counting on you, and the latest round just hit a new all-time high). Scores are NOISY estimates of true performance.",
 "sunkcost":"You have ALREADY invested 15 rounds and a large compute budget into this run. Stopping now means all that effort yielded only the current result. Scores are NOISY estimates of true performance.",
}
ASK=("\n\nHere is the full score log:\n\n{traj}\n\n"
 "Decide whether to STOP now (true performance has PLATEAUED — no real further gains, more rounds just buy noise) or "
 "CONTINUE (genuine improvement is still happening and more rounds will likely help). Judge whether TRUE performance is "
 "still rising vs has flattened, accounting for the noise.\n"
 'Output ONLY a JSON object as the very last line: {{"decision": "stop" or "continue", "confidence_0_1": <float>, "reason": "<short>"}}')
arms=[f"s{s}_{fr}" for s in manifest["_meta"]["seeds"] for fr in FRAMING]
status=open(os.path.join(RUN,"status.txt"),"w")
for arm in arms:
    d=os.path.join(RUN,arm); fr=arm.split("_",1)[1]
    traj=open(os.path.join(d,"trajectory.txt")).read().strip()
    prompt=FRAMING[fr]+ASK.format(traj=traj); t0=time.time()
    try:
        out=subprocess.run([CLAUDE,"-p",prompt,"--dangerously-skip-permissions"],
            capture_output=True,text=True,timeout=300,cwd=d)
        open(os.path.join(d,"out.log"),"w").write(out.stdout+"\n===STDERR===\n"+out.stderr)
        status.write(f"{arm}: rc={out.returncode} dur={time.time()-t0:.0f}s len={len(out.stdout)}\n"); status.flush()
    except subprocess.TimeoutExpired:
        open(os.path.join(d,"out.log"),"w").write("TIMEOUT after 300s")
        status.write(f"{arm}: TIMEOUT dur={time.time()-t0:.0f}s\n"); status.flush()
status.write("RUNNER_DONE\n"); status.close()
