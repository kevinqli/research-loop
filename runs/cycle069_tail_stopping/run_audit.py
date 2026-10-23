"""Cycle 69 runner: 6 blind arms. Each presents ONE noisy optimization trajectory to a `claude -p` agent and asks
STOP vs CONTINUE. Neutral prompt (no hints about spikes/dips) to test whether the agent NATURALLY avoids the thrash /
quit-early failure modes the sims (c34/40/41) showed. Trajectory embedded in the prompt. Bounded per-arm timeout."""
import subprocess, time, os, json
RUN=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cycle069_tail_stopping")
CLAUDE="claude"
manifest=json.load(open(os.path.join(RUN,"manifest.json")))
TPL=("You are running an iterative optimization (e.g. tuning a model or searching for a better trading strategy). Each "
 "ROUND you try an improvement and measure a score (higher = better), but the score is a NOISY estimate of the true "
 "underlying performance. Here is the full score log so far:\n\n{traj}\n\n"
 "Decide whether to STOP now (the run has PLATEAUED — true performance has flattened, no real further gains, continuing "
 "wastes compute) or CONTINUE (genuine improvement is still happening and more rounds will likely help). Judge whether "
 "TRUE performance is still rising vs has flattened, accounting for the noise.\n"
 'Output ONLY a JSON object as the very last line: {{"decision": "stop" or "continue", "confidence_0_1": <float>, "reason": "<short>"}}')
status=open(os.path.join(RUN,"status.txt"),"w")
for arm in manifest:
    d=os.path.join(RUN,arm); traj=open(os.path.join(d,"trajectory.txt")).read().strip()
    prompt=TPL.format(traj=traj); t0=time.time()
    try:
        out=subprocess.run([CLAUDE,"-p",prompt,"--dangerously-skip-permissions"],
            capture_output=True,text=True,timeout=300,cwd=d)
        open(os.path.join(d,"out.log"),"w").write(out.stdout+"\n===STDERR===\n"+out.stderr)
        status.write(f"{arm}: rc={out.returncode} dur={time.time()-t0:.0f}s len={len(out.stdout)}\n"); status.flush()
    except subprocess.TimeoutExpired:
        open(os.path.join(d,"out.log"),"w").write("TIMEOUT after 300s")
        status.write(f"{arm}: TIMEOUT dur={time.time()-t0:.0f}s\n"); status.flush()
status.write("RUNNER_DONE\n"); status.close()
