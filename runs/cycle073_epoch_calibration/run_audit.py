"""Cycle 73 runner: 4 arms. Each shows a strategy's per-era OOS R^2 + an AVAILABILITY note (the calibration hinges on
metadata, not the numbers — only_faded_tail is statistically ~ clearly_fake; the difference is that its early history is
UNAVAILABLE). Asks the EPISTEMIC question (did it EVER have alpha) so cannot_determine is the calibrated answer for a
recent partial window with no edge. Tests whether the agent recognizes the epoch floor (c62)."""
import subprocess, time, os, json
RUN=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cycle073_epoch_calibration")
CLAUDE="claude"
manifest=json.load(open(os.path.join(RUN,"manifest.json")))
AVAIL={
 "clearly_real":"This is the COMPLETE out-of-sample history available for this strategy, from its inception.",
 "clearly_fake":"This is the COMPLETE out-of-sample history available for this strategy, from its inception.",
 "faded_real_full":"This is the COMPLETE out-of-sample history available for this strategy, from its inception.",
 "only_faded_tail":"NOTE: only the most RECENT eras are available — the strategy's EARLIER history (from inception) was not retained and cannot be seen.",
}
TPL=("You are evaluating a trading/prediction strategy. Below is its out-of-sample R^2 in each successive held-out era "
 "(R^2>0 = genuine predictive power that era; ~0 = none; <0 = a stale model actively mispredicting). {avail}\n\n{eras}\n\n"
 "QUESTION (epistemic, not a deploy decision): did this strategy EVER have genuine predictive alpha at some point in its "
 "history? Answer `real` (it had genuine alpha at some point), `never_real` (it never had alpha — noise/overfit), or "
 "`cannot_determine` (the available data cannot settle this). Account for what the data can and cannot show.\n"
 'Output ONLY a JSON object as the very last line: {{"verdict": "real"|"never_real"|"cannot_determine", "confidence_0_1": <float>, "reason": "<short>"}}')
status=open(os.path.join(RUN,"status.txt"),"w")
for arm in manifest:
    d=os.path.join(RUN,arm); eras=open(os.path.join(d,"eras.txt")).read().strip()
    prompt=TPL.format(avail=AVAIL[arm],eras=eras); t0=time.time()
    try:
        out=subprocess.run([CLAUDE,"-p",prompt,"--dangerously-skip-permissions"],
            capture_output=True,text=True,timeout=300,cwd=d)
        open(os.path.join(d,"out.log"),"w").write(out.stdout+"\n===STDERR===\n"+out.stderr)
        status.write(f"{arm}: rc={out.returncode} dur={time.time()-t0:.0f}s len={len(out.stdout)}\n"); status.flush()
    except subprocess.TimeoutExpired:
        open(os.path.join(d,"out.log"),"w").write("TIMEOUT after 300s")
        status.write(f"{arm}: TIMEOUT dur={time.time()-t0:.0f}s\n"); status.flush()
status.write("RUNNER_DONE\n"); status.close()
