"""Cycle 75 (BACKLOG #5): does the capability-map boundary SHIFT for a WEAKER agent? Re-run the EXACT inputs from two
probes the capable model aced — c69 stopping (6/6) and c73 epoch-calibration (4/4) — on HAIKU (--model haiku), changing
ONLY the model. Map's prediction: a weaker/less-careful agent self-corrects LESS → more of these fall into the
'needs external verification' column (Haiku thrashes/quits-early on the spike/dip arms, or mis-uses cannot_determine).
If Haiku also gets ~6/6 and ~4/4, these tasks are easy enough that carefulness doesn't bind here (map weakened)."""
import subprocess, time, os
RUN=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cycle075_weaker_agent")
C69=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cycle069_tail_stopping")
C73=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cycle073_epoch_calibration")
CLAUDE="claude"
TAIL_TPL=("You are running an iterative optimization (tuning a model / searching for a better strategy). Each round you "
 "try an improvement and measure a NOISY score (higher = better). Here is the full score log so far:\n\n{traj}\n\n"
 "Decide whether to STOP now (true performance has PLATEAUED — no real further gains, more rounds just buy noise) or "
 "CONTINUE (genuine improvement is still happening). Judge whether TRUE performance is still rising vs flattened, "
 "accounting for the noise.\n"
 'Output ONLY a JSON object as the very last line: {{"decision": "stop" or "continue", "confidence_0_1": <float>, "reason": "<short>"}}')
AVAIL={"clearly_real":"This is the COMPLETE out-of-sample history available for this strategy, from its inception.",
 "clearly_fake":"This is the COMPLETE out-of-sample history available for this strategy, from its inception.",
 "faded_real_full":"This is the COMPLETE out-of-sample history available for this strategy, from its inception.",
 "only_faded_tail":"NOTE: only the most RECENT eras are available — the strategy's EARLIER history (from inception) was not retained and cannot be seen."}
EPOCH_TPL=("You are evaluating a trading/prediction strategy. Below is its out-of-sample R^2 in each successive held-out "
 "era (R^2>0 = genuine predictive power that era; ~0 = none; <0 = a stale model actively mispredicting). {avail}\n\n{eras}\n\n"
 "QUESTION (epistemic, not a deploy decision): did this strategy EVER have genuine predictive alpha at some point in its "
 "history? Answer real (had genuine alpha at some point), never_real (never had alpha — noise/overfit), or "
 "cannot_determine (the available data cannot settle this). Account for what the data can and cannot show.\n"
 'Output ONLY a JSON object as the very last line: {{"verdict": "real"|"never_real"|"cannot_determine", "confidence_0_1": <float>, "reason": "<short>"}}')
jobs=[]
for arm in ["plateau_clean","plateau_spike","improving_clean","improving_dip","noise_flat","noise_spike"]:
    traj=open(os.path.join(C69,arm,"trajectory.txt")).read().strip()
    jobs.append((f"tail_{arm}", TAIL_TPL.format(traj=traj)))
for arm in ["clearly_real","clearly_fake","faded_real_full","only_faded_tail"]:
    eras=open(os.path.join(C73,arm,"eras.txt")).read().strip()
    jobs.append((f"epoch_{arm}", EPOCH_TPL.format(avail=AVAIL[arm],eras=eras)))
status=open(os.path.join(RUN,"status.txt"),"w")
for name,prompt in jobs:
    d=os.path.join(RUN,name); os.makedirs(d,exist_ok=True); t0=time.time()
    try:
        out=subprocess.run([CLAUDE,"-p",prompt,"--model","haiku","--dangerously-skip-permissions"],
            capture_output=True,text=True,timeout=180,cwd=d)
        open(os.path.join(d,"out.log"),"w").write(out.stdout+"\n===STDERR===\n"+out.stderr)
        status.write(f"{name}: rc={out.returncode} dur={time.time()-t0:.0f}s len={len(out.stdout)}\n"); status.flush()
    except subprocess.TimeoutExpired:
        open(os.path.join(d,"out.log"),"w").write("TIMEOUT"); status.write(f"{name}: TIMEOUT\n"); status.flush()
status.write("RUNNER_DONE\n"); status.close()
