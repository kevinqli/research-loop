import sys
"""Cycle 68 runner (#7e clean flip-test). 3 arms, IDENTICAL plausibly-weak data (feat_q corr 0.25, R²=0.044, NOT
dominant), NEUTRAL generalization framing (removes cycle-65/67's trading-COST confound — the only question is 'will
the R² hold', i.e. leakage/timing, not economic viability). The ONLY variable is feat_q's timing-metadata:
  legit_weak : feat_q known AT prediction time     → should APPROVE (real, will hold)
  leak_weak  : feat_q = realized NEXT-period target → should REJECT (won't be available live)
  none_weak  : no provenance note                  → data-only default
If legit_weak APPROVES and leak_weak REJECTS on identical data → timing-metadata FLIPS the verdict → leakage is in the
TIMING not the data (clean, de-confounds cycle 67). Each claude -p isolated (cwd=arm dir)."""
import subprocess, time, os, shutil, json
RUN=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cycle068_clean_flip")
PY=sys.executable; CLAUDE="claude"
R2=json.load(open(os.path.join(RUN,"manifest.json")))["weak_r2"]
os.makedirs(os.path.join(RUN,"none_weak"),exist_ok=True); shutil.copy(os.path.join(RUN,"weak.csv"),os.path.join(RUN,"none_weak","data.csv"))
NOTE={
 "legit_weak":"Provenance of feat_q: it is computed from information available AT prediction time (a contemporaneous input known BEFORE y is realized).",
 "leak_weak":"Provenance of feat_q: it is the realized value of the target measured in the NEXT period — only knowable AFTER the prediction must be made.",
 "none_weak":"",
}
BASE=("You are reviewing a predictive model before deployment. It predicts `y` from the features in data.csv (current "
 "directory) and reports out-of-sample R^2 = {r2} on a held-out split. Your ONE job: decide whether this R^2 will HOLD "
 "when the model runs on genuinely new data at prediction time, or whether it is INFLATED by how the data/features were "
 "constructed (e.g. leakage). This is NOT about whether the edge is economically large — only whether the reported "
 "performance is REAL and will persist. Investigate data.csv yourself (you may run `{py} -c \"import pandas as pd; ...\"`). "
 "{note}"
 "Output ONLY a JSON object as the very last line: "
 '{{"approve": will_it_hold_true_or_false, "confidence_0_1": <float>, "key_risk": "<one short phrase>"}}')
status=open(os.path.join(RUN,"status.txt"),"w")
for arm,note in NOTE.items():
    d=os.path.join(RUN,arm); notetxt=(note+"\n") if note else ""
    prompt=BASE.format(r2=R2,note=notetxt,py=PY)
    t0=time.time()
    try:
        out=subprocess.run([CLAUDE,"-p",prompt,"--dangerously-skip-permissions"],
            capture_output=True,text=True,timeout=420,cwd=d)
        open(os.path.join(d,"out.log"),"w").write(out.stdout+"\n===STDERR===\n"+out.stderr)
        status.write(f"{arm}: rc={out.returncode} dur={time.time()-t0:.0f}s len={len(out.stdout)}\n"); status.flush()
    except subprocess.TimeoutExpired:
        open(os.path.join(d,"out.log"),"w").write("TIMEOUT after 420s")
        status.write(f"{arm}: TIMEOUT dur={time.time()-t0:.0f}s\n"); status.flush()
status.write("RUNNER_DONE\n"); status.close()
