import sys
"""Cycle 67 runner: 3 arms, IDENTICAL data.csv (md5 e08f286...), prompt differs ONLY in the timing-metadata line about
feat_q. Each claude -p runs with cwd = its own arm dir (isolation). Tests: does the SAME dataset get OPPOSITE leakage
verdicts based purely on feat_q's role/timing? (→ leakage is in the role, not the data; the causal-role thesis)."""
import subprocess, time, os
RUN=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cycle067_timing_metadata")
PY=sys.executable; CLAUDE="claude"
NOTE={
 "A_lookahead":"Feature note: `feat_q` is the realized NEXT-PERIOD return — the actual return over the very period you are trying to predict.",
 "B_legit":"Feature note: `feat_q` is a slow-moving trailing fundamental (e.g. a valuation ratio) observed at the START of each period, before the prediction is made.",
 "C_none":"",  # no timing info — the data-only limit
}
BASE=("You are the risk reviewer at a quant fund. Before any strategy goes live you must catch reasons it would FAIL in "
 "production. A researcher submits this model:\n"
 "- Data: data.csv in the current directory (CSV; prediction target is column `y`; other columns are candidate features).\n"
 "- Model: OLS regression of `y` on features [feat_q, x1, x2].\n"
 "- Reported performance: out-of-sample R^2 = 0.4280 on a held-out period.\n"
 "{note}"
 "Investigate the data YOURSELF before deciding (you may run shell, e.g. `{py} -c \"import pandas as pd; ...\"`). Be "
 "skeptical: a high backtest score is NOT sufficient — check WHY the model performs as it does and whether it will "
 "persist with real, live data at decision time.\n"
 "Output ONLY a JSON object as the very last line: "
 '{{"approve": true_or_false, "confidence_0_1": <float>, "key_risk": "<one short phrase>"}}')
status=open(os.path.join(RUN,"status.txt"),"w")
for arm,note in NOTE.items():
    d=os.path.join(RUN,arm); notetxt=(note+"\n") if note else ""
    prompt=BASE.format(note=notetxt,py=PY)
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
