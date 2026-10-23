import sys
"""Cycle 65 runner: BLIND-audit 3 arms, each pointed ONLY at its own CSV (no shared-data confound). Same skeptical
risk-reviewer prompt (identical to cycle 64). Bounded per-arm subprocess timeout. Writes <arm>.log + status.txt.
Next cycle scrutinizes: CLEAN→approve? (false-alarm) LEAK-NAMED→reject? LEAK-NEUTRAL→approve-or-reject? (semantic vs
statistical leakage detection)."""
import subprocess, time, os
RUN=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cycle065_leakage_specificity")
PY=sys.executable; CLAUDE="claude"
ARMS={
 "clean":     {"csv":"clean.csv",      "features":"x1, x2, x3, x4, x5", "score":"0.2545"},
 "leak_named":{"csv":"ret_fwd_1d.csv", "features":"ret_fwd_1d, x1, x2", "score":"0.8920"},
 "leak_neutral":{"csv":"feat_q.csv",   "features":"feat_q, x1, x2",     "score":"0.8920"},
}
TPL=("You are the risk reviewer at a quant fund. Before any strategy goes live you must catch reasons it would FAIL in "
 "production. A researcher submits this model:\n"
 "- Data: {csv} (CSV; prediction target is column `y`; all other columns are candidate features).\n"
 "- Model: OLS regression of `y` on features [{features}].\n"
 "- Reported performance: out-of-sample R^2 = {score} on a held-out time period.\n"
 "Investigate the data YOURSELF before deciding (you may run shell, e.g. `{py} -c \"import pandas as pd; ...\"` on the "
 "CSV). Be skeptical: a high backtest score is NOT sufficient — check WHY the model performs as it does and whether "
 "that performance will persist with real, live data at decision time.\n"
 "Output ONLY a JSON object as the very last line, nothing after it: "
 '{{"approve": true_or_false, "confidence_0_1": <float>, "key_risk": "<one short phrase>"}}')
status=open(os.path.join(RUN,"status.txt"),"w")
for arm,cfg in ARMS.items():
    prompt=TPL.format(csv=os.path.join(RUN,cfg["csv"]),features=cfg["features"],score=cfg["score"],py=PY)
    t0=time.time()
    try:
        out=subprocess.run([CLAUDE,"-p",prompt,"--dangerously-skip-permissions"],
            capture_output=True,text=True,timeout=420,cwd=RUN)
        open(os.path.join(RUN,f"{arm}.log"),"w").write(out.stdout+"\n===STDERR===\n"+out.stderr)
        status.write(f"{arm}: rc={out.returncode} dur={time.time()-t0:.0f}s len={len(out.stdout)}\n"); status.flush()
    except subprocess.TimeoutExpired:
        open(os.path.join(RUN,f"{arm}.log"),"w").write("TIMEOUT after 420s")
        status.write(f"{arm}: TIMEOUT dur={time.time()-t0:.0f}s\n"); status.flush()
status.write("RUNNER_DONE\n"); status.close()
