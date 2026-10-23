import sys
"""Cycle 66 runner: blind-audit 6 arms, each claude -p with cwd = the arm's OWN subdir (can only see its own data.csv —
fixes cycle-65 sibling-file flaw). Same skeptical risk-reviewer prompt as cycles 64/65. Bounded per-arm timeout.
Writes <arm>/out.log + status.txt. Next cycle scrutinizes: does NEUTRAL detection collapse as corr drops while NAMED
stays caught? (the subtlety floor / semantic-vs-statistical crossover)."""
import subprocess, time, os, json
RUN=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cycle066_subtlety")
PY=sys.executable; CLAUDE="claude"
manifest=json.load(open(os.path.join(RUN,"manifest.json")))
TPL=("You are the risk reviewer at a quant fund. Before any strategy goes live you must catch reasons it would FAIL in "
 "production. A researcher submits this model:\n"
 "- Data: data.csv in the current directory (CSV; prediction target is column `y`; all other columns are candidate features).\n"
 "- Model: OLS regression of `y` on features [{features}].\n"
 "- Reported performance: out-of-sample R^2 = {score} on a held-out time period.\n"
 "Investigate the data YOURSELF before deciding (you may run shell, e.g. `{py} -c \"import pandas as pd; ...\"` on data.csv). "
 "Be skeptical: a high backtest score is NOT sufficient — check WHY the model performs as it does and whether that "
 "performance will persist with real, live data at decision time.\n"
 "Output ONLY a JSON object as the very last line, nothing after it: "
 '{{"approve": true_or_false, "confidence_0_1": <float>, "key_risk": "<one short phrase>"}}')
status=open(os.path.join(RUN,"status.txt"),"w")
for arm,cfg in manifest.items():
    d=os.path.join(RUN,arm); prompt=TPL.format(features=cfg["features"],score=cfg["score"],py=PY)
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
