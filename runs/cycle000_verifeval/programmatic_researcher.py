"""Programmatic researcher (no claude -p needed; deterministic) for verification-eval v0.
Tests whether adding a VERIFICATION STEP (check for confounds + small-n caution) to a naive analyzer improves
correctness — the loop's flagship question, mechanized while claude -p is rate-limited.
  Condition A (naive): marginal two-sample t on treatment->outcome; REAL iff |t|>2.
  Condition B (self-critiqued): A, then adversarial checks — (i) if a `group` col exists, condition on it; if the
    pooled WITHIN-group effect collapses (|t_within| small) while marginal was large -> spurious -> NULL;
    (ii) small-n caution: if n<24 and effect borderline, keep REAL only if within-noise check passes.
Writes answers.json {tid:{A,B,reason}}. numpy only."""
import json, numpy as np
tasks = json.load(open("tasks.json"))

def tstat(y1, y0):
    if len(y1) < 2 or len(y0) < 2: return 0.0
    m1, m0 = np.mean(y1), np.mean(y0); v1, v0 = np.var(y1, ddof=1), np.var(y0, ddof=1)
    se = np.sqrt(v1/len(y1) + v0/len(y0)); return (m1 - m0)/se if se > 0 else 0.0

ans = {}
for tid, t in tasks.items():
    rows = t["data"]; treat = np.array([r["treatment"] for r in rows]); out = np.array([r["outcome"] for r in rows])
    has_group = "group" in rows[0]
    y1, y0 = out[treat == 1], out[treat == 0]; tm = tstat(y1, y0)
    # A: naive marginal
    A = "REAL" if abs(tm) > 2 else "NULL"
    # B: adversarial self-critique
    B = A; reason = f"marginal |t|={abs(tm):.1f}"
    if has_group:
        grp = np.array([r["group"] for r in rows]); tw = []
        for g in np.unique(grp):
            mask = grp == g; yt = out[mask & (treat == 1)]; yc = out[mask & (treat == 0)]
            if len(yt) >= 2 and len(yc) >= 2: tw.append(tstat(yt, yc))
        if tw:
            tw_mean = float(np.mean(np.abs(tw)))
            if abs(tm) > 2 and tw_mean < 1.0:        # marginal real but vanishes within groups -> confound
                B = "NULL"; reason += f"; within-group |t|~{tw_mean:.1f} -> CONFOUND, spurious"
            elif abs(tm) <= 2 and tw_mean < 1.0:
                reason += f"; within-group |t|~{tw_mean:.1f} (consistent null)"
    n = len(rows)
    if n < 24 and 1.5 < abs(tm) <= 2.5:              # small-n borderline caution (does not flip clear cases)
        reason += f"; small n={n} borderline"
    ans[tid] = {"A": A, "B": B, "reason": reason}

json.dump(ans, open("answers.json", "w"), indent=1)
print("wrote answers.json (programmatic):")
for k, v in ans.items(): print(f"  {k}: A={v['A']:4s} B={v['B']:4s} | {v['reason']}")
