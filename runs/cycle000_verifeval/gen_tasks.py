"""verification-eval v0 — generate small 'is the effect real?' tasks with KNOWN ground truth.
Ground truth is set by construction. Includes clear-real, clear-null, an underpowered-real, and a CONFOUND trap
(marginal association is spurious; vanishes when you condition on `group`) — the case self-critique should catch.
Writes tasks.json (what the researcher sees) + ground_truth.json (hidden). CPU, deterministic (seeded)."""
import json, numpy as np
rng = np.random.default_rng(0)

def fmt(rows): return [ {k: (round(float(v),2) if isinstance(v,(int,float,np.floating)) else v) for k,v in r.items()} for r in rows ]

tasks = {}; truth = {}

def make(tid, n, effect, with_group=False, confound=False, note=""):
    rows = []
    if confound:
        # group drives BOTH treatment assignment and outcome -> spurious marginal association; truth = NULL
        for _ in range(n):
            g = int(rng.integers(0,2))
            treat = int(rng.random() < (0.8 if g==1 else 0.2))      # group->treatment
            out = 50 + 10*g + rng.normal(0,3)                        # group->outcome, treatment has NO direct effect
            rows.append({"treatment": treat, "outcome": out, "group": g})
        gt = "NULL"
    else:
        for _ in range(n):
            treat = int(rng.integers(0,2))
            out = 50 + effect*treat + rng.normal(0,5)
            r = {"treatment": treat, "outcome": out}
            if with_group: r["group"] = int(rng.integers(0,2))      # irrelevant group
            rows.append(r)
        gt = "REAL" if effect != 0 else "NULL"
    tasks[tid] = {"question": "Does `treatment` have a real effect on `outcome`? Answer REAL or NULL.", "data": fmt(rows)}
    truth[tid] = {"ground_truth": gt, "note": note}

make("t0", 40,  8.0,  note="clear real (large effect)")
make("t1", 60,  5.0,  note="real, moderate effect")
make("t2", 40,  0.0,  note="clear null")
make("t3", 60,  0.0,  with_group=True, note="null with irrelevant group column (distractor)")
make("t4", 16,  4.0,  note="real but UNDERPOWERED (small n) — genuinely ambiguous")
make("t5", 60,  0.0,  confound=True, note="CONFOUND TRAP: marginal assoc spurious; condition on group -> null")

json.dump(tasks, open("tasks.json","w"), indent=1)
json.dump(truth, open("ground_truth.json","w"), indent=1)
print(f"wrote {len(tasks)} tasks. truth = " + ", ".join(f"{k}:{v['ground_truth']}" for k,v in truth.items()))
