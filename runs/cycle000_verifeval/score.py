"""Score researcher answers (answers.json: {tid:{A:REAL/NULL, B:REAL/NULL}}) vs hidden ground_truth.json.
Reports accuracy of condition A (direct) vs B (after self-critique), the overturn rate (B!=A), and whether
overturns HELPED (A wrong->B right) or HURT (A right->B wrong) — the core verification-eval metric."""
import json, sys
truth = {k: v["ground_truth"] for k, v in json.load(open("ground_truth.json")).items()}
ans = json.load(open(sys.argv[1] if len(sys.argv) > 1 else "answers.json"))
def norm(x): return str(x).strip().upper()
n = len(truth); accA = accB = overt = helped = hurt = 0; rows = []
for tid, gt in truth.items():
    a = norm(ans.get(tid, {}).get("A", "?")); b = norm(ans.get(tid, {}).get("B", a))
    cA, cB = (a == gt), (b == gt); accA += cA; accB += cB
    flip = (a != b)
    if flip:
        overt += 1
        if (not cA) and cB: helped += 1
        if cA and (not cB): hurt += 1
    rows.append((tid, gt, a, b, "ok" if cB else "WRONG", "flip" if flip else ""))
print(f"n={n}")
print(f"acc_A (direct)        = {accA}/{n} = {accA/n:.2f}")
print(f"acc_B (self-critiqued)= {accB}/{n} = {accB/n:.2f}")
print(f"overturns (B!=A)      = {overt}  (helped {helped}, hurt {hurt})")
print(f"net effect of self-critique = {accB-accA:+d} correct")
print("--- per task ---")
for tid, gt, a, b, v, f in rows: print(f"  {tid}: truth={gt:4s} A={a:4s} B={b:4s} {v} {f}")
json.dump({"n": n, "acc_A": accA/n, "acc_B": accB/n, "overturns": overt, "helped": helped, "hurt": hurt,
           "net": accB-accA}, open("score.json","w"), indent=1)
print("\nwrote score.json")
