"""Cycle 15: score the seed-1 LLM researcher vs ground truth + mechanical baselines, and check the
collider-aware conditioning DECISIONS replicate (the cycle-5 headline; this graduates it to n=2).
Robust to ground_truth.json being either {tid: gt} or {tid: [gt, note]}. numpy/json only. MPS-exploratory."""
import json

D = "runs/cycle014_llm_seed1/"
gt_raw = json.load(open(D + "ground_truth.json"))
truth = {k: (v[0] if isinstance(v, (list, tuple)) else v) for k, v in gt_raw.items()}
ans = json.load(open(D + "researcher_view/answers.json"))
mech = json.load(open(D + "mechanical.json"))
tids = sorted(truth, key=lambda t: int(t[1:]))

# accuracy of each verifier
def acc(pred):  # pred: {tid: verdict}
    c = sum(pred[t] == truth[t] for t in tids)
    return c, len(tids), c / len(tids)

llm_pred = {t: ans[t]["verdict"] for t in tids}
mech_A = {t: mech[t]["A"] for t in tids}
mech_Bs = {t: mech[t]["B_sound"] for t in tids}
mech_Bc = {t: mech[t]["B_collider"] for t in tids}

print("=== SEED-1 ACCURACY (n=8 tasks) ===")
for name, p in [("LLM agent", llm_pred), ("A no-check", mech_A),
                ("B_sound always-cond", mech_Bs), ("B_collider adaptive", mech_Bc)]:
    c, n, a = acc(p)
    print(f"  {name:24s}: {c}/{n} = {a:.3f}")

# CONDITIONING-DECISION check: did LLM condition exactly where collider-aware rule does?
print("\n=== CONDITIONING DECISIONS (the cycle-5 headline) ===")
print(f"  {'tid':4s} {'truth':5s} {'LLM verdict':12s} {'LLM cond_on':12s} {'B_collider':11s} {'match-cond?'}")
llm_cond = {}
match = 0
for t in tids:
    lc = ans[t]["conditioned_on"]
    llm_cond[t] = lc
    # B_collider conditions iff its verdict differs from A (it adjusted) -> infer decision
    bc_conditioned = mech[t]["A"] != mech[t]["B_collider"]
    llm_conditioned = (lc not in ("none", "", None))
    ok = (bc_conditioned == llm_conditioned)
    match += ok
    print(f"  {t:4s} {truth[t]:5s} {llm_pred[t]:12s} {str(lc):12s} "
          f"{mech[t]['B_collider']:11s} {'YES' if ok else 'no':4s} "
          f"(LLM cond={llm_conditioned}, Bc cond={bc_conditioned})")
print(f"\n  conditioning-decision agreement LLM vs collider-aware rule: {match}/{len(tids)}")

# where LLM erred, classify (conservative false-neg vs conditioning error)
errs = [t for t in tids if llm_pred[t] != truth[t]]
print(f"\n=== LLM ERRORS: {errs} ===")
for t in errs:
    kind = "FALSE-NEG (called REAL->NULL, conservative)" if truth[t] == "REAL" else "FALSE-POS"
    print(f"  {t}: truth={truth[t]} pred={llm_pred[t]} [{kind}] reason={ans[t]['reason']}")

out = {
    "seed": 1,
    "llm_acc": acc(llm_pred)[2],
    "A_acc": acc(mech_A)[2], "Bsound_acc": acc(mech_Bs)[2], "Bcollider_acc": acc(mech_Bc)[2],
    "cond_decision_agreement": f"{match}/{len(tids)}",
    "llm_errors": {t: {"truth": truth[t], "pred": llm_pred[t]} for t in errs},
    "llm_conditioned_on": llm_cond,
}
json.dump(out, open(D + "score_seed1.json", "w"), indent=1)
print("\nwrote " + D + "score_seed1.json")
print("DONE")
