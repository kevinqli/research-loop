"""Pillar-2 (tail) proxy: scan AIDE's 66 tree-search-CONVERGED solutions (repos/aideml/sample_results/*.py) for
validation hygiene — the structural enabler of 'hallucinated progress' (an agent optimizing a self-reported metric
that's optimistically biased). NOTE: these are FINAL solutions, not transcripts (can't see plateaus/loops directly);
this measures the validation SIGNATURE of the converged output. Categories per file:
  CV        : uses cross-validation (KFold/StratifiedKFold/cross_val/cv=/TimeSeriesSplit)
  holdout   : only a single train_test_split / validation split (no CV)
  none      : no explicit validation of its own predictions (train -> predict test)
  +leak?    : preprocessing fit on combined train+test (concat then fit / fit on full X) — classic leakage signature
Pure text scan; regex heuristics (noisy — verify a sample by eye)."""
import re, glob, os, json
D = os.environ.get("AIDEML_SAMPLE_RESULTS", "aideml/sample_results")
files = sorted(glob.glob(D + "/*.py"))
cv_re   = re.compile(r"\b(KFold|StratifiedKFold|cross_val_score|cross_validate|TimeSeriesSplit|GroupKFold)\b|cv\s*=", re.I)
hold_re = re.compile(r"\btrain_test_split\b", re.I)
leak_re = re.compile(r"concat\([^)]*test|fit_transform\(\s*(pd\.concat|np\.concatenate|all_|full_|X\b)", re.I)
cats = {"CV":[], "holdout":[], "none":[]}; leaks = []
for f in files:
    t = open(f, errors="ignore").read()
    cat = "CV" if cv_re.search(t) else ("holdout" if hold_re.search(t) else "none")
    cats[cat].append(os.path.basename(f))
    if leak_re.search(t): leaks.append(os.path.basename(f))
n = len(files)
print(f"n={n} AIDE converged solutions scanned\n")
for c in ("CV","holdout","none"):
    print(f"  {c:8s}: {len(cats[c]):2d}/{n} = {len(cats[c])/n:.0%}")
print(f"\n  preprocessing-leak signature (fit on combined train+test): {len(leaks)}/{n} = {len(leaks)/n:.0%}")
print(f"    e.g.: {leaks[:5]}")
print(f"\n  >> 'hallucinated-progress risk' = holdout+none (validation weaker than CV): "
      f"{(len(cats['holdout'])+len(cats['none']))}/{n} = {(len(cats['holdout'])+len(cats['none']))/n:.0%}")
json.dump({"n":n, "CV":len(cats["CV"]), "holdout":len(cats["holdout"]), "none":len(cats["none"]),
           "leak_signature":len(leaks)}, open("score.json","w"), indent=1)
print("\nwrote score.json")
