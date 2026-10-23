import json
import numpy as np
from scipy import stats

with open('./tasks.json', 'r') as f:
    tasks = json.load(f)

results = {}

for task_id, task_data in tasks.items():
    data = task_data['data']

    # Extract treatment and outcome
    treatment = np.array([d['treatment'] for d in data])
    outcome = np.array([d['outcome'] for d in data])

    # Basic stats
    t1_idx = treatment == 1.0
    t0_idx = treatment == 0.0

    mean_t1 = outcome[t1_idx].mean()
    mean_t0 = outcome[t0_idx].mean()
    sd_t1 = outcome[t1_idx].std()
    sd_t0 = outcome[t0_idx].std()
    n_t1 = t1_idx.sum()
    n_t0 = t0_idx.sum()

    # Two-sample t-test
    t_stat, p_val = stats.ttest_ind(outcome[t1_idx], outcome[t0_idx])

    # Effect size (Cohen's d)
    pooled_sd = np.sqrt(((n_t1-1)*sd_t1**2 + (n_t0-1)*sd_t0**2) / (n_t1+n_t0-2))
    cohens_d = (mean_t1 - mean_t0) / pooled_sd if pooled_sd > 0 else 0

    print(f"\n{'='*60}")
    print(f"Task {task_id}:")
    print(f"N: {len(data)} (treatment=1: {n_t1}, treatment=0: {n_t0})")
    print(f"Mean(treatment=1): {mean_t1:.2f}, SD: {sd_t1:.2f}")
    print(f"Mean(treatment=0): {mean_t0:.2f}, SD: {sd_t0:.2f}")
    print(f"Difference: {mean_t1 - mean_t0:.2f}")
    print(f"Cohen's d: {cohens_d:.3f}")
    print(f"t-statistic: {t_stat:.3f}, p-value: {p_val:.4f}")

    # Check for group variable (confounder)
    if 'group' in data[0]:
        group = np.array([d['group'] for d in data])
        print(f"\n--- CONFOUNDER CHECK (group variable present) ---")

        # Within-group analysis
        for g_val in np.unique(group):
            g_idx = group == g_val
            t1_g_idx = (treatment == 1.0) & (g_idx)
            t0_g_idx = (treatment == 0.0) & (g_idx)

            if t1_g_idx.sum() > 0 and t0_g_idx.sum() > 0:
                mean_t1_g = outcome[t1_g_idx].mean()
                mean_t0_g = outcome[t0_g_idx].mean()
                n_t1_g = t1_g_idx.sum()
                n_t0_g = t0_g_idx.sum()

                t_stat_g, p_val_g = stats.ttest_ind(outcome[t1_g_idx], outcome[t0_g_idx])

                print(f"Group={g_val:.0f}: N_t1={n_t1_g}, N_t0={n_t0_g}")
                print(f"  Mean(t1)={mean_t1_g:.2f}, Mean(t0)={mean_t0_g:.2f}, diff={mean_t1_g-mean_t0_g:.2f}")
                print(f"  t={t_stat_g:.3f}, p={p_val_g:.4f}")

        # Check if treatment and group are confounded
        treatment_by_group = {}
        for g_val in np.unique(group):
            g_idx = group == g_val
            t1_prop = (treatment[g_idx] == 1.0).mean()
            treatment_by_group[f"group_{g_val:.0f}"] = t1_prop
        print(f"Treatment=1 proportion by group: {treatment_by_group}")

    # Quick verdict
    verdict_a = "REAL" if p_val < 0.05 else "NULL"

    print(f"\nQuick Verdict (A): {verdict_a}")

    results[task_id] = {
        'mean_t1': mean_t1,
        'mean_t0': mean_t0,
        'diff': mean_t1 - mean_t0,
        'cohens_d': cohens_d,
        't_stat': t_stat,
        'p_val': p_val,
        'has_group': 'group' in data[0],
        'verdict_a': verdict_a
    }

print(f"\n\n{'='*60}")
print("SUMMARY FOR VERDICTS:")
print('='*60)
for task_id in sorted(results.keys()):
    r = results[task_id]
    print(f"{task_id}: p={r['p_val']:.4f}, d={r['cohens_d']:.3f}, group={r['has_group']}, A={r['verdict_a']}")
