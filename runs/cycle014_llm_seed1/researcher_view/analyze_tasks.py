import json
import numpy as np
from scipy import stats

with open('tasks.json') as f:
    tasks = json.load(f)

results = {}

for task_id, task_data in tasks.items():
    data = task_data['data']

    # Extract variables
    treatments = [d['treatment'] for d in data]
    outcomes = [d['outcome'] for d in data]
    has_group = 'group' in data[0]

    if has_group:
        groups = [d['group'] for d in data]

    # Marginal effect: treatment=1 vs treatment=0 (no stratification)
    t0_outcomes = [outcomes[i] for i in range(len(outcomes)) if treatments[i] == 0.0]
    t1_outcomes = [outcomes[i] for i in range(len(outcomes)) if treatments[i] == 1.0]

    mean_t0 = np.mean(t0_outcomes)
    mean_t1 = np.mean(t1_outcomes)
    marginal_effect = mean_t1 - mean_t0

    # T-test on marginal effect
    t_stat, p_val = stats.ttest_ind(t1_outcomes, t0_outcomes)

    verdict = "REAL" if p_val < 0.05 else "NULL"
    conditioned_on = "none"
    reason = f"Marginal: T1 mean={mean_t1:.2f}, T0 mean={mean_t0:.2f}, diff={marginal_effect:.2f}, p={p_val:.3f}"

    # If group variable exists, stratify and check if adjustment changes the effect
    if has_group:
        print(f"\n{task_id}: Has group variable")

        # Group-specific effects
        for g in [0.0, 1.0]:
            idx_g = [i for i in range(len(groups)) if groups[i] == g]
            t0_g = [outcomes[i] for i in idx_g if treatments[i] == 0.0]
            t1_g = [outcomes[i] for i in idx_g if treatments[i] == 1.0]

            if len(t0_g) > 0 and len(t1_g) > 0:
                mean_t0_g = np.mean(t0_g)
                mean_t1_g = np.mean(t1_g)
                effect_g = mean_t1_g - mean_t0_g
                print(f"  Group={g}: T1 mean={mean_t1_g:.2f}, T0 mean={mean_t0_g:.2f}, effect={effect_g:.2f}")

        # Check if group confounds the treatment
        # Compute weighted average of within-group effects
        effects_within = []
        for g in [0.0, 1.0]:
            idx_g = [i for i in range(len(groups)) if groups[i] == g]
            t0_g = [outcomes[i] for i in idx_g if treatments[i] == 0.0]
            t1_g = [outcomes[i] for i in idx_g if treatments[i] == 1.0]

            if len(t0_g) > 0 and len(t1_g) > 0:
                effect_g = np.mean(t1_g) - np.mean(t0_g)
                n_g = len(idx_g)
                effects_within.append((effect_g, n_g))

        if effects_within:
            # Weighted average of within-group effects
            pooled_effect = sum(e * n for e, n in effects_within) / sum(n for _, n in effects_within)

            print(f"  Marginal effect: {marginal_effect:.2f}")
            print(f"  Pooled within-group effect: {pooled_effect:.2f}")

            # Does adjusting shrink the effect? (confounding)
            if abs(pooled_effect) < abs(marginal_effect):
                print(f"  -> Adjusting SHRINKS effect: likely confounder")
                conditioned_on = "group"
                verdict = "REAL" if abs(pooled_effect) > 0.5 else "NULL"  # Use within-group effect
                reason = f"Confounding detected. Within-group effect={pooled_effect:.2f}, marginal={marginal_effect:.2f}"
            # Does adjusting grow the effect or flip sign? (collider bias)
            elif abs(pooled_effect) > abs(marginal_effect) or (marginal_effect * pooled_effect < 0):
                print(f"  -> Adjusting GROWS/flips effect: collider bias warning")
                conditioned_on = "none"
                reason = f"Collider bias detected (don't adjust). Marginal={marginal_effect:.2f}, within={pooled_effect:.2f}"
            else:
                print(f"  -> Similar effects, no clear confounding")
                conditioned_on = "none"

    results[task_id] = {
        "verdict": verdict,
        "conditioned_on": conditioned_on,
        "reason": reason
    }

    print(f"{task_id}: {verdict}, conditioned_on={conditioned_on}")
    print(f"  {reason}")

# Write answers
with open('answers.json', 'w') as f:
    json.dump(results, f, indent=2)

print("\n\nFinal answers written to answers.json")
