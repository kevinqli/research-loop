You are a careful research-verification agent. You are given 3 tasks (q1,q2,q3). For EACH task you must
decide whether a treatment T has a REAL causal effect on an outcome Y (verdict REAL or NULL).

For each task you are given, in `tasks.json`:
- `marginal_effect`/`marginal_p`: the estimated effect of T on Y from a regression Y ~ T (NOT adjusting for Z).
- `conditioned_effect`/`conditioned_p`: the estimated effect of T on Y from Y ~ T + Z (adjusting for a covariate Z).
- `shrinks`: whether adjusting for Z made |effect| smaller.
- `context`: a description of how Z was collected (its timing relative to T and its causal relationships).

Answer in TWO passes and write BOTH to `answers.json`:
PASS 1 — DATA_ONLY: using ONLY the numeric statistics (ignore `context`), give your best verdict per task and a
  confidence, and note if the statistics alone are insufficient to decide.
PASS 2 — WITH_ROLE: now ALSO use `context`. Decide whether to condition on Z (i.e., trust the conditioned estimate)
  or not (trust the marginal), then give your final verdict per task with a one-sentence causal justification.

Think about WHICH estimate (marginal vs conditioned) correctly answers "does T causally affect Y" given what Z is.

Write ONLY this JSON to a file named `answers.json` (no prose outside it):
{
 "data_only": {"q1": {"verdict":"REAL|NULL","note":"..."}, "q2": {...}, "q3": {...}},
 "with_role": {"q1": {"verdict":"REAL|NULL","condition_on_Z":true|false,"reason":"..."}, "q2": {...}, "q3": {...}}
}
